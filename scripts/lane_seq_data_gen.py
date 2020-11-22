import os
from map_traffic_lights_data import *
from torch.utils.data import DataLoader
from l5kit_modified.l5kit_modified import FramesDataset
from l5kit.data import LocalDataManager, ChunkedDataset
from datetime import datetime
from pytz import timezone
from functools import partial
import argparse
import numpy as np
import bisect
import zarr
from tqdm.auto import tqdm
import logging
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--timestamp-min', default='')
parser.add_argument('--timestamp-max', default='')
parser.add_argument('--splitting-dates', nargs='*', action='append', default=[]) # 2019-11-03 2019-11-17 2019-11-28 2019-12-14 2020-01-01 2020-01-17 2020-02-03 2020-02-17 2020-02-27
parser.add_argument('--fold-i', default=0, type=int)
parser.add_argument('--zarr-basename', default='train_full')
parser.add_argument('--traffic-light-predictions-base-name', default='train_full')

args = parser.parse_args()

timestamp_min = args.timestamp_min
timestamp_max = args.timestamp_max
fold_i = args.fold_i
zarr_basename = args.zarr_basename
tl_predictions_base_name = args.traffic_light_predictions_base_name
splitting_dates = args.splitting_dates[0] if len(args.splitting_dates) else []
ZARR_CHUNK_SIZE = 10_000

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(f'Lane Seq Data Gen ({zarr_basename})')


if timestamp_min == '':
    timestamp_min = datetime(1970, 2, 2).astimezone(timezone('US/Pacific'))
else:
    timestamp_min = datetime.strptime(timestamp_min, '%Y-%m-%d').astimezone(timezone('US/Pacific'))
if timestamp_max == '':
    timestamp_max = datetime(2021, 11, 20).astimezone(timezone('US/Pacific'))
else:
    timestamp_max = datetime.strptime(timestamp_max, '%Y-%m-%d').astimezone(timezone('US/Pacific'))
splitting_dates = [datetime.strptime(date, '%Y-%m-%d').astimezone(timezone('US/Pacific')) for date in splitting_dates]
boarder_dates_all = [timestamp_min] + sorted(splitting_dates) + [timestamp_max]
# converting to lyft timestamps
boarder_timestamps_all = [ts.timestamp()* 10**9 for ts in boarder_dates_all]

assert boarder_timestamps_all[0] < boarder_timestamps_all[1] and boarder_timestamps_all[-1] > boarder_timestamps_all[-2]

logger.info(f'start: {timestamp_min:%Y-%m-%d}, end: {timestamp_max:%Y-%m-%d}')
logger.info('splitting dates: ' + ', '.join([f'{date:%Y-%m-%d}' for date in splitting_dates]))

os.environ["L5KIT_DATA_FOLDER"] = "../input/"
dm = LocalDataManager()
dataset_path = dm.require(f'scenes/{zarr_basename}_filtered_min_frame_history_4_min_frame_future_1_with_mask_idx.zarr'
                          if zarr_basename != 'test' else 'scenes/test.zarr')
frame_dataset = FramesDataset(dataset_path,
                              agents_from_standard_mask_only=zarr_basename != 'test',
                              return_indices=True)
intersection_2_predictions = get_traffic_light_predictions_per_intersection(tl_predictions_base_name)
dataloader_frames = DataLoader(frame_dataset, shuffle=False, batch_size=64,
                               num_workers=12,
                               collate_fn=partial(agent_lanes_collate_fn,
                                                  intersection_2_predictions=intersection_2_predictions,
                                                  timestamp_min=timestamp_min,
                                                  timestamp_max=timestamp_max))
output_dataset_path = f'../input/agent_lane_seq_df_{zarr_basename}_{fold_i}.zarr'


###### creating separate zarr datasets for smaller time intervals, structural numpy arrays inside
LANE_SEQ_DTYPE = [
    ("agent_centroid", np.float64, (2,)),
    ("agent_track_id", np.int64),
    ("scene_idx", np.int64),
    ("timestamp", np.int64),
    ("map_segment_group", np.int8),
    ("agent_speed", np.float64),
    ("agent_yaw", np.float64),
    ("lane_id", "<U16"),
    ("lane_point_i", np.int8),
    ("remaining_speed_lim", np.float64),
    ("green_prob", np.float64),
    ("tl_tte_mode", np.float64),
    ("tl_tte_25th_perc", np.float64),
    ("tl_tte_75th_perc", np.float64),
    ("next_car_lane_points_dist", np.float64),
    ("next_car_dist", np.float64),
    ("next_car_closing_speed", np.float64),
    ("yield_closest_dist", np.float64),
    ("yield_speed_of_closest", np.float64)
]

zarr_dataset_root = zarr.open_group(output_dataset_path, mode='w')

# estimating sizes of each time-interval group
time_interval_size = [0 for _ in range(len(boarder_timestamps_all) - 1)]
for frame in tqdm(frame_dataset.zarr_root['frames'], 'Computing time interval size upper bounds..'):
    timestamp = frame['timestamp']
    time_interval_idx = bisect.bisect_right(boarder_timestamps_all, timestamp) - 1
    if 0 <= time_interval_idx < len(boarder_timestamps_all) - 1:
        if zarr_basename != 'test':
            agent_idx_start, agent_idx_end = frame['mask_agent_index_interval']
        else:
            agent_idx_start, agent_idx_end = frame['agent_index_interval']
        time_interval_size[time_interval_idx] += agent_idx_end - agent_idx_start


logger.info(f'time_interval_sizes: {time_interval_size}')

zarr_keys = []
zarr_key_2_dataset = dict()
# buffers for more efficient write (by chunks)
timeinterval_i_2_write_buffer = [np.ndarray(shape=(ZARR_CHUNK_SIZE,), dtype=LANE_SEQ_DTYPE)
                                 for _ in range(len(time_interval_size))]
timeinterval_i_2_write_buffer_size = [0 for _ in range(len(time_interval_size))]
timeinterval_i_2_write_buffer_chunk_i = [0 for _ in range(len(time_interval_size))]
lyft_ts_2_datetime = lambda ts: datetime.fromtimestamp(ts / 10 ** 9).astimezone(timezone('US/Pacific'))
for key_i in range(len(boarder_timestamps_all) - 1):
    start_date = lyft_ts_2_datetime(boarder_timestamps_all[key_i])
    end_date = lyft_ts_2_datetime(boarder_timestamps_all[key_i + 1])
    zarr_key_name = f'{start_date:%Y_%m_%d}__{end_date:%Y_%m_%d}'
    zarr_keys.append(zarr_key_name)
    zarr_key_2_dataset[zarr_key_name] = zarr_dataset_root.require_dataset(zarr_key_name,
                                                                          dtype=LANE_SEQ_DTYPE,
                                                                          chunks=(ZARR_CHUNK_SIZE,),
                                                                          shape=(time_interval_size[key_i],))


# some scenes would be splitted by the time split of the full train, but it should be negligible
for batch in tqdm(dataloader_frames, desc='Agents info events...'):
    for record in batch:
        timestamp = record[3]
        time_interval_idx = bisect.bisect_right(boarder_timestamps_all, timestamp) - 1
        next_entry_i = timeinterval_i_2_write_buffer_size[time_interval_idx]
        for dtype_entry, val in zip(LANE_SEQ_DTYPE, record):
            timeinterval_i_2_write_buffer[time_interval_idx][next_entry_i][dtype_entry[0]] = val
        timeinterval_i_2_write_buffer_size[time_interval_idx] += 1

        start_date = lyft_ts_2_datetime(boarder_timestamps_all[time_interval_idx])
        end_date = lyft_ts_2_datetime(boarder_timestamps_all[time_interval_idx + 1])
        zarr_key_name = f'{start_date:%Y_%m_%d}__{end_date:%Y_%m_%d}'
        dataset_i_start = timeinterval_i_2_write_buffer_chunk_i[time_interval_idx] * ZARR_CHUNK_SIZE
        dataset_i_end = (timeinterval_i_2_write_buffer_chunk_i[time_interval_idx] + 1) * ZARR_CHUNK_SIZE

        if timeinterval_i_2_write_buffer_size[time_interval_idx] == ZARR_CHUNK_SIZE:
            timeinterval_i_2_write_buffer_size[time_interval_idx] = 0
            start_date = lyft_ts_2_datetime(boarder_timestamps_all[time_interval_idx])
            end_date = lyft_ts_2_datetime(boarder_timestamps_all[time_interval_idx + 1])
            zarr_key_name = f'{start_date:%Y_%m_%d}__{end_date:%Y_%m_%d}'
            dataset_i_start = timeinterval_i_2_write_buffer_chunk_i[time_interval_idx] * ZARR_CHUNK_SIZE
            dataset_i_end = (timeinterval_i_2_write_buffer_chunk_i[time_interval_idx] + 1) * ZARR_CHUNK_SIZE
            zarr_key_2_dataset[zarr_key_name][dataset_i_start:
                                              dataset_i_end] = timeinterval_i_2_write_buffer[time_interval_idx]

            timeinterval_i_2_write_buffer_chunk_i[time_interval_idx] += 1


# flush buffers
for timeinterval_i, zarr_key_name in enumerate(zarr_keys):
    dataset_i_start = timeinterval_i_2_write_buffer_chunk_i[timeinterval_i] * ZARR_CHUNK_SIZE
    dataset_i_end = (timeinterval_i_2_write_buffer_chunk_i[timeinterval_i] + 1) * ZARR_CHUNK_SIZE
    zarr_key_2_dataset[zarr_key_name][dataset_i_start:dataset_i_end] = timeinterval_i_2_write_buffer[timeinterval_i]

    # record true sizes
    buff_size = timeinterval_i_2_write_buffer_size[timeinterval_i]
    total_size = dataset_i_start + buff_size
    with open(f'../input/{output_dataset_path.replace(".", "_")}_zarr_key_name_size.pkl', 'wb') as f:
        pickle.dump(total_size, f)
