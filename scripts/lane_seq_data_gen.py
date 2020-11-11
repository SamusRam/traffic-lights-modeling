import os
from map_traffic_lights_data import *
from torch.utils.data import DataLoader
from l5kit_modified.l5kit_modified import FramesDataset
from l5kit.data import LocalDataManager, ChunkedDataset
from datetime import datetime
from pytz import timezone
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--timestamp-min', default='')
parser.add_argument('--timestamp-max', default='')
parser.add_argument('--fold-i', default=0, type=int)
parser.add_argument('--zarr-basename', default='train_full')

args = parser.parse_args()

timestamp_min = args.timestamp_min
timestamp_max = args.timestamp_max
fold_i = args.fold_i
zarr_basename = args.zarr_basename

# Jan 17 for train_full split
if timestamp_min == '':
    timestamp_min = datetime(1970, 2, 2).astimezone(timezone('US/Pacific'))
else:
    timestamp_min = datetime.strptime(timestamp_min, '%Y-%m-%d').astimezone(timezone('US/Pacific'))
if timestamp_max == '':
    timestamp_max = datetime(2021, 11, 20).astimezone(timezone('US/Pacific'))
else:
    timestamp_max = datetime.strptime(timestamp_max, '%Y-%m-%d').astimezone(timezone('US/Pacific'))
print('='*20)
print('start: ', timestamp_min, 'end: ', timestamp_max)
print('='*20)
os.environ["L5KIT_DATA_FOLDER"] = "../input/"
dm = LocalDataManager()
dataset_path = dm.require(f'scenes/{zarr_basename}_filtered_min_frame_history_4_min_frame_future_1_with_mask_idx.zarr')
frame_dataset = FramesDataset(dataset_path, agents_from_standard_mask_only=True, return_indices=True)
dataloader_frames = DataLoader(frame_dataset, shuffle=False, batch_size=32,
                               num_workers=12,
                               collate_fn=partial(agent_lanes_collate_fn, timestamp_min=timestamp_min, timestamp_max=timestamp_max))
agent_lanes_df = get_agent_lanes_df(dataloader_frames)
agent_lanes_df.to_hdf(f'../input/agent_lane_seq_df_{zarr_basename}_{fold_i}.hdf5', key='data')
