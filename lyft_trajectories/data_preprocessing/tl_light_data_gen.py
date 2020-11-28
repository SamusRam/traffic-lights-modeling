import os
from .map_traffic_lights_data import (
    tl_seq_collate_fn,
    get_tl_events_df,
    compute_tl_signal_classes,
    compute_time_to_tl_change,
    compute_rnn_inputs,
)
from torch.utils.data import DataLoader
from ..utils.l5kit_modified.l5kit_modified import FramesDataset
from l5kit.data import LocalDataManager
from datetime import datetime
from pytz import timezone
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-name", default="train_full")
parser.add_argument("--timestamp-min", default="")
parser.add_argument("--timestamp-max", default="")
parser.add_argument("--fold-i", default=0, type=int)

args = parser.parse_args()
input_name = args.input_name
timestamp_min = args.timestamp_min
timestamp_max = args.timestamp_max
fold_i = args.fold_i

if timestamp_min == "":
    timestamp_min = datetime(2017, 2, 2).astimezone(timezone("US/Pacific"))
else:
    timestamp_min = datetime.strptime(timestamp_min, "%Y-%m-%d").astimezone(
        timezone("US/Pacific")
    )
if timestamp_max == "":
    timestamp_max = datetime(2021, 11, 20).astimezone(timezone("US/Pacific"))
else:
    timestamp_max = datetime.strptime(timestamp_max, "%Y-%m-%d").astimezone(
        timezone("US/Pacific")
    )
print("=" * 20)
print("start: ", timestamp_min, "end: ", timestamp_max)
print("=" * 20)
os.environ["L5KIT_DATA_FOLDER"] = "input/"
dm = LocalDataManager()
dataset_path = dm.require(
    f"scenes/{input_name}_filtered_min_frame_history_4_min_frame_future_1_with_mask_idx.zarr"
)

frame_dataset = FramesDataset(dataset_path, return_indices=True)
dataloader_frames = DataLoader(
    frame_dataset,
    shuffle=False,
    batch_size=32,
    num_workers=12,
    collate_fn=partial(
        tl_seq_collate_fn, timestamp_min=timestamp_min, timestamp_max=timestamp_max
    ),
)
tl_events_df = get_tl_events_df(dataloader_frames)
compute_tl_signal_classes(tl_events_df)
compute_time_to_tl_change(tl_events_df)
compute_rnn_inputs(tl_events_df)
tl_events_df[
    [
        "scene_idx",
        "frame_idx",
        "master_intersection_idx",
        "timestamp",
        "rnn_inputs_raw",
        "tl_signal_classes",
        "time_to_tl_change",
    ]
].to_hdf(f"input/tl_events_df_{input_name}_{fold_i}.hdf5", key="data")
