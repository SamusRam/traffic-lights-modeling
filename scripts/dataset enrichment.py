import sys

import zarr

sys.path.insert(0, '../scripts')
import os

os.environ["L5KIT_DATA_FOLDER"] = "../input/"
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.data.zarr_dataset import *
from l5kit_modified.l5kit_modified import get_agent_indices_set
import numpy as np
from tqdm.auto import tqdm
import argparse

MIN_FRAME_HISTORY = 10
MIN_FRAME_FUTURE = 1

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path')
parser.add_argument('--min-frame-history', default=MIN_FRAME_HISTORY, type=int)
parser.add_argument('--min-frame-future', default=MIN_FRAME_FUTURE, type=int)
parser.add_argument('--add-standard-mask-indices', default=False, action='store_true')
args = parser.parse_args()

dataset_path = args.dataset_path
min_frame_history = args.min_frame_history
min_frame_future = args.min_frame_future
add_standard_mask_indices = args.add_standard_mask_indices

dm = LocalDataManager(None)
dataset_zarr = ChunkedDataset(dm.require(dataset_path)).open(cached=False)
agent_indices_set = get_agent_indices_set(dataset_zarr, min_frame_history=min_frame_history,
                                          min_frame_future=min_frame_future, filter_agents_threshold=0.5)

print('len agents', len(agent_indices_set))

if add_standard_mask_indices:
    mask_agent_indices_set = get_agent_indices_set(dataset_zarr, min_frame_history=MIN_FRAME_HISTORY,
                                                   min_frame_future=MIN_FRAME_FUTURE, filter_agents_threshold=0.5)
    num_agents_mask = len(mask_agent_indices_set)
    print('len agents masked', len(mask_agent_indices_set))

num_scenes = len(dataset_zarr.scenes)
num_frames = len(dataset_zarr.frames)
num_agents = len(dataset_zarr.agents)
num_agents_new = len(agent_indices_set)
num_tl_faces = len(dataset_zarr.tl_faces)

new_path = f"{dm.require(dataset_path).split('.zarr')[0]}_filtered_min_frame_history_{min_frame_history}_min_frame_future_{min_frame_future}{'_with_mask_idx' if add_standard_mask_indices else ''}.zarr"

root = zarr.open_group(new_path, mode='w')

if add_standard_mask_indices:
    FRAME_DTYPE.append(("mask_agent_index_interval", np.int64, (2,)))
    MASK_AGENT_INDICES_ARRAY_KEY = 'mask_agent_indices'
    MASK_AGENT_INDICES_ARRAY_DTYPE = [("agent_index", np.int64)]
    MASK_AGENT_INDICES_ARRAY_CHUNK_SIZE = AGENT_CHUNK_SIZE

    mask_agent_indices = root.require_dataset(
        MASK_AGENT_INDICES_ARRAY_KEY, dtype=MASK_AGENT_INDICES_ARRAY_DTYPE, chunks=MASK_AGENT_INDICES_ARRAY_CHUNK_SIZE,
        shape=num_agents_mask
    )

frames = root.require_dataset(
    FRAME_ARRAY_KEY, dtype=FRAME_DTYPE, chunks=FRAME_CHUNK_SIZE, shape=(num_frames,)
)
agents = root.require_dataset(
    AGENT_ARRAY_KEY, dtype=AGENT_DTYPE, chunks=AGENT_CHUNK_SIZE, shape=(num_agents_new,)
)
scenes = root.require_dataset(
    SCENE_ARRAY_KEY, dtype=SCENE_DTYPE, chunks=SCENE_CHUNK_SIZE, shape=(num_scenes,)
)
tl_faces = root.require_dataset(
    TL_FACE_ARRAY_KEY, dtype=TL_FACE_DTYPE, chunks=TL_FACE_CHUNK_SIZE, shape=(num_tl_faces,)
)

# traffic lights
for tl_i_step in tqdm(range(0, num_tl_faces, TL_FACE_CHUNK_SIZE[0]), desc='Traffic lights..'):
    upper_idx = min(tl_i_step + TL_FACE_CHUNK_SIZE[0], num_tl_faces)
    root.traffic_light_faces[tl_i_step: upper_idx] = dataset_zarr.tl_faces[tl_i_step: upper_idx]

# scenes
for i_step in tqdm(range(0, num_scenes, SCENE_CHUNK_SIZE[0]), desc='Scenes..'):
    upper_idx = min(i_step + SCENE_CHUNK_SIZE[0], num_scenes)
    root.scenes[i_step: upper_idx] = dataset_zarr.scenes[i_step: upper_idx]

agent_count = 0
if add_standard_mask_indices:
    mask_agent_count = 0

for i_step in tqdm(range(0, num_frames, FRAME_CHUNK_SIZE[0]), desc='Frames..'):
    upper_idx = min(i_step + FRAME_CHUNK_SIZE[0], num_frames)
    batch_read = dataset_zarr.frames[i_step: upper_idx]
    if add_standard_mask_indices:
        batch_read_ = np.ndarray(batch_read.shape, dtype=FRAME_DTYPE)
        for col_name in batch_read.dtype.fields.keys():
            batch_read_[col_name] = batch_read[col_name]
        batch_read = batch_read_
    for frame_i in range(i_step, upper_idx):
        original_agent_index_interval = batch_read[frame_i - i_step]['agent_index_interval'].copy()
        agent_indices_remained = [agent_index for agent_index in range(*original_agent_index_interval)
                                  if agent_index in agent_indices_set]
        batch_read[frame_i - i_step]['agent_index_interval'] = np.array(
            [agent_count, agent_count + len(agent_indices_remained)])

        if add_standard_mask_indices:
            mask_agent_indices_remained = [agent_index for agent_index in range(*original_agent_index_interval)
                                           if agent_index in mask_agent_indices_set]
            batch_read_[frame_i - i_step]['mask_agent_index_interval'] = np.array(
                [mask_agent_count, mask_agent_count + len(mask_agent_indices_remained)])
            mask_agent_count += len(mask_agent_indices_remained)

        agent_count += len(agent_indices_remained)
    root.frames[i_step: upper_idx] = batch_read

# agents
agents_chunk_buffers = np.ndarray(shape=AGENT_CHUNK_SIZE, dtype=AGENT_DTYPE)
agents_chunk_buffer_size = 0
agents_chunk_i = 0

if add_standard_mask_indices:
    mask_agent_indices_chunk_buffers = np.ndarray(shape=MASK_AGENT_INDICES_ARRAY_CHUNK_SIZE,
                                                  dtype=MASK_AGENT_INDICES_ARRAY_DTYPE)
    mask_agent_indices_chunk_buffer_size = 0
    mask_agent_indices_chunk_i = 0

for i_step in tqdm(range(0, num_agents, AGENT_CHUNK_SIZE[0]), desc='Agents..'):
    upper_idx = min(i_step + AGENT_CHUNK_SIZE[0], num_agents)
    batch_read = dataset_zarr.agents[i_step: upper_idx]

    for agent_i in range(i_step, upper_idx):
        if agent_i in agent_indices_set:
            agents_chunk_buffers[agents_chunk_buffer_size] = batch_read[agent_i - i_step]
            if add_standard_mask_indices and agent_i in mask_agent_indices_set:
                mask_agent_indices_chunk_buffers[
                    mask_agent_indices_chunk_buffer_size] = agents_chunk_buffer_size + agents_chunk_i * \
                                                            AGENT_CHUNK_SIZE[0]
                mask_agent_indices_chunk_buffer_size += 1
            agents_chunk_buffer_size += 1
            if agents_chunk_buffer_size == AGENT_CHUNK_SIZE[0]:
                agents_chunk_buffer_size = 0
                root.agents[agents_chunk_i * AGENT_CHUNK_SIZE[0]: (agents_chunk_i + 1) * AGENT_CHUNK_SIZE[0]] = \
                    agents_chunk_buffers
                agents_chunk_i += 1

            if add_standard_mask_indices and mask_agent_indices_chunk_buffer_size == \
                    MASK_AGENT_INDICES_ARRAY_CHUNK_SIZE[0]:
                mask_agent_indices_chunk_buffer_size = 0
                root.mask_agent_indices[mask_agent_indices_chunk_i * MASK_AGENT_INDICES_ARRAY_CHUNK_SIZE[0]:
                                        (mask_agent_indices_chunk_i + 1) * MASK_AGENT_INDICES_ARRAY_CHUNK_SIZE[0]] = \
                    mask_agent_indices_chunk_buffers
                mask_agent_indices_chunk_i += 1

root.agents[agents_chunk_i * AGENT_CHUNK_SIZE[0]:] = agents_chunk_buffers[:agents_chunk_buffer_size]
if add_standard_mask_indices:
    root.mask_agent_indices[
    mask_agent_indices_chunk_i * MASK_AGENT_INDICES_ARRAY_CHUNK_SIZE[0]:] = mask_agent_indices_chunk_buffers[
                                                                            :mask_agent_indices_chunk_buffer_size]

check_zarr = zarr.open_group(new_path, mode='r')
