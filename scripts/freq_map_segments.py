import sys

sys.path.insert(0, '../scripts')
from map_traffic_lights_data import *
from l5kit_modified.l5kit_modified import FramesDataset
import numpy as np
import pandas as pd

from l5kit.data import LocalDataManager
from tqdm.auto import tqdm

import os
import pickle
from collections import deque, defaultdict
from sklearn.cluster import KMeans

os.environ["L5KIT_DATA_FOLDER"] = "../input/"
SEGMENTS_OUTPUT_PATH = '../outputs/map_segments'
if not os.path.exists(SEGMENTS_OUTPUT_PATH):
    os.makedirs(SEGMENTS_OUTPUT_PATH)
n_base_segments = 13

dm = LocalDataManager()
dataset_path = dm.require(
    'scenes/validate_filtered_min_frame_history_4_min_frame_future_1_with_mask_idx.zarr')  # validate_filtered_min_frame_history_4_min_frame_future_1.zarr')
frame_dataset = FramesDataset(dataset_path, agents_from_standard_mask_only=True)


def get_agents_lanes(frame_sample):
    results = []
    agents_with_wheels = [(agent, ALL_WHEELS_CLASS if np.nonzero(agent[-1])[0][0] == CAR_CLASS else BIKE_CLASS) for
                          agent in frame_sample['agents'] if np.nonzero(agent[-1])[0][0] in [CAR_CLASS, BIKE_CLASS]]
    for agent, agent_class in agents_with_wheels:
        agent_centroid = agent[0]
        agent_yaw = agent[2]
        closest_lane_id = find_closest_lane(agent_centroid, agent_yaw, agent_class,
                                            return_point_i=False,
                                            return_blocked_tl_signals=False,
                                            intersections_only=False)
        if closest_lane_id is not None:
            results.append(closest_lane_id)
    return results


def get_lane_counts(frame_dataset, scene_idx_bounds, scenes_per_video=10):
    lane_id_2_count = defaultdict(int)
    for start_scene in tqdm(range(*scene_idx_bounds, scenes_per_video), desc='Scenes...'):
        end_scene = start_scene + scenes_per_video
        for scene_idx in range(start_scene, end_scene):
            indexes = frame_dataset.get_scene_indices(scene_idx)
            for idx in indexes:
                agents_lanes = get_agents_lanes(frame_dataset[idx])
                for lane_id in agents_lanes:
                    lane_id_2_count[lane_id] += 1
    return lane_id_2_count


lane_id_2_count_val_path = '../outputs/lane_id_2_count_val.pkl'
if os.path.exists(lane_id_2_count_val_path):
    with open(lane_id_2_count_val_path, 'rb') as f:
        lane_id_2_count_val = pickle.load(f)
else:
    lane_id_2_count_val = get_lane_counts(frame_dataset, [0, len(frame_dataset.zarr_root['scenes'])])
    with open(lane_id_2_count_val_path, 'wb') as f:
        pickle.dump(lane_id_2_count_val, f)


def rotate_point(x, y, angle_rad=0.26 * np.pi, back=False):
    if back:
        angle_rad *= -1
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    return x * cos - y * sin, x * sin + y * cos


######################### CLUSTERING VAL VISITED LANES AND ITS NEIGHBOURS ###########################
lane_ids = []
coordinates = []


def add_lane(lane_id):
    lane_ids.append(lane_id)
    center_line = get_lane_center_line(lane_id)
    mean_x, mean_y = center_line[:, 0].mean(), center_line[:, 1].mean()
    coordinates.append([mean_x, mean_y])


for lane_id in tqdm(lane_id_2_count_val, desc='Prep lanes for clustering..'):
    add_lane(lane_id)
    for lane_neighb_id in get_lane_neighbours_all(lane_id) + get_lane_neighbours_based_on_dist(lane_id):
        if lane_neighb_id not in lane_id_2_count_val:
            add_lane(lane_id)

coords_np = np.array(coordinates)

kmeans = KMeans(n_clusters=n_base_segments, random_state=0)
kmeans.fit(coordinates)

lanes_clusters = pd.DataFrame({'cluster': kmeans.labels_, 'lane_id': lane_ids})
cluster_2_lanes = lanes_clusters.groupby('cluster')['lane_id'].agg(list)

lane_2_clusters = dict()
for cluster, lanes in cluster_2_lanes.iteritems():
    for lane_id in lanes:
        lane_2_clusters[lane_id] = cluster

intersection_sets = []
covered_lanes = set()
for lane_id in lane_id_2_count_val:
    if lane_id in lane_2_master_intersection_related_lanes and lane_id not in covered_lanes:
        intersection_set = lane_2_master_intersection_related_lanes[lane_id].copy()
        intersection_set.add(lane_id)
        covered_lanes.update(intersection_set)
        intersection_sets.append(intersection_set)
    if len(intersection_sets) == 10:
        break

for intersection_set in intersection_sets:
    intersection_clusters = defaultdict(int)
    missing_lanes = []
    for lane_id in intersection_set:
        if lane_id in lane_2_clusters:
            intersection_clusters[lane_2_clusters[lane_id]] += 1
        else:
            missing_lanes.append(lane_id)
    if len(intersection_clusters) > 1 or len(missing_lanes):
        main_cluster = sorted(intersection_clusters.items(), key=lambda x: -x[1])[0][0]
        for lane_id in intersection_set:
            lane_2_clusters[lane_id] = main_cluster

segment_idx_x_min_rot = [float('inf') for _ in range(n_base_segments)]
segment_idx_y_min_rot = [float('inf') for _ in range(n_base_segments)]
segment_idx_x_max_rot = [-float('inf') for _ in range(n_base_segments)]
segment_idx_y_max_rot = [-float('inf') for _ in range(n_base_segments)]
# estimating segment boarders       
for lane_id, cluster in lane_2_clusters.items():
    segment_lane_center_rot = get_lane_center_line(lane_id).copy()
    for point_i in range(len(segment_lane_center_rot)):
        segment_lane_center_rot[point_i] = rotate_point(*segment_lane_center_rot[point_i])

    segment_idx_x_min_rot[cluster] = min(segment_idx_x_min_rot[cluster], segment_lane_center_rot[:, 0].min())
    segment_idx_y_min_rot[cluster] = min(segment_idx_y_min_rot[cluster], segment_lane_center_rot[:, 1].min())
    segment_idx_x_max_rot[cluster] = max(segment_idx_x_max_rot[cluster], segment_lane_center_rot[:, 0].max())
    segment_idx_y_max_rot[cluster] = max(segment_idx_y_max_rot[cluster], segment_lane_center_rot[:, 1].max())

segment_x_coords = []
segment_y_coords = []
TYPE_MAX = 0
TYPE_MIN = -1
for segment_idx in range(n_base_segments):
    segment_x_coords.append((segment_idx_x_min_rot[segment_idx], segment_idx, TYPE_MIN))
    segment_x_coords.append((segment_idx_x_max_rot[segment_idx], segment_idx, TYPE_MAX))
    segment_y_coords.append((segment_idx_y_min_rot[segment_idx], segment_idx, TYPE_MIN))
    segment_y_coords.append((segment_idx_y_max_rot[segment_idx], segment_idx, TYPE_MAX))

segment_x_coords = sorted(segment_x_coords, key=lambda x: x[0])
segment_y_coords = sorted(segment_y_coords, key=lambda x: x[0])


def map_intervals_to_coordinate_segments(segment_coords):
    active_segments = set()
    interval_2_segments = []
    for coord_interval, (coord, segment_idx, point_type) in enumerate(segment_coords):
        if point_type == TYPE_MIN:
            active_segments.add(segment_idx)
        elif point_type == TYPE_MAX:
            active_segments.remove(segment_idx)
        else:
            raise NotImplementedError('Unknown point type')
        interval_2_segments.append(active_segments.copy())
    return interval_2_segments


interval_2_segments_x = map_intervals_to_coordinate_segments(segment_x_coords)
interval_2_segments_y = map_intervals_to_coordinate_segments(segment_y_coords)
segment_x_coords_only = [x[0] for x in segment_x_coords]
segment_y_coords_only = [x[0] for x in segment_y_coords]

map_segment_2_lanes = [set() for _ in range(n_base_segments)]

for lane_id, map_segment_idx in lane_2_clusters.items():
    map_segment_2_lanes[map_segment_idx].add(lane_id)

# for matches on boarders as well to then have several seq2seq models for lane-seq prediction -> add at least 50 points of consecutive lanes for each segment
min_lane_points_dist = 50

for map_segment_idx in range(n_base_segments):
    lanes_to_add = set()
    for lane_id in map_segment_2_lanes[map_segment_idx]:
        queue = deque()
        closed_set = set()
        queue.append((lane_id, 0))
        while len(queue):
            next_lane_id, dist_from_init = queue.popleft()
            closed_set.add(next_lane_id)
            if next_lane_id not in map_segment_2_lanes[map_segment_idx] and next_lane_id not in lanes_to_add:
                lanes_to_add.add(next_lane_id)
            next_dist_from_init = dist_from_init + get_lane_len(next_lane_id)
            if next_dist_from_init < min_lane_points_dist:
                for next_lane_neigh in get_lane_neighbours_all(next_lane_id):
                    if next_lane_neigh not in closed_set:
                        queue.append((next_lane_neigh, next_dist_from_init))
    map_segment_2_lanes[map_segment_idx].update(lanes_to_add)

# ensuring that each match inside the boundaries is present
for lane_id in lane_id_2_idx.keys():
    lane_center_line = get_lane_center_line(lane_id)
    for coord in lane_center_line:
        map_segment_indices = match_point_2_map_segment(*coord, segment_x_coords_only, segment_y_coords_only,
                                                        interval_2_segments_x, interval_2_segments_y)
        for map_segment_idx in map_segment_indices:
            if lane_id not in map_segment_2_lanes[map_segment_idx]:
                map_segment_2_lanes[map_segment_idx].add(lane_id)

with open(os.path.join(SEGMENTS_OUTPUT_PATH, 'map_segment_2_lanes.pkl'), 'wb') as f:
    pickle.dump(map_segment_2_lanes, f)

with open(os.path.join(SEGMENTS_OUTPUT_PATH, 'interval_2_segments_x.pkl'), 'wb') as f:
    pickle.dump(interval_2_segments_x, f)
with open(os.path.join(SEGMENTS_OUTPUT_PATH, 'interval_2_segments_y.pkl'), 'wb') as f:
    pickle.dump(interval_2_segments_y, f)
with open(os.path.join(SEGMENTS_OUTPUT_PATH, 'segment_x_coords_only.pkl'), 'wb') as f:
    pickle.dump(segment_x_coords_only, f)
with open(os.path.join(SEGMENTS_OUTPUT_PATH, 'segment_y_coords_only.pkl'), 'wb') as f:
    pickle.dump(segment_y_coords_only, f)
