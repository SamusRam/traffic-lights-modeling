import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import os
    import sys

    if os.name == 'nt':
        from scripts.l5kit_modified.map_api import *
        from scripts.l5kit_modified.l5kit_modified import AgentDatasetModified

        os.environ["L5KIT_DATA_FOLDER"] = "input/"
    elif os.name == 'posix':
        from l5kit_modified.map_api import *
        from l5kit_modified.l5kit_modified import AgentDatasetModified

        os.environ["L5KIT_DATA_FOLDER"] = "../input/"
    else:
        sys.exit('Unsupported platform')

from l5kit.data import ChunkedDataset, LocalDataManager
from random import sample
from scipy.spatial import KDTree
from tqdm.auto import tqdm
from copy import deepcopy
from collections import defaultdict

############# CONSTANTS ###############
FILTER_AGENTS_THRESHOLD = 0.5
NUM_FRAMES_TO_CHOP = 100
NUM_FUTURE_FRAMES = 50
MIN_FUTURE_STEPS = 10
BATCH_SIZE = 256
NUM_PREDICTED_TRACKS = 3
NUM_KD_CANDIDATES = 5

world_to_ecef = np.asarray(
    [
        [8.46617444e-01, 3.23463078e-01, -4.22623402e-01, -2.69876744e06],
        [-5.32201938e-01, 5.14559352e-01, -6.72301845e-01, -4.29315158e06],
        [-3.05311332e-16, 7.94103464e-01, 6.07782600e-01, 3.85516476e06],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ],
    dtype=np.float64,
)

semantic_map_key = 'semantic_map/semantic_map.pb'

cfg = {"model_params": {"history_num_frames": 100,
                        "history_step_size": 1,
                        "future_num_frames": 50,
                        "future_step_size": 1},
       "raster_params": {"filter_agents_threshold": FILTER_AGENTS_THRESHOLD}}

dataset_path = 'scenes/sample.zarr'


################# HELPING FUNCTIONS ####################
def densify_sparse_segments(x_coordinates_seq_np, y_coordinates_seq_np, max_diff=0.01):
    # the operation is done just once, so performing it naively
    assert len(x_coordinates_seq_np) == len(y_coordinates_seq_np), "Different lens of x/y coordinates"
    x_final_seq_np = np.array([])
    y_final_seq_np = np.array([])
    for i in range(0, len(x_coordinates_seq_np) - 1):
        abs_diff = np.hypot(x_coordinates_seq_np[i + 1] - x_coordinates_seq_np[i],
                            y_coordinates_seq_np[i + 1] - y_coordinates_seq_np[i])
        if abs_diff > max_diff:
            n_interpolated_points = int(np.ceil(abs_diff / max_diff)) + 1
            x_points_to_append = np.interp(np.linspace(0, 1, n_interpolated_points),
                                           np.arange(2), x_coordinates_seq_np[i:i + 2])

            y_points_to_append = np.interp(np.linspace(0, 1, n_interpolated_points),
                                           np.arange(2), y_coordinates_seq_np[i:i + 2])

        else:
            x_points_to_append = x_coordinates_seq_np[i:i + 2]
            y_points_to_append = y_coordinates_seq_np[i:i + 2]
        if i > 0:
            x_points_to_append = x_points_to_append[1:]
            y_points_to_append = y_points_to_append[1:]
        x_final_seq_np = np.append(x_final_seq_np, x_points_to_append)
        y_final_seq_np = np.append(y_final_seq_np, y_points_to_append)
    return x_final_seq_np, y_final_seq_np


def sparsify(xy_seq_np, final_min_coord_dist_metr=2):
    prev_point = xy_seq_np[0]
    final_list = [prev_point]
    for i in range(1, len(xy_seq_np)):
        delta_x = prev_point[0] - xy_seq_np[i, 0]
        delta_y = prev_point[1] - xy_seq_np[i, 1]
        dist = np.hypot(delta_x, delta_y)
        if dist >= final_min_coord_dist_metr:
            final_list.append(xy_seq_np[i])
            prev_point = xy_seq_np[i]
    if np.any(prev_point != xy_seq_np[-1]):
        final_list.append(xy_seq_np[-1])
    return np.array(final_list)


def get_lane_cumul_distances(xy_seq_np):
    idx_2_cumul_distances = [[] for _ in range(len(xy_seq_np))]

    def get_dist(idx_0, idx_1):
        deltas = xy_seq_np[idx_0] - xy_seq_np[idx_1]
        return np.hypot(deltas[0], deltas[1])

    for i in range(len(xy_seq_np)):
        # computing backward along the lane
        dist_backward = 0
        backward_dists = []
        for k in range(i - 1, -1, -1):
            segment_len = get_dist(k, k + 1)
            dist_backward += segment_len
            backward_dists.append(dist_backward)
        idx_2_cumul_distances[i].extend(reversed(backward_dists))
        idx_2_cumul_distances[i].append(0)

        # computing forward along the lane
        dist_forward = 0
        for k in range(i + 1, len(xy_seq_np), 1):
            segment_len = get_dist(k, k - 1)
            dist_forward += segment_len
            idx_2_cumul_distances[i].append(dist_forward)
    return idx_2_cumul_distances


def get_helping_angle(vector_1):
    vector_0 = np.array([1, 0])
    angle_cos = np.dot(vector_0, vector_1) / np.linalg.norm(vector_1)
    alpha = np.arccos(angle_cos)
    if vector_1[1] < 0:
        alpha = 2 * np.pi - alpha
    return alpha


def get_lane_segments_sin_cosine(xy_seq_np):
    idx_2_sin_cos_forward = []
    for i in range(len(xy_seq_np) - 1):
        vector_1 = xy_seq_np[i + 1] - xy_seq_np[i]
        helping_angle = get_helping_angle(vector_1)
        idx_2_sin_cos_forward.append((np.sin(helping_angle), np.cos(helping_angle)))
    idx_2_sin_cos_forward.append((np.nan, np.nan))

    idx_2_sin_cos_backward = []
    for i in range(len(xy_seq_np) - 1, 0, -1):
        vector_1 = xy_seq_np[i - 1] - xy_seq_np[i]
        helping_angle = get_helping_angle(vector_1)
        idx_2_sin_cos_backward.append((np.sin(helping_angle), np.cos(helping_angle)))
    idx_2_sin_cos_backward.append((np.nan, np.nan))
    return idx_2_sin_cos_forward, list(reversed(idx_2_sin_cos_backward))


def precompute_map_elements(proto_API):
    lanes_ids = []
    crosswalks_ids = []
    center_line_coords = []
    xy_left_coords = []
    xy_right_coords = []
    xy_left_coords_ = []
    xy_right_coords_ = []

    lane_point_idx_2_cumul_distances = []
    lane_point_idx_2_sin_cos_forward = []
    lane_point_idx_2_sin_cos_backward = []

    lanes_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]
    crosswalks_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]

    for element in tqdm(proto_API):
        element_id = MapAPI.id_as_str(element.id)

        if proto_API.is_lane(element):
            lane = proto_API.get_lane_coords(element_id)
            x_min = min(np.min(lane["xyz_left"][:, 0]), np.min(lane["xyz_right"][:, 0]))
            y_min = min(np.min(lane["xyz_left"][:, 1]), np.min(lane["xyz_right"][:, 1]))
            x_max = max(np.max(lane["xyz_left"][:, 0]), np.max(lane["xyz_right"][:, 0]))
            y_max = max(np.max(lane["xyz_left"][:, 1]), np.max(lane["xyz_right"][:, 1]))

            x_left_, y_left_ = densify_sparse_segments(lane["xyz_left"][:, 0],
                                                       lane["xyz_left"][:, 1])
            x_right_, y_right_ = densify_sparse_segments(lane["xyz_right"][:, 0],
                                                         lane["xyz_right"][:, 1])

            if len(x_left_) == len(x_right_):
                x_right = x_right_
                x_left = x_left_

                y_right = y_right_
                y_left = y_left_

            elif len(x_left_) < len(x_right_):
                x_right = x_right_
                x_left = np.interp(np.linspace(0, len(x_left_) - 1, len(x_right)),
                                   np.arange(len(x_left_)), x_left_)

                y_right = y_right_
                y_left = np.interp(np.linspace(0, len(y_left_) - 1, len(y_right)),
                                   np.arange(len(y_left_)), y_left_)

            elif len(x_left_) > len(x_right_):
                x_left = x_left_
                x_right = np.interp(np.linspace(0, len(x_right_) - 1, len(x_left)),
                                    np.arange(len(x_right_)), x_right_)

                y_left = y_left_
                y_right = np.interp(np.linspace(0, len(y_right_) - 1, len(y_left)),
                                    np.arange(len(y_right_)), y_right_)
            else:
                raise Exception('Bug in lane length comparison')
            assert len(x_left) == len(x_right)

            center_line = np.transpose(np.vstack(((x_left + x_right) / 2,
                                                  (y_left + y_right) / 2)))
            center_line = sparsify(center_line)

            lanes_bounds = np.append(lanes_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)
            lanes_ids.append(element_id)
            center_line_coords.append(center_line)
            xy_left_coords.append(lane["xyz_left"][:, :2])
            xy_right_coords.append(lane["xyz_right"][:, :2])
            xy_left_dense = np.transpose(np.vstack((x_left, y_left)))
            xy_right_dense = np.transpose(np.vstack((x_right, y_right)))
            xy_left_coords_.append(sparsify(xy_left_dense))
            xy_right_coords_.append(sparsify(xy_right_dense))

            lane_point_idx_2_cumul_distances.append(get_lane_cumul_distances(center_line))
            idx_2_sin_cos_forward, idx_2_sin_cos_backward = get_lane_segments_sin_cosine(center_line)
            lane_point_idx_2_sin_cos_forward.append(idx_2_sin_cos_forward)
            lane_point_idx_2_sin_cos_backward.append(idx_2_sin_cos_backward)
            assert len(idx_2_sin_cos_forward) == len(idx_2_sin_cos_backward) == len(center_line)

        if proto_API.is_crosswalk(element):
            crosswalk = proto_API.get_crosswalk_coords(element_id)
            x_min = np.min(crosswalk["xyz"][:, 0])
            y_min = np.min(crosswalk["xyz"][:, 1])
            x_max = np.max(crosswalk["xyz"][:, 0])
            y_max = np.max(crosswalk["xyz"][:, 1])

            crosswalks_bounds = np.append(
                crosswalks_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0,
            )
            crosswalks_ids.append(element_id)

    return {
        "lanes": {"bounds": lanes_bounds, "ids": lanes_ids,
                  "center_line": center_line_coords,
                  "xy_left": xy_left_coords, "xy_right": xy_right_coords,
                  "xy_left_": xy_left_coords_, "xy_right_": xy_right_coords_,
                  "lane_point_idx_2_cumul_distances": lane_point_idx_2_cumul_distances,
                  "lane_point_idx_2_sin_cos_forward": lane_point_idx_2_sin_cos_forward,
                  "lane_point_idx_2_sin_cos_backward": lane_point_idx_2_sin_cos_backward
                  },
        "crosswalks": {"bounds": crosswalks_bounds, "ids": crosswalks_ids},
    }


def precompute_lane_adjacencies(id_2_idx, proto_API):
    lane_adj_list_forward = [[] for _ in range(len(id_2_idx))]
    lane_adj_list_backward = [[] for _ in range(len(id_2_idx))]
    lane_adj_list_right = [[] for _ in range(len(id_2_idx))]
    lane_adj_list_left = [[] for _ in range(len(id_2_idx))]

    for element in tqdm(proto_API, desc='Computing lane adjacency lists'):
        element_id = MapAPI.id_as_str(element.id)
        if proto_API.is_lane(element):
            lanes_ahead = proto_API.get_lanes_ahead(element_id)
            lane_adj_list_forward[id_2_idx[element_id]].extend(lanes_ahead)
            for lane_ahead_id in lanes_ahead:
                lane_adj_list_backward[id_2_idx[lane_ahead_id]].append(element_id)

            lane_left = proto_API.get_lane_to_left(element_id)
            if lane_left != '':
                lane_adj_list_left[id_2_idx[element_id]].append(lane_left)
            lane_right = proto_API.get_lane_to_right(element_id)
            if lane_right != '':
                lane_adj_list_right[id_2_idx[element_id]].append(lane_right)

    return lane_adj_list_forward, lane_adj_list_backward, lane_adj_list_right, lane_adj_list_left


def cos_dist(x1, y1, yaw_rad):
    x2 = np.cos(yaw_rad)
    y2 = np.sin(yaw_rad)
    vector_product = x1 * x2 + y1 * y2
    cos = vector_product / (np.hypot(x1, y1) * np.hypot(x2, y2))
    return 1 - cos


def find_closest_lane(coord, yaw, kd_tree, kd_idx_2_lane_id_idx, lanes_crosswalks, lane_id_2_idx):
    candidate_indices = kd_tree.query_ball_point(coord, r=3)
    closed_set = set()
    min_cos_dist = float('inf')
    result_lane_id = None
    for i, candidate_idx in enumerate(candidate_indices):
        lane_id, point_idx = kd_idx_2_lane_id_idx[candidate_idx]
        if lane_id not in closed_set:
            closed_set.add(lane_id)
            lane_center_line = lanes_crosswalks['lanes']['center_line'][lane_id_2_idx[lane_id]]
            if point_idx < len(lane_center_line) - 1:
                x1 = lane_center_line[point_idx + 1][0] - lane_center_line[point_idx][0]
                y1 = lane_center_line[point_idx + 1][1] - lane_center_line[point_idx][1]
            else:  # there're always at least 2 points
                x1 = lane_center_line[point_idx][0] - lane_center_line[point_idx - 1][0]
                y1 = lane_center_line[point_idx][1] - lane_center_line[point_idx - 1][1]
            lane_cos_dist = cos_dist(x1, y1, yaw)
            if lane_cos_dist < min_cos_dist:
                min_cos_dist = lane_cos_dist
                result_lane_id = lane_id
    return result_lane_id

################# NAIVE LANE FOLLOWER ##################
class ConstantSpeedLaneFollower:

    def __init__(self, dataset_path=dataset_path, semantic_map_key=semantic_map_key,
                 world_to_ecef=world_to_ecef, cfg=cfg):
        dm = LocalDataManager(None)
        eval_zarr = ChunkedDataset(dm.require(dataset_path)).open()
        self.dataset = AgentDatasetModified(cfg, eval_zarr)
        semantic_map_path = dm.require(semantic_map_key)
        self.proto_API = MapAPI(semantic_map_path, world_to_ecef)
        self.lanes_crosswalks = precompute_map_elements(self.proto_API)
        self.id_2_idx = {lane_id: i for i, lane_id in enumerate(self.lanes_crosswalks['lanes']['ids'])}
        self.lane_adj_list_forward, self.lane_adj_list_backward, _, _ = precompute_lane_adjacencies(self.id_2_idx, self.proto_API)
        all_center_coords = np.concatenate(self.lanes_crosswalks['lanes']['center_line'], axis=0)
        self.kd_tree = KDTree(all_center_coords)

        self.kd_idx_2_lane_id_idx = []
        for lane_id, center_line in zip(self.lanes_crosswalks['lanes']['ids'],
                                        self.lanes_crosswalks['lanes']['center_line']):
            next_entries = [(lane_id, i) for i in range(len(center_line))]
            self.kd_idx_2_lane_id_idx.extend(next_entries)

    def get_predicted_coordinates(self, start_point_coord, lane_id, lane_point_idx, forward, speed_m_per_frame,
                                  current_overshoot_m=0, n_prediction_steps=50, include_start=False):
        if speed_m_per_frame == 0:
            return [start_point_coord.reshape(1, 2)]
        adj_list = self.lane_adj_list_forward if forward else self.lane_adj_list_backward
        lane_idx = self.id_2_idx[lane_id]
        lane_point_idx_2_cumul_distances = deepcopy(
            self.lanes_crosswalks['lanes']['lane_point_idx_2_cumul_distances'][lane_idx])
        center_line = deepcopy(self.lanes_crosswalks['lanes']['center_line'][lane_idx])
        center_line[lane_point_idx] = start_point_coord

        boarder_idx = len(center_line) - 1 if forward else 0

        # when already on the edge of the lane, proceed to the next lane
        if lane_point_idx == boarder_idx:
            consec_future_coordinates_tracks = []
            for consecutive_lane_id in adj_list[lane_idx]:
                consec_lane_point_idx = 0 if forward else len(
                    self.lanes_crosswalks['lanes']['center_line'][self.id_2_idx[consecutive_lane_id]]) - 1
                consec_future_coordinates_tracks.extend(
                    self.get_predicted_coordinates(start_point_coord, consecutive_lane_id,
                                                   consec_lane_point_idx, forward, speed_m_per_frame,
                                                   current_overshoot_m,
                                                   n_prediction_steps=n_prediction_steps, include_start=True))
            if n_prediction_steps == NUM_FUTURE_FRAMES and len(consec_future_coordinates_tracks) == 0:
                return [start_point_coord.reshape(1, 2)]
            return consec_future_coordinates_tracks

        direction_multiplier = 1 if forward else -1

        # fixing distances after insertion of optional start point into lane
        deltas = center_line[lane_point_idx + direction_multiplier] - start_point_coord
        dist_to_next_point = np.hypot(deltas[0], deltas[1])
        dist_from_start = dist_to_next_point
        dist_delta = dist_to_next_point - lane_point_idx_2_cumul_distances[lane_point_idx][
            lane_point_idx + direction_multiplier]
        lane_point_idx_2_cumul_distances[lane_point_idx][lane_point_idx + direction_multiplier] += dist_delta
        fix_k = 2 * direction_multiplier
        while 0 <= lane_point_idx + fix_k < len(center_line):
            lane_point_idx_2_cumul_distances[lane_point_idx][lane_point_idx + fix_k] += dist_delta
            dist_from_start += lane_point_idx_2_cumul_distances[lane_point_idx][lane_point_idx + fix_k]
            fix_k += direction_multiplier

        if forward:
            lane_point_idx_2_sin_cos = deepcopy(
                self.lanes_crosswalks['lanes']['lane_point_idx_2_sin_cos_forward'][lane_idx])
        else:
            lane_point_idx_2_sin_cos = deepcopy(
                self.lanes_crosswalks['lanes']['lane_point_idx_2_sin_cos_backward'][lane_idx])

        # computing new sin/cos
        vector_1 = center_line[lane_point_idx + direction_multiplier] - start_point_coord
        helping_angle = get_helping_angle(vector_1)
        lane_point_idx_2_sin_cos[lane_point_idx] = (np.sin(helping_angle), np.cos(helping_angle))

        def find_next_point_idx_and_overshoot(travelled_dist, start_point_idx, current_overshoot_m):
            point_idx = start_point_idx + direction_multiplier
            cumul_distances = lane_point_idx_2_cumul_distances[start_point_idx]
            while 0 <= point_idx < len(cumul_distances) and cumul_distances[
                point_idx] - current_overshoot_m < travelled_dist:
                point_idx += direction_multiplier

            point_idx -= direction_multiplier
            overshoot = travelled_dist - cumul_distances[point_idx] + current_overshoot_m
            return point_idx, overshoot

        current_point_idx = lane_point_idx
        if not include_start:
            future_coordinates = []
        else:
            future_coordinates = [
                start_point_coord + current_overshoot_m * np.array([np.cos(helping_angle), np.sin(helping_angle)])]
            n_prediction_steps -= 1
        prediction_finished = False

        for step_i in range(n_prediction_steps):
            if prediction_finished:
                # to be able to check if there were consecutive lanes using step_i
                step_i -= 1
                break
            travelled_dist = speed_m_per_frame
            current_point_idx, current_overshoot_m = find_next_point_idx_and_overshoot(travelled_dist,
                                                                                       current_point_idx,
                                                                                       current_overshoot_m)
            current_point_coords = center_line[current_point_idx]
            if current_point_idx == boarder_idx:
                prediction_finished = True
                # process consecutive lane(s)
                consec_future_coordinates_tracks = []
                for consecutive_lane_id in adj_list[lane_idx]:
                    consec_lane_point_idx = 0 if forward else len(
                        self.lanes_crosswalks['lanes']['center_line'][self.id_2_idx[consecutive_lane_id]]) - 1
                    consec_future_coordinates_tracks.extend(
                        self.get_predicted_coordinates(center_line[boarder_idx], consecutive_lane_id,
                                                  consec_lane_point_idx, forward, speed_m_per_frame,
                                                  current_overshoot_m,
                                                  n_prediction_steps=n_prediction_steps - step_i, include_start=True))
                consec_future_coordinates_tracks = [x for x in consec_future_coordinates_tracks if len(x) > 0]
                if len(future_coordinates):
                    future_coordinates = np.vstack(future_coordinates)
                    if len(consec_future_coordinates_tracks):
                        future_coordinates_tracks = [np.concatenate((future_coordinates, x), axis=0)
                                                     for x in consec_future_coordinates_tracks]
                    else:
                        future_coordinates_tracks = [future_coordinates]
                elif len(consec_future_coordinates_tracks):
                    future_coordinates_tracks = consec_future_coordinates_tracks
                elif current_point_idx > 0:  # there was at least one prev point
                    sin_, cos_ = lane_point_idx_2_sin_cos[current_point_idx - 1]
                    next_position_coordinates = current_point_coords + current_overshoot_m * np.array([cos_, sin_])
                    future_coordinates_tracks = [next_position_coordinates.reshape(1, 2)]
                else:
                    future_coordinates_tracks = [current_point_coords.reshape(1, 2)]

            else:
                sin_, cos_ = lane_point_idx_2_sin_cos[current_point_idx]
                next_position_coordinates = current_point_coords + current_overshoot_m * np.array([cos_, sin_])
                future_coordinates.append(next_position_coordinates)

        if n_prediction_steps == 0 or step_i == n_prediction_steps - 1:
            if len(future_coordinates):
                future_coordinates_tracks = [np.vstack(future_coordinates)]
            elif n_prediction_steps == NUM_FUTURE_FRAMES:  # not a recursive call
                future_coordinates_tracks = [start_point_coord.reshape(1, 2)]
            else:
                future_coordinates_tracks = []

        return future_coordinates_tracks

    def is_lane_direction_forward(self, coordinates, lane_id, lane_point_idx, speed_vector):
        lane_idx = self.id_2_idx[lane_id]
        center_line = self.lanes_crosswalks['lanes']['center_line'][lane_idx]
        if lane_point_idx < len(center_line) - 1:
            forward_point = center_line[lane_point_idx + 1]
        elif len(self.lane_adj_list_forward[lane_idx]):
            consecutive_lane_id = self.lane_adj_list_forward[lane_idx][0]
            forward_point = self.lanes_crosswalks['lanes']['center_line'][self.id_2_idx[consecutive_lane_id]][0]
        else:
            forward_point = center_line[lane_point_idx]

        if lane_point_idx > 0:
            backward_point = center_line[lane_point_idx - 1]
        elif len(self.lane_adj_list_backward[lane_idx]):
            consecutive_lane_id = self.lane_adj_list_backward[lane_idx][0]
            backward_point = self.lanes_crosswalks['lanes']['center_line'][self.id_2_idx[consecutive_lane_id]][-1]
        else:
            backward_point = center_line[lane_point_idx]

        next_point = coordinates + speed_vector * 10
        diff_backward = next_point - backward_point
        diff_forward = next_point - forward_point
        return np.hypot(diff_forward[0], diff_forward[1]) < np.hypot(diff_backward[0], diff_backward[1])

    def get_prediction(self, start_coordinates, start_speed_m_per_frame):
        dists_to_start, candidate_start_indices = self.kd_tree.query(start_coordinates, k=NUM_KD_CANDIDATES)
        dists_to_next_point, candidate_next_indices = self.kd_tree.query(start_coordinates + start_speed_m_per_frame,
                                                                         k=NUM_KD_CANDIDATES)
        lane_2_dist = defaultdict(list)
        for dist_to_start, idx in zip(dists_to_start, candidate_start_indices):
            lane_id = self.kd_idx_2_lane_id_idx[idx][0]
            lane_2_dist[lane_id].append(dist_to_start)
        for dist_to_next_point, idx in zip(dists_to_next_point, candidate_next_indices):
            lane_id = self.kd_idx_2_lane_id_idx[idx][0]
            if lane_id in lane_2_dist:
                lane_2_dist[lane_id].append(dist_to_next_point)

        chosen_lane_id = min([(lane_id, np.mean(dists))
                              for lane_id, dists in lane_2_dist.items() if len(dists) >= 2], key=lambda x: x[1])[0]
        lane_point_idx = None
        i = 0
        while i < NUM_KD_CANDIDATES and lane_point_idx is None:
            lane_id, point_idx = self.kd_idx_2_lane_id_idx[candidate_start_indices[i]]
            if lane_id == chosen_lane_id:
                lane_point_idx = point_idx
            i += 1
        assert lane_point_idx is not None, f'start_coordinates: {start_coordinates}, start_speed_m_per_frame: {start_speed_m_per_frame}'

        is_forward = self.is_lane_direction_forward(start_coordinates, chosen_lane_id, lane_point_idx, start_speed_m_per_frame)
        future_coordinates_tracks = self.get_predicted_coordinates(start_coordinates,
                                                                   chosen_lane_id, lane_point_idx,
                                                                   is_forward,
                                                                   np.hypot(start_speed_m_per_frame[0], start_speed_m_per_frame[1]),
                                                                   n_prediction_steps=NUM_FUTURE_FRAMES)
        if len(future_coordinates_tracks) > NUM_PREDICTED_TRACKS:
            randomly_picked_future_coordinates_tracks = sample(future_coordinates_tracks, NUM_PREDICTED_TRACKS)
        else:
            randomly_picked_future_coordinates_tracks = future_coordinates_tracks
        for i, track in enumerate(randomly_picked_future_coordinates_tracks):
            if len(track) < NUM_FUTURE_FRAMES:
                track = np.concatenate((track, track[-1] * np.ones((NUM_FUTURE_FRAMES - len(track), 2))))
                randomly_picked_future_coordinates_tracks[i] = track
        if len(randomly_picked_future_coordinates_tracks) == 0:
            print(f'!!! start_coordinates, start_speed_m_per_frame', start_coordinates, start_speed_m_per_frame)
        return np.expand_dims(np.stack(randomly_picked_future_coordinates_tracks), 0)
