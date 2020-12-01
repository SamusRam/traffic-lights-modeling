from lyft_trajectories.utils.l5kit_modified.map_api import MapAPI
import numpy as np
from tqdm.auto import tqdm
from typing import Dict

# CONSTANTS ###############
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

semantic_map_key = "semantic_map/semantic_map.pb"


# HELPING FUNCTIONS ####################
def densify_sparse_segments(
    x_coordinates_seq_np: np.ndarray,
    y_coordinates_seq_np: np.ndarray,
    max_diff: float = 0.01,
):
    # the operation is done just once, so performing it naively
    if len(x_coordinates_seq_np) != len(
        y_coordinates_seq_np
    ):
        raise AssertionError("Different lens of x/y coordinates")
    x_final_seq_np = np.array([])
    y_final_seq_np = np.array([])
    for i in range(0, len(x_coordinates_seq_np) - 1):
        abs_diff = np.hypot(
            x_coordinates_seq_np[i + 1] - x_coordinates_seq_np[i],
            y_coordinates_seq_np[i + 1] - y_coordinates_seq_np[i],
        )
        if abs_diff > max_diff:
            n_interpolated_points = int(np.ceil(abs_diff / max_diff)) + 1
            x_points_to_append = np.interp(
                np.linspace(0, 1, n_interpolated_points),
                np.arange(2),
                x_coordinates_seq_np[i : i + 2],
            )

            y_points_to_append = np.interp(
                np.linspace(0, 1, n_interpolated_points),
                np.arange(2),
                y_coordinates_seq_np[i : i + 2],
            )

        else:
            x_points_to_append = x_coordinates_seq_np[i : i + 2]
            y_points_to_append = y_coordinates_seq_np[i : i + 2]
        if i > 0:
            x_points_to_append = x_points_to_append[1:]
            y_points_to_append = y_points_to_append[1:]
        x_final_seq_np = np.append(x_final_seq_np, x_points_to_append)
        y_final_seq_np = np.append(y_final_seq_np, y_points_to_append)
    return x_final_seq_np, y_final_seq_np


def sparsify(xy_seq_np: np.ndarray, final_min_coord_dist_metr: float = 2.0):
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


def get_lane_cumul_distances(xy_seq_np: np.ndarray):
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


def get_helping_angle(vector_1: np.ndarray):
    vector_0 = np.array([1, 0])
    angle_cos = np.dot(vector_0, vector_1) / np.linalg.norm(vector_1)
    alpha = np.arccos(angle_cos)
    if vector_1[1] < 0:
        alpha = 2 * np.pi - alpha
    return alpha


def get_lane_segments_sin_cosine(xy_seq_np: np.ndarray):
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


def precompute_map_elements(proto_API: MapAPI):
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

    lanes_bounds = np.empty(
        (0, 2, 2), dtype=np.float
    )  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]
    crosswalks_bounds = np.empty(
        (0, 2, 2), dtype=np.float
    )  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]

    for element in tqdm(proto_API):
        element_id = MapAPI.id_as_str(element.id)

        if proto_API.is_lane(element):
            lane = proto_API.get_lane_coords(element_id)
            x_min = min(np.min(lane["xyz_left"][:, 0]), np.min(lane["xyz_right"][:, 0]))
            y_min = min(np.min(lane["xyz_left"][:, 1]), np.min(lane["xyz_right"][:, 1]))
            x_max = max(np.max(lane["xyz_left"][:, 0]), np.max(lane["xyz_right"][:, 0]))
            y_max = max(np.max(lane["xyz_left"][:, 1]), np.max(lane["xyz_right"][:, 1]))

            x_left_, y_left_ = densify_sparse_segments(
                lane["xyz_left"][:, 0], lane["xyz_left"][:, 1]
            )
            x_right_, y_right_ = densify_sparse_segments(
                lane["xyz_right"][:, 0], lane["xyz_right"][:, 1]
            )

            if len(x_left_) == len(x_right_):
                x_right = x_right_
                x_left = x_left_

                y_right = y_right_
                y_left = y_left_

            elif len(x_left_) < len(x_right_):
                x_right = x_right_
                x_left = np.interp(
                    np.linspace(0, len(x_left_) - 1, len(x_right)),
                    np.arange(len(x_left_)),
                    x_left_,
                )

                y_right = y_right_
                y_left = np.interp(
                    np.linspace(0, len(y_left_) - 1, len(y_right)),
                    np.arange(len(y_left_)),
                    y_left_,
                )

            elif len(x_left_) > len(x_right_):
                x_left = x_left_
                x_right = np.interp(
                    np.linspace(0, len(x_right_) - 1, len(x_left)),
                    np.arange(len(x_right_)),
                    x_right_,
                )

                y_left = y_left_
                y_right = np.interp(
                    np.linspace(0, len(y_right_) - 1, len(y_left)),
                    np.arange(len(y_right_)),
                    y_right_,
                )
            else:
                raise Exception("Bug in lane length comparison")
            if len(x_left) != len(x_right):
                raise AssertionError

            center_line = np.transpose(
                np.vstack(((x_left + x_right) / 2, (y_left + y_right) / 2))
            )
            center_line = sparsify(center_line)

            lanes_bounds = np.append(
                lanes_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0
            )
            lanes_ids.append(element_id)
            center_line_coords.append(center_line)
            xy_left_coords.append(lane["xyz_left"][:, :2])
            xy_right_coords.append(lane["xyz_right"][:, :2])
            xy_left_dense = np.transpose(np.vstack((x_left, y_left)))
            xy_right_dense = np.transpose(np.vstack((x_right, y_right)))
            xy_left_coords_.append(sparsify(xy_left_dense))
            xy_right_coords_.append(sparsify(xy_right_dense))

            lane_point_idx_2_cumul_distances.append(
                get_lane_cumul_distances(center_line)
            )
            (
                idx_2_sin_cos_forward,
                idx_2_sin_cos_backward,
            ) = get_lane_segments_sin_cosine(center_line)
            lane_point_idx_2_sin_cos_forward.append(idx_2_sin_cos_forward)
            lane_point_idx_2_sin_cos_backward.append(idx_2_sin_cos_backward)
            if not (len(idx_2_sin_cos_forward) == len(idx_2_sin_cos_backward) and len(
                idx_2_sin_cos_forward
            ) == len(center_line)):
                raise AssertionError

        if proto_API.is_crosswalk(element):
            crosswalk = proto_API.get_crosswalk_coords(element_id)
            x_min = np.min(crosswalk["xyz"][:, 0])
            y_min = np.min(crosswalk["xyz"][:, 1])
            x_max = np.max(crosswalk["xyz"][:, 0])
            y_max = np.max(crosswalk["xyz"][:, 1])

            crosswalks_bounds = np.append(
                crosswalks_bounds,
                np.asarray([[[x_min, y_min], [x_max, y_max]]]),
                axis=0,
            )
            crosswalks_ids.append(element_id)

    return {
        "lanes": {
            "bounds": lanes_bounds,
            "ids": lanes_ids,
            "center_line": center_line_coords,
            "xy_left": xy_left_coords,
            "xy_right": xy_right_coords,
            "xy_left_": xy_left_coords_,
            "xy_right_": xy_right_coords_,
            "lane_point_idx_2_cumul_distances": lane_point_idx_2_cumul_distances,
            "lane_point_idx_2_sin_cos_forward": lane_point_idx_2_sin_cos_forward,
            "lane_point_idx_2_sin_cos_backward": lane_point_idx_2_sin_cos_backward,
        },
        "crosswalks": {"bounds": crosswalks_bounds, "ids": crosswalks_ids},
    }


def precompute_lane_adjacencies(id_2_idx: Dict[str, int], proto_API: MapAPI):
    lane_adj_list_forward = [[] for _ in range(len(id_2_idx))]
    lane_adj_list_backward = [[] for _ in range(len(id_2_idx))]
    lane_adj_list_right = [[] for _ in range(len(id_2_idx))]
    lane_adj_list_left = [[] for _ in range(len(id_2_idx))]

    for element_id in tqdm(id_2_idx.keys(), desc="Computing lane adjacency lists"):
        lanes_ahead = [
            lane_id
            for lane_id in proto_API.get_lanes_ahead(element_id)
            if lane_id in id_2_idx
        ]
        lane_adj_list_forward[id_2_idx[element_id]].extend(lanes_ahead)
        for lane_ahead_id in lanes_ahead:
            lane_adj_list_backward[id_2_idx[lane_ahead_id]].append(element_id)

        lane_left = proto_API.get_lane_to_left(element_id)

        if lane_left != "" and lane_left in id_2_idx:
            lane_adj_list_left[id_2_idx[element_id]].append(lane_left)
        lane_right = proto_API.get_lane_to_right(element_id)
        if lane_right != "" and lane_left in id_2_idx:
            lane_adj_list_right[id_2_idx[element_id]].append(lane_right)

    return (
        lane_adj_list_forward,
        lane_adj_list_backward,
        lane_adj_list_right,
        lane_adj_list_left,
    )
