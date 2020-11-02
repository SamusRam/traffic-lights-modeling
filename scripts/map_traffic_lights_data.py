import warnings
warnings.filterwarnings('ignore')
import os
from lane_processing import *
import pickle
from l5kit.data import LocalDataManager, ChunkedDataset
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pytz import timezone
from datetime import datetime
from l5kit.data.filter import filter_tl_faces_by_status
import pandas as pd
from tqdm.auto import tqdm

from collections import defaultdict, deque

os.environ["L5KIT_DATA_FOLDER"] = "../input/"
CAR_CLASS = 3
BIKE_CLASS = 10
ALL_WHEELS_CLASS = -1
TL_GREEN_COLOR = 1
TL_RED_COLOR = 0
TL_YELLOW_COLOR = 0
VIS_WIP = False
LANE_IDLE_CODE = 0
LANE_ON_MOVE_CODE = 1


dm = LocalDataManager(None)

# ------------------------------------

# Checking junction info from the semantic map
semantic_map_path = dm.require(semantic_map_key)
proto_API = MapAPI(semantic_map_path, world_to_ecef)

if os.path.exists('../input/lanes_crosswalks.pkl'):
    with open('../input/lanes_crosswalks.pkl', 'rb') as f:
        lanes_crosswalks = pickle.load(f)
else:
    lanes_crosswalks = precompute_map_elements(proto_API)
    with open('../input/lanes_crosswalks.pkl', 'wb') as f:
        pickle.dump(lanes_crosswalks, f)


def get_lanes_dict_and_id_mapping(filter_function):
    lanes = dict()
    lanes_indices = [idx for idx, lane_id in enumerate(lanes_crosswalks['lanes']['ids'])
                     if filter_function(lane_id)]
        
    for key, values in lanes_crosswalks['lanes'].items():
        if len(lanes_crosswalks['lanes']['ids']) == len(values):
            lanes[key] = values[lanes_indices] if isinstance(values, (np.ndarray, np.generic)) else [values[i] for i in lanes_indices]

    lane_id_2_idx = {lane_id: i for i, lane_id in enumerate(lanes['ids'])}
    return lanes, lane_id_2_idx

lanes_not_bike, lane_id_2_idx_not_bike = get_lanes_dict_and_id_mapping(filter_function=lambda x: not proto_API.is_bike_only_lane(x))
lanes_bike, lane_id_2_idx_bike = get_lanes_dict_and_id_mapping(filter_function=lambda x: proto_API.is_bike_only_lane(x))
lane_id_2_idx = {lane_id: i for i, lane_id in enumerate(lanes_crosswalks['lanes']['ids'])}

lane_adj_list_forward_bike, lane_adj_list_backward_bike, lane_adj_list_right_bike, lane_adj_list_left_bike = precompute_lane_adjacencies(lane_id_2_idx_bike, proto_API)
lane_adj_list_forward_not_bike, lane_adj_list_backward_not_bike, lane_adj_list_right_not_bike, lane_adj_list_left_not_bike = precompute_lane_adjacencies(lane_id_2_idx_not_bike, proto_API)


# right turn under red fixes (initially assumed additional green arrow face)
# example: parent m+dt, 
#     leave tl control connection to li+4 (if more, the first one must be the closest to the turn lane); 
#     remove connection to Rvxh
#     lanes to yield on red-light turn:  ['nwfo', 'AAR1']]
turn_under_red_fixes = [['m+dt', ['li+4'], 'Rvxh', ['nwfo', 'AAR1']],
                        ['/E65', ['u016'], 'Waxo', ['qSSB', 'pQYu']],
                        ['T1zT', ['nicu'], 'hE9x', ['li+4']],
                        ['RRp7', ['9qCn'], 'le35', ['SYA4']],
                        ['/zLv', ['Pl2o'], '3KTc', ['4LEI']],
                        ['VUqQ', ['YeLI'], 'fMAY', ['/9cr', 'KtDh']],
                        ['cKUZ', ['x+C2'], 'b2Km', ['mTBO', 'JDFg']],
                        ['Myw0', ['ZjWH', '/9cr', 'biWH', '6iWH'], 'KHxd', ['x+C2', 'SE96']],
                        ['89HH', ['RbC9'], '5z+h', ['wqVO', 'yU+6']],
                       ['jMm1', ['KTUI'], 'wTjJ', ['SOK7', 'JhJI', 'UCr8']],
                        ['YkSE', ['SOK7'], '+Klg', ['RbC9', 'JAlK']],
                        ['ake8', ['z5yd'], '+SHD', ['xb0W', '/aG3', 'A+L+']],
                        ['vijX', ['A+L+'], 'H09x', ['u016', 'wc+d']],
                        ['hINg', ['2OcN', 'XOcN'], 'VPcN', ['z5yd', '8weJ']],
                        ['aVLS', ['zojk'], '+B39', ['I4aR']],
                        ['CVyh', ['1WGF', 'WWGF', 'wtG4', 'PuG4'], 'UXGF', ['zojk']],
                        ['emZY', ['84vV'], 'FWhR', []],
                        ['YCLp', ['zvWg'], 'L5L1', ['84vV']],
                        ['TnVm', ['nwfo'], 'hQc1', ['78bQ', 'M3Fu']],
                        ['DvGy', ['qgZI'], 'UB+4', ['+P/J', '9qCn']],
                        ['b75w', ['SYA4'], '4gKt', ['Q20G']]
                       ]

fix_i_2_junction_bro = dict()
red_turn_lane_2_yield_lanes = dict()
added_lanes = set()


def get_helping_angle(vector_1):
    vector_0 = np.array([1, 0])
    angle_cos = np.dot(vector_0, vector_1)/np.linalg.norm(vector_1)
    alpha = np.arccos(angle_cos)    
    if vector_1[1] < 0:
        alpha = 2*np.pi - alpha
    return alpha


def get_connecting_coordinates(start_point_coord, end_point_coord, total_dist=None, step_m=2):
    vector_1 = end_point_coord - start_point_coord
    helping_angle = get_helping_angle(vector_1)
    sin_, cos_ = np.sin(helping_angle), np.cos(helping_angle)
    if total_dist is None:
        total_dist = np.hypot(start_point_coord[0] - end_point_coord[0], start_point_coord[1] - end_point_coord[1])
    dist = 0
    current_point_coords = start_point_coord
    all_coords = [start_point_coord.copy()]
    while dist <= total_dist:
        current_point_coords += step_m*np.array([cos_, sin_])
        dist += step_m
        all_coords.append(current_point_coords.copy())
    return np.vstack(all_coords)


lane_id_2_idx_specialized = [lane_id_2_idx_bike, lane_id_2_idx_not_bike]
lane_adj_list_forward_specialized = [lane_adj_list_forward_bike, lane_adj_list_forward_not_bike]
lane_adj_list_backward_specialized = [lane_adj_list_backward_bike, lane_adj_list_backward_not_bike]
lane_adj_list_left_specialized = [lane_adj_list_left_bike, lane_adj_list_left_not_bike]
lane_adj_list_right_specialized = [lane_adj_list_right_bike, lane_adj_list_right_not_bike]
lanes_specialized = [lanes_bike, lanes_not_bike]

for fix_i, turn_under_red_fix in enumerate(turn_under_red_fixes):
    parent, lanes_active, lane_to_remove, turn_yield_lanes = turn_under_red_fix
    lane_spec = int(parent in lane_id_2_idx_not_bike)
    lane_id_2_idx_specialized[lane_spec][f'lane{2*fix_i}'] = len(lane_adj_list_forward_specialized[lane_spec])
    lane_adj_list_forward_specialized[lane_spec].append([lane_to_remove])
    lane_adj_list_backward_specialized[lane_spec].append([parent])
    lane_adj_list_left_specialized[lane_spec].append([])
    lane_adj_list_right_specialized[lane_spec].append([])

    lane_id_2_idx_specialized[lane_spec][f'lane{2*fix_i + 1}'] = len(lane_adj_list_forward_specialized[lane_spec])
    lane_adj_list_forward_specialized[lane_spec].append(lanes_active)
    lane_adj_list_backward_specialized[lane_spec].append([parent])
    lane_adj_list_left_specialized[lane_spec].append([])
    lane_adj_list_right_specialized[lane_spec].append([])

    lane_adj_list_forward_specialized[lane_spec][lane_id_2_idx_specialized[lane_spec][parent]] = [f'lane{2*fix_i}', f'lane{2*fix_i + 1}']

    for lane_id_left in lanes_active:
        lane_adj_list_backward_specialized[lane_spec][lane_id_2_idx_specialized[lane_spec][lane_id_left]] = [f'lane{2*fix_i + 1}'] + [x for x in lane_adj_list_backward_specialized[lane_spec][lane_id_2_idx_specialized[lane_spec][lane_id_left]] if x != parent]
    lane_adj_list_backward_specialized[lane_spec][lane_id_2_idx_specialized[lane_spec][lane_to_remove]] = [f'lane{2*fix_i}'] + [x for x in lane_adj_list_backward_specialized[lane_spec][lane_id_2_idx_specialized[lane_spec][lane_to_remove]] if x != parent]

    assert len(lanes_specialized[lane_spec]['center_line'][lane_id_2_idx_specialized[lane_spec][lane_to_remove]]) >= 4
    assert len(lanes_specialized[lane_spec]['center_line'][lane_id_2_idx_specialized[lane_spec][lanes_active[0]]]) >= 5
    
    lanes_specialized[lane_spec]['ids'].extend([f'lane{2*fix_i}', f'lane{2*fix_i + 1}'])
    lanes_crosswalks['lanes']['ids'].extend([f'lane{2*fix_i}', f'lane{2*fix_i + 1}'])
    added_lanes.update({f'lane{2*fix_i}', f'lane{2*fix_i + 1}'})
    
    # last 3 points of parent to be forked
    parent_center_line = lanes_crosswalks['lanes']['center_line'][lane_id_2_idx[parent]]
    init_parent_last_coord = parent_center_line[-1].copy()
    lanes_specialized[lane_spec]['center_line'][lane_id_2_idx_specialized[lane_spec][parent]] = parent_center_line[:max(1, len(parent_center_line)-3)]
    lanes_crosswalks['lanes']['center_line'][lane_id_2_idx[parent]] = parent_center_line[:max(1, len(parent_center_line)-3)]
    new_parent_last_coord = lanes_specialized[lane_spec]['center_line'][lane_id_2_idx_specialized[lane_spec][parent]][-1].copy()
    total_parent_segment_len = np.hypot(new_parent_last_coord[0] - init_parent_last_coord[0], 
                                        new_parent_last_coord[1] - init_parent_last_coord[1])
    # the turn lane to remove
    directional_point_for_fork = lanes_specialized[lane_spec]['center_line'][lane_id_2_idx_specialized[lane_spec][lane_to_remove]][3]
    turn_fork_center_line = get_connecting_coordinates(new_parent_last_coord, directional_point_for_fork, 
                                                       total_dist=total_parent_segment_len)
                               
    lanes_specialized[lane_spec]['center_line'].append(turn_fork_center_line)
    lanes_crosswalks['lanes']['center_line'].append(turn_fork_center_line) 
    # dummy boarder in order not to break vis
    lanes_specialized[lane_spec]['xy_left_'].append(turn_fork_center_line)
    lanes_specialized[lane_spec]['xy_right_'].append(turn_fork_center_line)
    # shouldn't be used from lanes_crosswalks['lanes']
    
    # the first lane to leave tl control must be the closest one to the turn-lane
    directional_point_for_fork = lanes_specialized[lane_spec]['center_line'][lane_id_2_idx_specialized[lane_spec][lanes_active[0]]][4]
    new_parent_last_coord = lanes_specialized[lane_spec]['center_line'][lane_id_2_idx_specialized[lane_spec][parent]][-1].copy()
    straight_center_line = get_connecting_coordinates(new_parent_last_coord, directional_point_for_fork, 
                                                       total_dist=total_parent_segment_len)
                               
    lanes_specialized[lane_spec]['center_line'].append(straight_center_line)  
    lanes_crosswalks['lanes']['center_line'].append(straight_center_line)  
    # dummy boarder in order not to break vis
    lanes_specialized[lane_spec]['xy_left_'].append(straight_center_line)
    lanes_specialized[lane_spec]['xy_right_'].append(straight_center_line)

    fix_i_2_junction_bro[fix_i] = parent
    # TODO!: each red_turn_lane has "stop-sign"
    red_turn_lane_2_yield_lanes[f'lane{2*fix_i}'] = turn_yield_lanes


def get_kd_tree_and_idx_map(lanes):
    kd_tree = KDTree(np.concatenate(lanes['center_line'], axis=0))
    kd_idx_2_lane_id_idx = []
    for lane_id, center_line in zip(lanes['ids'],
                                    lanes['center_line']):
        next_entries = [(lane_id, i) for i in range(len(center_line))]
        kd_idx_2_lane_id_idx.extend(next_entries)
    return kd_tree, kd_idx_2_lane_id_idx    
    
kd_tree_bike, kd_idx_2_lane_id_idx_bike = get_kd_tree_and_idx_map(lanes_bike)
kd_tree_not_bike, kd_idx_2_lane_id_idx_not_bike = get_kd_tree_and_idx_map(lanes_not_bike)
kd_tree, kd_idx_2_lane_id_idx = get_kd_tree_and_idx_map(lanes_crosswalks['lanes'])


tl_control_2_junctions = defaultdict(list)
junction_2_tl_controls = defaultdict(list)
lane_2_junctions = defaultdict(list)
junction_2_lanes = defaultdict(list)
for element in proto_API:
    if MapAPI.is_junction(element):
        junction_element_id = MapAPI.id_as_str(element.id)
        traffic_control_elements = proto_API.get_traffic_control_elements_at_junction(junction_element_id)
        junction_2_tl_controls[junction_element_id].extend(traffic_control_elements)
        for tl_control in traffic_control_elements:
            tl_control_2_junctions[tl_control].append(junction_element_id)
            
        lanes_related_to_junction = proto_API.get_all_lanes_at_junction(junction_element_id)
        for lane in lanes_related_to_junction:
            junction_2_lanes[junction_element_id].append(lane)
            lane_2_junctions[lane].append(junction_element_id)
# to address potential duplications
for lane_id, junctions in lane_2_junctions.items():
    lane_2_junctions[lane_id] = list(set(junctions))
for junction_id, lanes in junction_2_lanes.items():
    junction_2_lanes[junction_id] = list(set(lanes))
    
for fix_i in range(len(turn_under_red_fixes)):
    junction_neighb = fix_i_2_junction_bro[fix_i]
    lane_2_junctions[f'lane{2*fix_i}'] = lane_2_junctions[junction_neighb].copy()
    lane_2_junctions[f'lane{2*fix_i + 1}'] = lane_2_junctions[junction_neighb].copy()
    for junction_id in lane_2_junctions[junction_neighb]:
        junction_2_lanes[junction_id].extend([f'lane{2*fix_i}', f'lane{2*fix_i + 1}'])

if os.path.exists('../traffic_light_ids_all.pkl'):
    with open('../traffic_light_ids_all.pkl', 'rb') as f:
        traffic_light_ids_all = pickle.load(f)
else:
    dataset_path = 'scenes/train_full_filtered_min_frame_history_4_min_frame_future_1_with_mask_idx.zarr'
    train_full_zarr = ChunkedDataset(dm.require(dataset_path)).open()
    traffic_light_ids_all = set(train_full_zarr.tl_faces['traffic_light_id'])
    with open('../traffic_light_ids_all.pkl', 'wb') as f:
        pickle.dump(traffic_light_ids_all, f)

if os.path.exists('../traffic_faces_ids_all.pkl'):
    with open('../traffic_faces_ids_all.pkl', 'rb') as f:
        traffic_faces_ids_all = pickle.load(f)
else:
    dataset_path = 'scenes/train_full_filtered_min_frame_history_4_min_frame_future_1_with_mask_idx.zarr'
    train_full_zarr = ChunkedDataset(dm.require(dataset_path)).open()
    traffic_faces_ids_all = set(train_full_zarr.tl_faces['face_id'])
    with open('../traffic_faces_ids_all.pkl', 'wb') as f:
        pickle.dump(traffic_faces_ids_all, f)

traffic_light_id_2_coord = dict()
tl_ids_without_sem_map_info = []
for el_id in traffic_light_ids_all:
    if el_id in proto_API.ids_to_el:
        coordinates = proto_API.get_traffic_light_coords(el_id)['xyz']
        traffic_light_id_2_coord[el_id] = coordinates[:, :2].mean(axis=0)
    else:
        tl_ids_without_sem_map_info.append(el_id)


added_lanes_tl_coord_tl_id = [[['Myw0'], (684, -407)],
                       [['YCLp', 'H1RI'], (630, -2242)],
                       [['m1RI'], (630, -2242)],
                       [['ii1t'], (634, -2268)],
                       [['TnVm', 'urF8', 'PrF8'], (635, -1355)],
                       [['wqF8'], (636, -1358)],
                       [['T1zT', 'udk+', 'Pdk+'], (672, -1342)],
                       [['wck+'], (667, -1339)],
                       [['wmAx', 'RRp7'], (465, -160)],
                       [['CPtm'], (439, -190)],
                       [['8OiE', 'FHu2'], (249, 82)],
                       [['6PiE'], (213, 56)],
                       [['WlSE', 'pXNd', 'YkSE'],  (202, 63)],
                       [['RLKQ'], (-35, 350)],
                       [['jUyh', 'CVyh'], (-758, 1127)],
                       [['57g+'], (-733, 1125)],
                       [['W9g+', '38g+', 'Y8g+'], (-735, 1122)],
                       [['mzvz'], (-491, 896)],
                       [['F0vz', 'k0vz'], (-488, 894)],
                       [['90Lv'], (-527, 874)],
                       [['e0Lv', '/zLv'], (-530, 876)],
                              [['T3uS', 'y3uS'], (-973.5, 1360)],
                             [['RJu6'],  (-974, 1359)],
                              [['8V6+'], (-950, 1440)], 
                              [['LXvy'], (-510, 875)],
                             [['YVHi'], (-516, 901)],
                              [['9ie8'], (741, -463)]
                             ]  
for i in range(len(added_lanes_tl_coord_tl_id)):
    added_tl_id = f'Luda_{i}'
    added_lanes_tl_coord_tl_id[i].append(added_tl_id)
    traffic_light_id_2_coord[added_tl_id] = np.array(added_lanes_tl_coord_tl_id[i][1])
    traffic_light_ids_all.add(added_tl_id)
    for lane_id in added_lanes_tl_coord_tl_id[i][0]:
        tl_control_2_junctions[added_tl_id].extend(lane_2_junctions[lane_id])
        for junction in lane_2_junctions[lane_id]:
            junction_2_tl_controls[junction].append(added_tl_id)


tl_control_2_close_lanes = defaultdict(list)
for tl_control, junctions in tl_control_2_junctions.items():
    if tl_control in traffic_light_ids_all or tl_control in traffic_faces_ids_all:
        for junction in junctions:
            tl_control_2_close_lanes[tl_control].extend(junction_2_lanes[junction])
            # predecessor lanes
            for junction_lane in junction_2_lanes[junction]:
                if junction_lane in lane_id_2_idx_bike:
                    tl_control_2_close_lanes[tl_control].extend(lane_adj_list_backward_bike[lane_id_2_idx_bike[junction_lane]])
                elif junction_lane in lane_id_2_idx_not_bike:
                    tl_control_2_close_lanes[tl_control].extend(lane_adj_list_backward_not_bike[lane_id_2_idx_not_bike[junction_lane]])
        tl_control_2_close_lanes[tl_control] = list(set(tl_control_2_close_lanes[tl_control]))
    
lane_2_close_tl_controls = defaultdict(list)
for tl_control, close_lanes in tl_control_2_close_lanes.items():
    for lane in close_lanes:
        lane_2_close_tl_controls[lane].append(tl_control)
        
lane_2_tl_related_lanes_wip = defaultdict(set)
for lane_id, close_tl_controls in lane_2_close_tl_controls.items():
    for traffic_control_id in close_tl_controls:
        lane_2_tl_related_lanes_wip[lane_id].update(tl_control_2_close_lanes[traffic_control_id])

lane_ids_with_wrong_tl_associations = {'i2bp', 'AxmM', 'LXvy', 'YVHi', '8V6+', 'Bm9O', '9ie8'}

tl_added_lane_id_2_tl_id = dict()
for lanes, _, tl_id in added_lanes_tl_coord_tl_id:
    for lane_id in lanes:
        assert lane_id not in tl_added_lane_id_2_tl_id
        tl_added_lane_id_2_tl_id[lane_id] = tl_id
        lane_2_close_tl_controls[lane_id].append(tl_id)


def get_lane_center_line(lane_id):
    if lane_id in lane_id_2_idx_bike:
        return lanes_bike['center_line'][lane_id_2_idx_bike[lane_id]]
    if lane_id in lane_id_2_idx_not_bike:
        return lanes_not_bike['center_line'][lane_id_2_idx_not_bike[lane_id]]
    else:
        raise ValueError('Lane neither bike only, nor others')


def get_lane_len(lane_id):
    return len(get_lane_center_line(lane_id))


lane_2_direct_tl_lights = defaultdict(set)
lane_2_direct_tl_faces = defaultdict(set)
# only lane controlled by traffic light and its predecessor(s) [at least min_required_len_before_tl center line points in total]
min_required_len_before_tl = 30

def get_lane_predecessors(lane_id):
    if lane_id in lane_id_2_idx_bike:
        return lane_adj_list_backward_bike[lane_id_2_idx_bike[lane_id]]
    elif lane_id in lane_id_2_idx_not_bike:
        return lane_adj_list_backward_not_bike[lane_id_2_idx_not_bike[lane_id]]
    else:
        return []


def get_lane_successors(lane_id):
    if lane_id in lane_id_2_idx_bike:
        return lane_adj_list_forward_bike[lane_id_2_idx_bike[lane_id]]
    elif lane_id in lane_id_2_idx_not_bike:
        return lane_adj_list_forward_not_bike[lane_id_2_idx_not_bike[lane_id]]
    else:
        return []


def get_lane_left_neighbours(lane_id):
    if lane_id in lane_id_2_idx_bike:
        return lane_adj_list_left_bike[lane_id_2_idx_bike[lane_id]]
    elif lane_id in lane_id_2_idx_not_bike:
        return lane_adj_list_left_not_bike[lane_id_2_idx_not_bike[lane_id]]
    else:
        return []


def get_lane_right_neighbours(lane_id):
    if lane_id in lane_id_2_idx_bike:
        return lane_adj_list_right_bike[lane_id_2_idx_bike[lane_id]]
    elif lane_id in lane_id_2_idx_not_bike:
        return lane_adj_list_right_not_bike[lane_id_2_idx_not_bike[lane_id]]
    else:
        return []


def get_lane_neighbours_all(lane_id):
    return (get_lane_right_neighbours(lane_id) +
            get_lane_left_neighbours(lane_id) +
            get_lane_successors(lane_id) +
            get_lane_predecessors(lane_id))


def get_lane_neighbours_based_on_dist(lane_id, dist_max_m=10):
    neighbour_lanes = set()
    for lane_point in get_lane_center_line(lane_id):
        kd_indices = kd_tree.query_ball_point(lane_point, r=dist_max_m)
        for kd_idx in kd_indices:
            lane_id_neighb, _ = get_lane_point_from_kdtree(kd_idx, ALL_WHEELS_CLASS)
            neighbour_lanes.add(lane_id_neighb)
    return neighbour_lanes


lanes_under_red_turn_fix = [lane_under_red_turn_fix for lane_under_red_turn_fix, _, _, _ in turn_under_red_fixes]

for element in proto_API:
    if proto_API.is_lane(element):
        lane_id = MapAPI.id_as_str(element.id)
        if lane_id not in lane_ids_with_wrong_tl_associations:
            lane_traffic_controls = [MapAPI.id_as_str(x) for x in proto_API.get_lane_traffic_controls(lane_id)]
        else:
            lane_traffic_controls = []
        lane_traffic_lights = [x for x in lane_traffic_controls if x in traffic_light_ids_all]
        lane_traffic_faces = [x for x in lane_traffic_controls if x in traffic_faces_ids_all]
        tl_faces_elements = [proto_API[x].element.traffic_control_element for x in lane_traffic_faces]
            
        if lane_id in tl_added_lane_id_2_tl_id:
            lane_traffic_lights.append(tl_added_lane_id_2_tl_id[lane_id])
            lane_traffic_faces.append(tl_added_lane_id_2_tl_id[lane_id])
        if len(lane_traffic_lights):
            dist_from_tl = get_lane_len(lane_id)
            queue = deque() 
            queue.append((lane_id, dist_from_tl))
            while len(queue):
                next_lane, dist_from_tl = queue.popleft()
                lane_2_direct_tl_lights[next_lane].update(lane_traffic_lights)
                if dist_from_tl == get_lane_len(next_lane): # don't propagate faces backwards
                    lane_2_direct_tl_faces[next_lane].update(lane_traffic_faces)
                if dist_from_tl < min_required_len_before_tl and next_lane not in lanes_under_red_turn_fix:
                    lane_predecessors = get_lane_predecessors(next_lane)
                    for lane_predecessor_id in lane_predecessors:
                        queue.append((lane_predecessor_id, dist_from_tl + get_lane_len(lane_predecessor_id)))
                        
# manual sem map fixes
for fix_i, parent_lane in enumerate(lanes_under_red_turn_fix):
    lane_2_direct_tl_lights[f'lane{2*fix_i + 1}'] = lane_2_direct_tl_lights[parent_lane].copy()
    lane_2_direct_tl_faces[f'lane{2*fix_i + 1}'] = lane_2_direct_tl_faces[parent_lane].copy()
    del lane_2_direct_tl_lights[parent_lane]
    del lane_2_direct_tl_faces[parent_lane]
    

tl_light_2_directly_controlled_lanes = defaultdict(set)
for lane_id, tl_lights in lane_2_direct_tl_lights.items():
    for tl_light in tl_lights:
        tl_light_2_directly_controlled_lanes[tl_light].add(lane_id)


master_intersection_idx_2_traffic_lights = []
traffic_light_id_2_master_intersection_idx = defaultdict(list)

for tls_united_by_lane in lane_2_close_tl_controls.values():
    master_intersection_idx = None
    for tl_id in tls_united_by_lane:
        if tl_id in traffic_light_id_2_master_intersection_idx:
            if master_intersection_idx is not None:
                assert traffic_light_id_2_master_intersection_idx[tl_id] == master_intersection_idx, (f'tl_id: {tl_id}', f'traffic_light_id_2_master_intersection_idx[tl_id]: {traffic_light_id_2_master_intersection_idx[tl_id]}', 
                                                                                                      f'master_intersection_idx: {master_intersection_idx}')
            else:
                master_intersection_idx = traffic_light_id_2_master_intersection_idx[tl_id]
            assert tl_id in master_intersection_idx_2_traffic_lights[master_intersection_idx]
        else:
            if master_intersection_idx is None:
                master_intersection_idx = len(master_intersection_idx_2_traffic_lights)
                master_intersection_idx_2_traffic_lights.append(set())
            traffic_light_id_2_master_intersection_idx[tl_id] = master_intersection_idx            
            master_intersection_idx_2_traffic_lights[master_intersection_idx].add(tl_id)


master_intersection_idx_2_traffic_lights = [set(intersection_tls) for intersection_tls in sorted([tuple(sorted(list(x))) for x in master_intersection_idx_2_traffic_lights])]
# splitting the master intersection 1 (there's some long connecting lane likely)
from sklearn.cluster import KMeans
orig_tl_ids = list(master_intersection_idx_2_traffic_lights[3])
coordinates_data = np.array([traffic_light_id_2_coord[tl_id] for tl_id in orig_tl_ids])
kmeans_clusters = KMeans(n_clusters=2, random_state=42).fit(coordinates_data).labels_
master_intersection_idx_new = len(master_intersection_idx_2_traffic_lights)
master_intersection_idx_2_traffic_lights.append(set())
for tl_id, cluster in zip(orig_tl_ids, kmeans_clusters):
    if cluster == 1:
        traffic_light_id_2_master_intersection_idx[tl_id] =  master_intersection_idx_new           
        master_intersection_idx_2_traffic_lights[master_intersection_idx_new].add(tl_id)
        master_intersection_idx_2_traffic_lights[3].remove(tl_id)
        

def get_lane_boarders(lane_id):
    if lane_id in lane_id_2_idx_bike:
        return lanes_bike['xy_left_'][lane_id_2_idx_bike[lane_id]], lanes_bike['xy_right_'][lane_id_2_idx_bike[lane_id]]
    elif lane_id in lane_id_2_idx_not_bike:
        return lanes_not_bike['xy_left_'][lane_id_2_idx_not_bike[lane_id]], lanes_not_bike['xy_right_'][lane_id_2_idx_not_bike[lane_id]]
    else:
        raise ValueError('Lane nether bike only, nor others')


def is_the_end_controlled_lane(lane_id, controlled_lanes):
    the_end_lane = True
    for next_lane_id in get_lane_successors(lane_id):
        # small, O(n) is ok
        if next_lane_id in controlled_lanes:
            the_end_lane = False
            break
    return the_end_lane


def is_predecessor(lane_id, candidate_predecessor_lane_id, max_depth=15):
    # bfs
    queue = deque() 
    queue.append((candidate_predecessor_lane_id, 0))
    while len(queue):
        next_lane, depth = queue.popleft()
        if depth + 1 < max_depth: 
            lanes_next = get_lane_successors(next_lane)
            for lane_next_id in lanes_next:
                if lane_next_id == lane_id:
                    return True
                queue.append((lane_next_id, depth + 1))
    return False
    

# the goal is to group traffic light faces controlling the same lanes (assuming they duplicate the same info)
tl_faces_set_2_lanes = defaultdict(set)
tl_faces_set_2_master_intersections = defaultdict(set)
# controlled_lanes_set_2_tl_face_ids = defaultdict(list)

# dummy count for the check only
tl_ids_raw_count = 0
for master_intersection_idx, master_intersection_traffic_lights in enumerate(master_intersection_idx_2_traffic_lights):
    tl_ids_raw_count += len(master_intersection_traffic_lights)
    for tl_id in master_intersection_traffic_lights:
        # all lanes directly controlled by the tl
        controlled_lanes = tl_light_2_directly_controlled_lanes[tl_id]
        
        # differentiate lanes based on tl faces, which are available for the final exit lanes only
        for lane_id in controlled_lanes:
            if is_the_end_controlled_lane(lane_id, controlled_lanes):
                tl_faces_tuple = tuple(sorted(lane_2_direct_tl_faces[lane_id]))
                tl_faces_set_2_lanes[tl_faces_tuple].add(lane_id)
                # propagate the tl-faces set to the predecessors
                for controlled_lane_id in controlled_lanes:
                    if is_predecessor(lane_id, controlled_lane_id):
                        tl_faces_set_2_lanes[tl_faces_tuple].add(controlled_lane_id)
                tl_faces_set_2_master_intersections[tl_faces_tuple].add(master_intersection_idx)


for tl_faces, intersection_idx_set in tl_faces_set_2_master_intersections.items():
    assert len(intersection_idx_set) == 1, f'{tl_faces}, {intersection_idx_set}'
    tl_faces_set_2_master_intersections[tl_faces] = list(intersection_idx_set)[0]
# tl_signal_idx corresponds to the set of tl faces with the same set of lanes under control and therefore (presumably) the same signal

controlled_lane_id_2_tl_signal_idx = dict()
exit_lane_id_2_tl_signal_idx = dict()
tl_signal_idx_2_controlled_lanes = []
tl_signal_idx_2_exit_lanes = []
tl_signal_idx_2_stop_coordinates = []
master_intersection_idx_2_tl_signal_indices = defaultdict(list)
tl_face_id_2_tl_signal_indices = defaultdict(set)

min_required_len_after_stop_line = 7

for tl_face_ids, controlled_lanes in sorted(tl_faces_set_2_lanes.items(), key=lambda x: x[0]):
    tl_signal_idx = len(tl_signal_idx_2_controlled_lanes)
    for lane_id in controlled_lanes:
        controlled_lane_id_2_tl_signal_idx[lane_id] = tl_signal_idx 
    tl_signal_idx_2_controlled_lanes.append(set(controlled_lanes))
    master_intersection_idx_2_tl_signal_indices[tl_faces_set_2_master_intersections[tl_face_ids]].append(tl_signal_idx)
    stop_coordinates = []
    tl_signal_idx_2_exit_lanes.append(set())
    for lane_id in controlled_lanes:
        
        if is_the_end_controlled_lane(lane_id, controlled_lanes):
            stop_coordinates.append(get_lane_center_line(lane_id)[-1])
            
            dist_from_stop_line = 0
            queue = deque() 
            queue.append((lane_id, dist_from_stop_line))
            while len(queue):
                next_lane, dist_from_stop_line = queue.popleft()
                if dist_from_stop_line != 0: # not the end lane
                    tl_signal_idx_2_exit_lanes[tl_signal_idx].add(next_lane)
                    exit_lane_id_2_tl_signal_idx[next_lane] = tl_signal_idx
                if dist_from_stop_line < min_required_len_after_stop_line:
                    lanes_next = get_lane_successors(next_lane)
                    for lane_next_id in lanes_next:
                        queue.append((lane_next_id, dist_from_stop_line + get_lane_len(lane_next_id)))
                
                
    tl_signal_idx_2_stop_coordinates.append(stop_coordinates)
    for tl_face_id in tl_face_ids:
        tl_face_id_2_tl_signal_indices[tl_face_id].add(tl_signal_idx)


lane_2_master_intersection_related_lanes = dict() #->set
lane_id_2_master_intersection_idx = dict()
for master_intersection_idx, master_intersection_tl_signal_indices in master_intersection_idx_2_tl_signal_indices.items():
    all_master_intersection_lanes = set()
    for tl_signal_idx in master_intersection_tl_signal_indices:
        all_master_intersection_lanes.update(tl_signal_idx_2_controlled_lanes[tl_signal_idx])
        all_master_intersection_lanes.update(tl_signal_idx_2_exit_lanes[tl_signal_idx])
        for lane_id in tl_signal_idx_2_controlled_lanes[tl_signal_idx].union(tl_signal_idx_2_exit_lanes[tl_signal_idx]):
            lane_id_2_master_intersection_idx[lane_id] = master_intersection_idx
        
        # to also associate the neighbourhood to an intersection, bfs from the controlled lanes (mainly to include the artificially separated turns-under-red)
        lanes_closed_set = tl_signal_idx_2_controlled_lanes[tl_signal_idx].union(tl_signal_idx_2_exit_lanes[tl_signal_idx])
        for lane_id in tl_signal_idx_2_controlled_lanes[tl_signal_idx]:
            dist_from_exit = 0
            queue = deque() 
            queue.append((lane_id, dist_from_exit))
            while len(queue):
                next_lane_id, dist_from_exit = queue.popleft()
                lane_id_2_master_intersection_idx[next_lane_id] = master_intersection_idx
                if dist_from_exit < 70: # points
                    lanes_next = get_lane_successors(next_lane_id) + get_lane_left_neighbours(next_lane_id) + get_lane_right_neighbours(next_lane_id)
                    for further_lane_id in lanes_next:
                        if further_lane_id not in lanes_closed_set:
                            queue.append((further_lane_id, dist_from_exit + get_lane_len(further_lane_id)))
                            lanes_closed_set.add(further_lane_id)
    for lane_id in all_master_intersection_lanes:
        lane_2_master_intersection_related_lanes[lane_id] = {neighbour_lane_id for neighbour_lane_id in all_master_intersection_lanes 
                                                             if neighbour_lane_id != lane_id}

# added lanes belong to intersections
lanes_not_bike_intersection, lane_id_2_idx_not_bike_intersection = get_lanes_dict_and_id_mapping(filter_function=lambda x: x in added_lanes or (not proto_API.is_bike_only_lane(x) and x in lane_id_2_master_intersection_idx))
lanes_bike_intersection, lane_id_2_idx_bike_intersection = get_lanes_dict_and_id_mapping(filter_function=lambda x: x in added_lanes or (proto_API.is_bike_only_lane(x) and x in lane_id_2_master_intersection_idx))
lanes_intersection, lane_id_2_idx_intersection = get_lanes_dict_and_id_mapping(filter_function=lambda x: x in added_lanes or x in lane_id_2_master_intersection_idx)

kd_tree_bike_intersection, kd_idx_2_lane_id_idx_bike_intersection = get_kd_tree_and_idx_map(lanes_bike_intersection)
kd_tree_not_bike_intersection, kd_idx_2_lane_id_idx_not_bike_intersection = get_kd_tree_and_idx_map(lanes_not_bike_intersection)
kd_tree_intersection, kd_idx_2_lane_id_idx_intersection = get_kd_tree_and_idx_map(lanes_intersection)

def get_traffic_light_coordinates(tl_el_id):
    if el_id in proto_API.ids_to_el:
        coordinates = proto_API.get_traffic_light_coords(el_id)['xyz']
        return coordinates[:, :2].mean(axis=0)


def get_closest_lanes(coord, agent_class, k_nearest, intersections_only=False):
    # TODO: distance_upper_bound kd_tree param
    if agent_class == BIKE_CLASS:
        if intersections_only:
            return kd_tree_bike_intersection.query(coord, k=k_nearest)
        return kd_tree_bike.query(coord, k=k_nearest)
    elif agent_class == CAR_CLASS:
        if intersections_only:
            return kd_tree_not_bike_intersection.query(coord, k=k_nearest)
        return kd_tree_not_bike.query(coord, k=k_nearest)
    elif agent_class == ALL_WHEELS_CLASS:
        if intersections_only:
            return kd_tree_intersection.query(coord, k=k_nearest)
        return kd_tree.query(coord, k=k_nearest)
    else:
        raise NotImplementedError('Only bikes and general cars supported')
        
def get_lane_point_from_kdtree(candidate_idx, agent_class, intersections_only=False):
    if agent_class == BIKE_CLASS:
        if intersections_only:
            return kd_idx_2_lane_id_idx_bike_intersection[candidate_idx]
        return kd_idx_2_lane_id_idx_bike[candidate_idx]
    elif agent_class == CAR_CLASS:
        if intersections_only:
            return kd_idx_2_lane_id_idx_not_bike_intersection[candidate_idx]
        return kd_idx_2_lane_id_idx_not_bike[candidate_idx]
    elif agent_class == ALL_WHEELS_CLASS:
        if intersections_only:
            return kd_idx_2_lane_id_idx_intersection[candidate_idx]
        return kd_idx_2_lane_id_idx[candidate_idx]


def cos_dist(x1, y1, yaw_rad):
    x2 = np.cos(yaw_rad)
    y2 = np.sin(yaw_rad)
    vector_product = x1 * x2 + y1 * y2
    cos = vector_product / (np.hypot(x1, y1) * np.hypot(x2, y2))
    return 1 - cos


def is_end_of_maneuver_lane(point_idx, lane_len, overlapped_last_points_count=5, min_len_maneuver_lane=7):
    return lane_len > min_len_maneuver_lane and point_idx >= lane_len - 1 - overlapped_last_points_count


def find_closest_lane(coord, yaw, agent_class, lane_point_2_blocked_lanes_set=None,
                      max_dist_m=4, k_nearest=25, return_point_i=False, 
                      return_blocked_tl_signals=False, blocked_dist_threshold_m=1.5, 
                      too_close_insensitivity_m=1.3, min_cos_dist_threshold=np.pi/2,
                      intersections_only=False, vis=False):
    if vis: 
        plt.figure(figsize=(15, 15))
    candidate_distances, candidate_indices = get_closest_lanes(coord, agent_class, k_nearest, intersections_only)    
    
    closed_set = set()
    min_cos_dist = float('inf')
    min_dist = candidate_distances[0]
    closest_lane_current_i = None
    result = None
    
    # processing in increasing order of L2
    def is_best_dist(lane_cos_dist, min_cos_dist, dist):
        if dist == min_dist:
            deadband = 0
        elif dist <= too_close_insensitivity_m:
            deadband = 0.001
        else:
            deadband = 0.05
        return lane_cos_dist + deadband < min_cos_dist
    
    for i, candidate_idx in enumerate(candidate_indices):
        lane_id, point_idx = get_lane_point_from_kdtree(candidate_idx, agent_class, intersections_only)
        if candidate_distances[i] > max_dist_m:
            break
        if lane_id not in closed_set:
            closed_set.add(lane_id)
            lane_center_line = get_lane_center_line(lane_id)
            lane_cos_dists =[]
            if point_idx > 0:
                x1_prev = lane_center_line[point_idx][0] - lane_center_line[point_idx - 1][0]
                y1_prev = lane_center_line[point_idx][1] - lane_center_line[point_idx - 1][1] 
                lane_cos_dists.append(cos_dist(x1_prev, y1_prev, yaw))
            if point_idx + 1 < len(lane_center_line):
                x1_next = lane_center_line[point_idx + 1][0] - lane_center_line[point_idx][0]
                y1_next = lane_center_line[point_idx + 1][1] - lane_center_line[point_idx][1] 
                lane_cos_dists.append(cos_dist(x1_next, y1_next, yaw))   
                
            lane_cos_dist = np.mean(lane_cos_dists)
            
            if vis: 
                plt.scatter(lane_center_line[:, 0], lane_center_line[:, 1], label=f'{lane_id} ({lane_cos_dist:.6f} ({is_best_dist(lane_cos_dist, min_cos_dist, candidate_distances[i])}) ({lane_cos_dists}), {candidate_distances[i]:.2f} m')
            if is_best_dist(lane_cos_dist, min_cos_dist, candidate_distances[i]):
                min_cos_dist = lane_cos_dist
                closest_lane_current_i = i
                result = (lane_id, point_idx)
    if vis: 
        plt.scatter(coord[0], coord[1], c='yellow', s=30)
        plt.arrow(coord[0], coord[1], 3*np.cos(yaw), 3*np.sin(yaw), color='r')
        plt.legend()
        plt.title(f'{result}, {lane_point_2_blocked_lanes_set.get(result, set()) if lane_point_2_blocked_lanes_set is not None else ""}')

    if result is None or min_cos_dist >= min_cos_dist_threshold:
        return None
    closest_lane = result[0]
                    
    if return_blocked_tl_signals:
        if lane_point_2_blocked_lanes_set is None:
            last_blocked_i = closest_lane_current_i
            while (last_blocked_i < len(candidate_distances) and 
                   candidate_distances[last_blocked_i] < candidate_distances[closest_lane_current_i] + blocked_dist_threshold_m):
                last_blocked_i += 1
            blocked_lane_candidates = [get_lane_point_from_kdtree(idx, agent_class, intersections_only) for idx in candidate_indices[:last_blocked_i]]

            blocked_lane_candidates = [lane_id for lane_id, lane_point_i in blocked_lane_candidates
                                      if not is_end_of_maneuver_lane(lane_point_i, get_lane_len(lane_id))]
            # to avoid blocking its own tl:
            if closest_lane in exit_lane_id_2_tl_signal_idx:
                tl_signal_idx_closest_lane = exit_lane_id_2_tl_signal_idx[closest_lane] 
                blocked_signals = {exit_lane_id_2_tl_signal_idx[lane_id_] for lane_id_ in blocked_lane_candidates 
                                   if (lane_id_ in exit_lane_id_2_tl_signal_idx 
                                       and exit_lane_id_2_tl_signal_idx[lane_id_] != tl_signal_idx_closest_lane)}
            else:
                blocked_signals = set()
        else:
            blocked_signals = lane_point_2_blocked_lanes_set.get(result, set())
        if not return_point_i:
            return closest_lane, blocked_signals
        return result, blocked_signals
    if not return_point_i:
        return closest_lane 
    return result

lane_point_2_blocked_lanes_set = dict()

checked_hard_cases = {('csUz', 2), ('RmBi', 7), ('RmBi', 6)}

for i, lane_id in tqdm(enumerate(exit_lane_id_2_tl_signal_idx.keys()), desc='Lane blocked sets..'):
    lane_len = get_lane_len(lane_id)
    lane_center_line = get_lane_center_line(lane_id)
    assert lane_len == len(lane_center_line)
    blocked_set_so_far = set()
    for point_idx in range(lane_len - 1, -1, -1):  
        yaws =[]
        if point_idx > 0:
            x1_prev = lane_center_line[point_idx][0] - lane_center_line[point_idx - 1][0]
            y1_prev = lane_center_line[point_idx][1] - lane_center_line[point_idx - 1][1] 
            yaws.append(get_helping_angle(np.array([x1_prev, y1_prev])))
        if point_idx + 1 < lane_len:
            x1_next = lane_center_line[point_idx + 1][0] - lane_center_line[point_idx][0]
            y1_next = lane_center_line[point_idx + 1][1] - lane_center_line[point_idx][1] 
            yaws.append(get_helping_angle(np.array([x1_next, y1_next])))
        # 1st and 4th quadrants boarder
        yaws = sorted(yaws)
        if len(yaws) > 1 and yaws[0] < np.pi/2 and yaws[1] > 3*np.pi/2:
            yaws[1] = yaws[1] - 2*np.pi
        yaw = np.mean(yaws)

        (closest_lane_id, closest_lane_point_i), blocked_tl_signals = find_closest_lane(lane_center_line[point_idx], yaw, 
                                                                                          agent_class=ALL_WHEELS_CLASS, 
                                                                                          k_nearest=15, return_point_i=True, 
                                                                                          return_blocked_tl_signals=True, 
                                                                                       intersections_only=True)
        assert (lane_id, point_idx) in checked_hard_cases or closest_lane_id == lane_id and point_idx == closest_lane_point_i, f'closest_lane_id: {closest_lane_id}[{closest_lane_point_i}], (true lane_id: {lane_id}[{point_idx}])'
        blocked_set_so_far.update(blocked_tl_signals)  
        lane_point_2_blocked_lanes_set[(lane_id, point_idx)] = deepcopy(blocked_set_so_far)


def get_info_per_related_lanes(frame_sample):
    timestamp = datetime.fromtimestamp(frame_sample['timestamp']/10**9).astimezone(timezone('US/Pacific'))
    ego_centroid = frame_sample['ego_centroid']
    ego_yaw = frame_sample['ego_yaw']
    ego_closest_lane_id = find_closest_lane(ego_centroid, ego_yaw, agent_class=ALL_WHEELS_CLASS, intersections_only=True)
    intersection_related_lanes = lane_2_master_intersection_related_lanes[ego_closest_lane_id] if ego_closest_lane_id in lane_2_master_intersection_related_lanes else set()
    if len(intersection_related_lanes) == 0:
        return []
    lane_2_speeds = defaultdict(list)
    lane_2_completions = defaultdict(list)
    lane_2_blocked_tl_signals = defaultdict(set)
    agents_with_wheels = [(agent, ALL_WHEELS_CLASS if np.nonzero(agent[-1])[0][0] == CAR_CLASS else BIKE_CLASS) for agent in frame_sample['agents'] if np.nonzero(agent[-1])[0][0] in [CAR_CLASS, BIKE_CLASS]]
    ego_speed = frame_sample['ego_speed']
    if ego_speed is not None:
        agents_with_wheels.append(((ego_centroid, None, ego_yaw, ego_speed, None, None), ALL_WHEELS_CLASS))
        intersection_related_lanes.add(ego_closest_lane_id)
    for agent, agent_class in agents_with_wheels:
        agent_speed = np.hypot(*agent[-3])
        agent_centroid = agent[0]
        agent_yaw = agent[2]
        find_closest_result = find_closest_lane(agent_centroid, agent_yaw, agent_class,
                                                                    lane_point_2_blocked_lanes_set=lane_point_2_blocked_lanes_set,
                                                                    return_point_i=True, 
                                                                    return_blocked_tl_signals=True, 
                                                                    intersections_only=True)
        if find_closest_result is not None:
            (lane_id, lane_point_i), blocked_tl_signals = find_closest_result
            lane_len = get_lane_len(lane_id)
            if not is_end_of_maneuver_lane(lane_point_i, lane_len):       
                lane_completion = lane_point_i/(lane_len - 1)
                if lane_id in intersection_related_lanes:
                    lane_2_speeds[lane_id].append(agent_speed)
                    lane_2_completions[lane_id].append(lane_completion)
                    lane_2_blocked_tl_signals[lane_id].update(blocked_tl_signals)
    lane_results = []
    for lane_id, speeds in lane_2_speeds.items():
        lane_results.append((lane_id, np.mean(speeds), np.max(speeds), len(speeds), np.min(lane_2_completions[lane_id]), lane_2_blocked_tl_signals[lane_id]))
    
    # tl's
    tl_results = set()
    tl_faces = filter_tl_faces_by_status(frame_sample['tl_faces'], "ACTIVE")
    for tl_face_id, tl_light_id, _ in tl_faces:
        tl_color = proto_API.get_traffic_face_colour(tl_face_id)
        if tl_color != 'unknown':
            for tl_signal_idx in tl_face_id_2_tl_signal_indices[tl_face_id]:
                for tl_controlled_lane in tl_signal_idx_2_controlled_lanes[tl_signal_idx]:
                    if tl_controlled_lane in intersection_related_lanes:
                        tl_color_code = TL_GREEN_COLOR if tl_color == 'green' else TL_RED_COLOR if tl_color == 'red' else TL_YELLOW_COLOR
                        tl_results.add((tl_light_id, tl_signal_idx, tl_color_code))
    
    if len(lane_results) or len(tl_results):
        master_intersection_idx = lane_id_2_master_intersection_idx[ego_closest_lane_id]
        results = (master_intersection_idx, timestamp, ego_centroid, ego_yaw, lane_results, tl_results)
    else:
        results = []
    return results

def get_tl_signal_current(lanes_info_only, tl_faces_info,
                          speed_activation_threshold=1.5, 
                          is_close_to_lane_start_completion_threshold=0.45, 
                          lane_stopped_speed_threshold=0.1, 
                          stopped_cars_min_count=2
                         ):
    tl_signals_GO, tl_signals_STOP, tl_events = set(), set(), []
    
    if len(lanes_info_only) == 0:
        return tl_signals_GO, tl_signals_STOP, tl_events   
    
    moving_lane_current_indices, stopped_lane_current_indices = [], []
    for current_i, lane_record in enumerate(lanes_info_only):
        if lane_record[1] < speed_activation_threshold:
            stopped_lane_current_indices.append(current_i)
        else:
            moving_lane_current_indices.append(current_i)
            
    # a car not moving doesn't sufficiently mean STOP for a line, but it's a suspect
    tl_signals_STOP_suspects = set()
    signal_2_stopped_car_speed_count = dict()
    signal_2_stopped_car_lanes = defaultdict(list)
    tl_stop_candidate_2_some_responsible_lane = dict()
    for current_i in stopped_lane_current_indices:
        lane_id, _, speed_max, car_counts, _, _ = lanes_info_only[current_i]
        if lane_id in controlled_lane_id_2_tl_signal_idx:
            controlling_tl_signal = controlled_lane_id_2_tl_signal_idx[lane_id]
            prev_speed, prev_count = signal_2_stopped_car_speed_count.get(controlling_tl_signal, (0, 0))
            signal_2_stopped_car_lanes[controlling_tl_signal].append(lane_id)
            signal_2_stopped_car_speed_count[controlling_tl_signal] = (max(prev_speed, speed_max), prev_count + car_counts)
            if (speed_max < lane_stopped_speed_threshold and 
                car_counts >= stopped_cars_min_count):
                    tl_signals_STOP_suspects.add(controlling_tl_signal)
                    tl_stop_candidate_2_some_responsible_lane[controlling_tl_signal] = lane_id
    # for the case of too short consec lanes
    for controlling_tl_signal, speed_count in signal_2_stopped_car_speed_count.items():
        speed_max, car_counts = speed_count
        if (speed_max < lane_stopped_speed_threshold and 
                car_counts >= stopped_cars_min_count):
                    tl_signals_STOP_suspects.add(controlling_tl_signal)
                    tl_stop_candidate_2_some_responsible_lane[controlling_tl_signal] = '_'.join(sorted(signal_2_stopped_car_lanes[controlling_tl_signal]))
            
    for current_i in moving_lane_current_indices:
        # tl exit lanes with moving cars close to start suggest status GO for corresponding tl
        lane_id, _, speed_max, _, min_lane_completion, blocked_tl_signals = lanes_info_only[current_i]
        if (min_lane_completion < is_close_to_lane_start_completion_threshold and
            min_lane_completion > 0 and
            lane_id in exit_lane_id_2_tl_signal_idx):
            tl_signal_idx = exit_lane_id_2_tl_signal_idx[lane_id]
            tl_signals_GO.add(tl_signal_idx)
            tl_events.append((lane_id, tl_signal_idx, TL_GREEN_COLOR))
            tl_signals_STOP.update(blocked_tl_signals)
            for tl_signal_idx in blocked_tl_signals:
                tl_events.append((f'{lane_id}_block', tl_signal_idx, TL_RED_COLOR))
        if (lane_id in controlled_lane_id_2_tl_signal_idx and
            speed_max > lane_stopped_speed_threshold and
            controlled_lane_id_2_tl_signal_idx[lane_id] in tl_signals_STOP_suspects
           ):
            tl_signals_STOP_suspects.remove(controlled_lane_id_2_tl_signal_idx[lane_id])
        

    remaining_suspects = tl_signals_STOP_suspects.difference(tl_signals_GO)
    tl_signals_STOP.update(remaining_suspects)
    for tl_signal_idx in remaining_suspects:
        tl_events.append((tl_stop_candidate_2_some_responsible_lane[tl_signal_idx], tl_signal_idx, TL_RED_COLOR))
        
    # processing observable tl_faces
    for tl_light_id, tl_signal_id, color_code in tl_faces_info:
        if color_code == TL_RED_COLOR:
            tl_signals_STOP.add(tl_signal_id)
        elif color_code == TL_GREEN_COLOR:
            tl_signals_GO.add(tl_signal_id)
        tl_events.append((tl_light_id, tl_signal_id, color_code))
    return tl_signals_GO.difference(tl_signals_STOP), tl_signals_STOP, set(tl_events)
            
        
def get_accumulated_tl_signals(timestamp, ego_centroid, master_intersection_idx,
                               tl_signals_GO, tl_signals_STOP, 
                               timestamp_prev, ego_centroid_prev, master_intersection_idx_prev,
                               tl_signals_buffer,
                               timediff_max_sec=2, distdiff_max_m=2, max_buffer_age_sec=15
                              ):
    timediff_sec = (timestamp - timestamp_prev).total_seconds()
    if master_intersection_idx != master_intersection_idx_prev:
        tl_signals_buffer = dict()
    elif timediff_sec > timediff_max_sec:
        tl_signals_buffer = dict()
    else:
        translation_m = np.hypot(ego_centroid_prev[0] - ego_centroid[0], ego_centroid_prev[1] - ego_centroid[1])
        if translation_m > distdiff_max_m:
            tl_signals_buffer = dict()
    for tl_signal_green_idx in tl_signals_GO:
        tl_signals_buffer[tl_signal_green_idx] = (timestamp, 1)
    for tl_signal_red_idx in tl_signals_STOP:
        tl_signals_buffer[tl_signal_red_idx] = (timestamp, 0)
        
    current_buffer_items = list(tl_signals_buffer.items())
    for tl_sig_idx, (timestamp_record, color_cod) in current_buffer_items:
        if (timestamp - timestamp_record).total_seconds() > max_buffer_age_sec:
            del tl_signals_buffer[tl_sig_idx]
    return tl_signals_buffer

def tl_seq_collate_fn(frames_batch, timestamp_min, timestamp_max):
    batch_result = []
    for frame in frames_batch:
        timestamp = datetime.fromtimestamp(frame['timestamp'] / 10 ** 9).astimezone(timezone('US/Pacific'))
        if timestamp_min < timestamp <= timestamp_max:
            info_related_lanes = get_info_per_related_lanes(frame)
            if len(info_related_lanes):
                master_intersection_idx, timestamp, ego_centroid, ego_yaw, lanes_info, tl_faces_info = info_related_lanes
                tl_signals_GO, tl_signals_STOP, observed_events = get_tl_signal_current(lanes_info, tl_faces_info)
                batch_result.append((master_intersection_idx, timestamp, ego_centroid, observed_events, tl_signals_GO, tl_signals_STOP))
    return np.array(batch_result)


def get_tl_events_df(dataloader_frames):
    master_intersection_idx_list, timestamp_list, ego_centroid_list, observed_events_list, tl_signals_GO_list, tl_signals_STOP_list = [[] for _ in range(6)]
    for batch in tqdm(dataloader_frames, desc='Tl events...'):
        for record in batch:
            master_intersection_idx, timestamp, ego_centroid, observed_events, tl_signals_GO, tl_signals_STOP = record
            master_intersection_idx_list.append(master_intersection_idx)
            timestamp_list.append(timestamp)
            ego_centroid_list.append(ego_centroid)
            observed_events_list.append(observed_events)
            tl_signals_GO_list.append(tl_signals_GO)
            tl_signals_STOP_list.append(tl_signals_STOP)

    tl_events_df = pd.DataFrame({'master_intersection_idx': master_intersection_idx_list,
                                 'timestamp': timestamp_list,
                                 'ego_centroid': ego_centroid_list,
                                 'observed_events': observed_events_list,
                                 'tl_signals_GO': tl_signals_GO_list,
                                 'tl_signals_STOP': tl_signals_STOP_list})
    tl_events_df.sort_values(by=['master_intersection_idx', 'timestamp'], inplace=True)
    return tl_events_df


def compute_tl_signal_classes(tl_events_df):
    tl_signals_buffer = dict()
    timestamp_prev, ego_centroid_prev = datetime(1970, 1, 1).astimezone(timezone('US/Pacific')), np.array([-9999, -9999])
    master_intersection_idx_prev = -9999
    tl_events_df['tl_signal_classes'] = [dict() for _ in range(len(tl_events_df))]
    for row_index, row in tqdm(tl_events_df.iterrows(), total=len(tl_events_df), desc="Cumul tl's.."):

        tl_signals_buffer = get_accumulated_tl_signals(row['timestamp'],
                                                         row['ego_centroid'],
                                                         row['master_intersection_idx'],
                                                         row['tl_signals_GO'],
                                                         row['tl_signals_STOP'],
                                                         timestamp_prev,
                                                       ego_centroid_prev,
                                                       master_intersection_idx_prev,
                                                       tl_signals_buffer)
        timestamp_prev = row['timestamp']
        ego_centroid_prev = row['ego_centroid']
        master_intersection_idx_prev = row['master_intersection_idx']
        tl_events_df.loc[row_index]['tl_signal_classes'].update({key:val[1] for key, val in tl_signals_buffer.items()})


def compute_time_to_tl_change(tl_events_df, ego_translation_m_max=1.5, time_to_event_ub=5.01, verbose=False):
    active_next_tl_signals = dict() # tl_sig_idx -> color_code
    relevant_tl_signal_change_event = dict() # tl_sig_idx -> (event_timestamp, new_event_color_code)
    timestamp_next, ego_centroid_next = datetime(2050, 1, 1).astimezone(timezone('US/Pacific')), np.array([-9999, -9999])
    master_intersection_idx_next = -9999
    tl_events_df['time_to_tl_change'] = [dict() for _ in range(len(tl_events_df))]
    timediff_max_sec = 1
    the_same_color_duration = dict()

    for row_i in tqdm(range(len(tl_events_df) - 1, -1, -1), desc='Time to color change..'):
        row = tl_events_df.iloc[row_i]

        # check that we have the same ego sdv, the same intersection, and consec. frame
        if row['master_intersection_idx'] != master_intersection_idx_next:
            active_next_tl_signals = dict()
            relevant_tl_signal_change_event = dict()
            for tl_signal_idx, color_code in row['tl_signal_classes'].items():
                relevant_tl_signal_change_event[tl_signal_idx] = (row['timestamp'], color_code)
                active_next_tl_signals[tl_signal_idx] = color_code
            the_same_color_duration = dict()
        elif (timestamp_next - row['timestamp']).total_seconds() > timediff_max_sec:
            active_next_tl_signals = dict()
            relevant_tl_signal_change_event = dict()
            for tl_signal_idx, color_code in row['tl_signal_classes'].items():
                relevant_tl_signal_change_event[tl_signal_idx] = (row['timestamp'], color_code)
                active_next_tl_signals[tl_signal_idx] = color_code
            the_same_color_duration = dict()
        else:
            translation_m = np.hypot(ego_centroid_next[0] - row['ego_centroid'][0], ego_centroid_next[1] - row['ego_centroid'][1])
            if translation_m > ego_translation_m_max:
                active_next_tl_signals = dict()
                relevant_tl_signal_change_event = dict()
                # going backward in time, row records would be future for the prev rows
                for tl_signal_idx, color_code in row['tl_signal_classes'].items():
                    relevant_tl_signal_change_event[tl_signal_idx] = (row['timestamp'], color_code)
                    active_next_tl_signals[tl_signal_idx] = color_code
                the_same_color_duration = dict()


        # if tl signal class (color) is currently unknown, then there's no reliable way to set the corresponding change event
        relevant_tl_signal_change_event_items = list(relevant_tl_signal_change_event.items())
        for tl_signal_idx, _ in relevant_tl_signal_change_event_items:
            if tl_signal_idx not in row['tl_signal_classes']:
                del relevant_tl_signal_change_event[tl_signal_idx]
                if verbose:
                    print(f'removing {tl_signal_idx} as currently non-present')

        time_to_tl_change_current = dict()
        relevant_tl_signal_change_event_items = list(relevant_tl_signal_change_event.items())
        for tl_signal_idx, (event_timestamp, event_color_code) in relevant_tl_signal_change_event_items:
            # if the current tl color is the same, as in the future change event, ...
            if event_color_code != row['tl_signal_classes'][tl_signal_idx]:
                time_to_tl_change_current[tl_signal_idx] = np.clip((event_timestamp - row['timestamp']).total_seconds(), 0, time_to_event_ub)
            else:
                # check color continuity
                if active_next_tl_signals[tl_signal_idx] == row['tl_signal_classes'][tl_signal_idx]:
                    relevant_tl_signal_change_event[tl_signal_idx] = (row['timestamp'], event_color_code)
                    if tl_signal_idx not in the_same_color_duration:
                        the_same_color_duration[tl_signal_idx] = row['timestamp']
                    elif (the_same_color_duration[tl_signal_idx] - row['timestamp']).total_seconds() >= time_to_event_ub:
                        time_to_tl_change_current[tl_signal_idx] = time_to_event_ub
                        if verbose:
                            print(f'too long the same color: TTE at least 5.01')
                else:
                    # the event stored in relevant_tl_signal_change_event is not the latest change event, cause the latest change happens in the following timestamp
                    relevant_tl_signal_change_event[tl_signal_idx] = (timestamp_next, active_next_tl_signals[tl_signal_idx])
                    time_to_tl_change_current[tl_signal_idx] = np.clip((timestamp_next - row['timestamp']).total_seconds(), 0, time_to_event_ub)
                    
        active_next_tl_signals = row['tl_signal_classes'].copy() 
        for tl_signal_idx, color_code in row['tl_signal_classes'].items():
            if tl_signal_idx not in relevant_tl_signal_change_event:
                relevant_tl_signal_change_event[tl_signal_idx] = (row['timestamp'], color_code)
        
        timestamp_next, ego_centroid_next, master_intersection_idx_next = row['timestamp'], row['ego_centroid'], row['master_intersection_idx']
        tl_events_df.iloc[row_i]['time_to_tl_change'].update(time_to_tl_change_current)


def get_rnn_inputs_from_events(obsereved_events):
    obsereved_events = [x for x in obsereved_events if '_block' not in x[0]]
    tl_events, lane_events = set(), set()
    for source_id, tl_signal_idx, color in obsereved_events:
        if source_id in traffic_light_ids_all:
            tl_events.add((source_id, (1, 0, 0, 0) if color == TL_GREEN_COLOR else (0, 1, 0, 0)))
        else:
            if color == TL_RED_COLOR:
                lane_events.add((source_id, (0, 0, 1, 0)))
            else:
                lane_events.add((source_id, (0, 0, 0, 1)))
    return list(tl_events), list(lane_events)


def compute_rnn_inputs(tl_events_df):
    tl_events_df['rnn_inputs_raw'] = [[] for _ in range(len(tl_events_df))]
    for row_idx, row in tl_events_df.iterrows():
        tl_inputs, lane_inputs = get_rnn_inputs_from_events(row['observed_events'])
        tl_events_df.loc[row_idx]['rnn_inputs_raw'].extend(tl_inputs + lane_inputs)
