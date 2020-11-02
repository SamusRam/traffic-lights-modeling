from functools import lru_cache
from typing import Iterator, Sequence, Union, no_type_check

import numpy as np
import pymap3d as pm
from l5kit.data.proto.road_network_pb2 import GeoFrame, GlobalId, MapElement, MapFragment, RoadNetworkSegment
from l5kit.geometry import transform_points

CACHE_SIZE = int(1e5)
ENCODING = "utf-8"


class MapAPI:
    def __init__(self, protobuf_map_path: str, world_to_ecef: np.ndarray):
        """
        Interface to the raw protobuf map file with the following features:
        - access to element using ID is O(1);
        - access to coordinates in world ref system for a set of elements is O(1) after first access (lru cache)
        - object support iteration using __getitem__ protocol

        Args:
            protobuf_map_path (str): path to the protobuf file
            world_to_ecef (np.ndarray): transformation matrix from world coordinates to ECEF (dataset dependent)
        """
        self.protobuf_map_path = protobuf_map_path
        self.ecef_to_world = np.linalg.inv(world_to_ecef)

        with open(protobuf_map_path, "rb") as infile:
            mf = MapFragment()
            mf.ParseFromString(infile.read())

        self.elements = mf.elements
        self.ids_to_el = {self.id_as_str(el.id): idx for idx, el in enumerate(self.elements)}  # store a look-up table

        self.lane_orientation_code_2_str = {0: "UNKNOWN_TRAVEL_DIRECTION",
                                            1: "TWO_WAY",
                                            2: "ONE_WAY_FORWARD",
                                            3: "ONE_WAY_BACKWARD",
                                            4: "ONE_WAY_REVERSIBLE"}

    @staticmethod
    @no_type_check
    def id_as_str(element_id: GlobalId) -> str:
        """
        Get the element id as a string.
        Elements ids are stored as a variable len sequence of bytes in the protobuf

        Args:
            element_id (GlobalId): the GlobalId in the protobuf

        Returns:
            str: the id as a str
        """
        return element_id.id.decode(ENCODING)

    @staticmethod
    def _undo_e7(value: float) -> float:
        """
        Latitude and longitude are stored as value*1e7 in the protobuf for efficiency and guaranteed accuracy.
        Convert them back to float.

        Args:
            value (float): the scaled value

        Returns:
            float: the unscaled value
        """
        return value / 1e7

    @no_type_check
    def unpack_deltas_cm(self, dx: Sequence[int], dy: Sequence[int], dz: Sequence[int], frame: GeoFrame) -> np.ndarray:
        """
        Get coords in world reference system (local ENU->ECEF->world).
        See the protobuf annotations for additional information about how coordinates are stored

        Args:
            dx (Sequence[int]): X displacement in centimeters in local ENU
            dy (Sequence[int]): Y displacement in centimeters in local ENU
            dz (Sequence[int]): Z displacement in centimeters in local ENU
            frame (GeoFrame): geo-location information for the local ENU. It contains lat and long origin of the frame

        Returns:
            np.ndarray: array of shape (Nx3) with XYZ coordinates in world ref system

        """
        x = np.cumsum(np.asarray(dx) / 100)
        y = np.cumsum(np.asarray(dy) / 100)
        z = np.cumsum(np.asarray(dz) / 100)
        frame_lat, frame_lng = self._undo_e7(frame.origin.lat_e7), self._undo_e7(frame.origin.lng_e7)
        xyz = np.stack(pm.enu2ecef(x, y, z, frame_lat, frame_lng, 0), axis=-1)
        xyz = transform_points(xyz, self.ecef_to_world)
        return xyz

    @staticmethod
    @no_type_check
    def is_lane(element: MapElement) -> bool:
        """
        Check whether an element is a valid lane

        Args:
            element (MapElement): a proto element

        Returns:
            bool: True if the element is a valid lane
        """
        return bool(element.element.HasField("lane"))

    @staticmethod
    @no_type_check
    def is_road_network_segment(element: MapElement) -> bool:
        """
        Check whether an element is a valid RoadNetworkSegment

        Args:
            element (MapElement): a proto element

        Returns:
            bool: True if the element is a valid segment
        """
        return bool(element.element.HasField("segment"))

    @staticmethod
    @no_type_check
    def is_junction(element: MapElement) -> bool:
        """
        Check whether an element is a valid Junction

        Args:
            element (MapElement): a proto element

        Returns:
            bool: True if the element is a valid Junction
        """
        return bool(element.element.HasField("junction"))

    def is_traffic_control_element(self, element_id: str) -> bool:
        """
        Args:
            element (MapElement): a proto element

        Returns:
        """
        element = self[element_id]
        return bool(element.element.HasField("traffic_control_element"))

    def is_primary_road(self, element_id: str) -> bool:
        """
        Check whether a RoadNetworkSegment is primary road

        Args:
            element_id (str): the id (utf-8 encode) of the element

        Returns:
            bool: True if the element is a primary road, False otherwise
        """
        element = self[element_id]
        if not element.element.HasField("segment"):
            return False
        segment = element.element.segment
        return bool(segment.road_class == RoadNetworkSegment.RoadClass.PRIMARY)

    def is_secondary_road(self, element_id: str) -> bool:
        """
        Check whether a RoadNetworkSegment is primary road

        Args:
            element_id (str): the id (utf-8 encode) of the element

        Returns:
            bool: True if the element is a primary road, False otherwise
        """
        element = self[element_id]
        if not element.element.HasField("segment"):
            return False
        segment = element.element.segment
        return bool(segment.road_class == RoadNetworkSegment.RoadClass.SECONDARY)

    @lru_cache(maxsize=CACHE_SIZE)
    def get_lane_coords(self, element_id: str) -> dict:
        """
        Get XYZ coordinates in world ref system for a lane given its id
        lru_cached for O(1) access

        Args:
            element_id (str): lane element id

        Returns:
            dict: a dict with the two boundaries coordinates as (Nx3) XYZ arrays
        """
        element = self[element_id]
        assert self.is_lane(element)

        lane = element.element.lane
        left_boundary = lane.left_boundary
        right_boundary = lane.right_boundary

        xyz_left = self.unpack_deltas_cm(
            left_boundary.vertex_deltas_x_cm,
            left_boundary.vertex_deltas_y_cm,
            left_boundary.vertex_deltas_z_cm,
            lane.geo_frame,
        )
        xyz_right = self.unpack_deltas_cm(
            right_boundary.vertex_deltas_x_cm,
            right_boundary.vertex_deltas_y_cm,
            right_boundary.vertex_deltas_z_cm,
            lane.geo_frame,
        )

        return {"xyz_left": xyz_left, "xyz_right": xyz_right}

    def get_lanes_ahead(self, element_id: str) -> list:
        """
        Get lanes that the follow the current one in the natural travel direction,
         i.e. without changing lanes

        Args:
            element_id (str): lane element id

        Returns:
            list: a list with element ids of consequent lanes
        """
        element = self[element_id]
        assert self.is_lane(element)

        lane = element.element.lane
        return [self.id_as_str(x) for x in lane.lanes_ahead]

    def get_traffic_control_elements_at_junction(self, element_id: str) -> list:
        """

        Args:
            element_id (str): junction element id

        Returns:
            list: a list with element ids of all traffic control elements regulating traffic in the intersection
        """
        element = self[element_id]
        assert self.is_junction(element)

        junction = element.element.junction
        return [self.id_as_str(x) for x in junction.traffic_control_elements]

    def get_road_network_nodes_at_junction(self, element_id: str) -> list:
        """

        Args:
            element_id (str): junction element id

        """
        element = self[element_id]
        assert self.is_junction(element)

        junction = element.element.junction
        return [self.id_as_str(x) for x in junction.road_network_nodes]

    def get_road_segments_at_road_network_node(self, node_element_id: str):
        element = self[node_element_id]
        road_network_node = element.element.node
        return [self.id_as_str(x) for x in road_network_node.road_segments]

    def get_lanes_at_road_segment(self, segment_element_id: str):
        element = self[segment_element_id]
        road_segment = element.element.segment
        return [self.id_as_str(x) for x in road_segment.lanes]

    def get_all_lanes_at_junction(self, junction_element_id: str) -> list:
        """
        Args:
            junction_element_id (str): junction element id

        Returns:
            list: a list with element ids of all related lanes
        """
        element = self[junction_element_id]
        assert self.is_junction(element)

        junction = element.element.junction
        direct_lanes = [self.id_as_str(x) for x in junction.lanes]
        road_network_nodes = self.get_road_network_nodes_at_junction(junction_element_id)
        all_road_segments = [x
                             for node in road_network_nodes
                             for x in self.get_road_segments_at_road_network_node(node)]
        all_lanes_assigned_to_segments = [x
                                          for segment_id in all_road_segments
                                          for x in self.get_lanes_at_road_segment(segment_id)]
        return direct_lanes + all_lanes_assigned_to_segments

    def get_lane_to_left(self, element_id: str) -> str:
        """
        // If any, the lanes a car would get to by executing a lane change maneuver.
    // The lane can be assumed to be physically adjacent to the current lane.
    // A 0 ID (the default when the field is not set) indicates the maneuver in that direction
    // is not allowed.

        Args:
            element_id (str): lane element id

        Returns:
        """
        element = self[element_id]
        assert self.is_lane(element)

        lane = element.element.lane
        return self.id_as_str(lane.adjacent_lane_change_left) if lane.HasField(
            'adjacent_lane_change_left') and self.id_as_str(lane.adjacent_lane_change_left) != '0' else ''

    def get_lane_to_right(self, element_id: str) -> str:
        """
        // If any, the lanes a car would get to by executing a lane change maneuver.
    // The lane can be assumed to be physically adjacent to the current lane.
    // A 0 ID (the default when the field is not set) indicates the maneuver in that direction
    // is not allowed.

        Args:
            element_id (str): lane element id

        Returns:
        """
        element = self[element_id]
        assert self.is_lane(element)

        lane = element.element.lane
        return self.id_as_str(lane.adjacent_lane_change_right) if lane.HasField(
            'adjacent_lane_change_right') and self.id_as_str(lane.adjacent_lane_change_right) != '0' else ''

    def get_lanes_to_yield(self, element_id: str) -> list:
        """
        // Set of lanes to support ceding right of way: the ego car is expected to yield to any
    // vehicles close enough in one of these lanes. If the lane is in a traffic-light-controlled
    // intersection, the yield set here only applies when the traffic light is not functional.
    // If it is functional, the traffic light state and yield sets associated with the light faces
    // override this.

        Args:
            element_id (str): lane element id

        Returns:
        """
        element = self[element_id]
        assert self.is_lane(element)

        lane = element.element.lane
        return list(lane.yield_to_lanes)

    def get_lane_traffic_controls(self, element_id: str) -> list:
        """
        // Traffic signals, e.g. traffic lights or individual "faces" of traffic lights, stop signs,
    // yield signs, etc. controlling the exit from the lane onto one of the lanes ahead.
    // It is assumed that conceptual line to cross according to the signal is the exit boundary of
    // the lane, i.e. the line formed by the last vertex of the left boundary and the last vertex
    // of the right boundary.

        Args:
            element_id (str): lane element id

        Returns:
        """
        element = self[element_id]
        assert self.is_lane(element)

        lane = element.element.lane
        return list(lane.traffic_controls)

    def get_traffic_control_lanes_under_control(self, element_id: str) -> list:
        """
        // The lanes that the signal controls. Each sequence starts with the lane in which the car
    // needs to observe the signal, followed by the lane sequence to which it can proceed according
    // to the signal.

        Args:
            element_id (str): traffic control element id

        Returns:
        """
        element = self[element_id]

        traffic_control = element.element.traffic_control_element
        return list(traffic_control.controlled_paths)

    def get_lane_orientation(self, element_id: str) -> str:
        """
        Get lanes that the follow the current one in the natural travel direction,
         i.e. without changing lanes

        Args:
            element_id (str): lane element id

        Returns:
            list: a list with element ids of consequent lanes
        """
        element = self[element_id]
        assert self.is_lane(element)

        lane = element.element.lane
        return self.lane_orientation_code_2_str[lane.orientation_in_parent_segment]

    @staticmethod
    @no_type_check
    def is_crosswalk(element: MapElement) -> bool:
        """
        Check whether an element is a valid crosswalk

        Args:
            element (MapElement): a proto element

        Returns:
            bool: True if the element is a valid crosswalk
        """
        if not element.element.HasField("traffic_control_element"):
            return False
        traffic_element = element.element.traffic_control_element
        return bool(traffic_element.HasField("pedestrian_crosswalk") and traffic_element.points_x_deltas_cm)

    @lru_cache(maxsize=CACHE_SIZE)
    def get_crosswalk_coords(self, element_id: str) -> dict:
        """
        Get XYZ coordinates in world ref system for a crosswalk given its id
        lru_cached for O(1) access

        Args:
            element_id (str): crosswalk element id

        Returns:
            dict: a dict with the polygon coordinates as an (Nx3) XYZ array
        """
        element = self[element_id]
        assert self.is_crosswalk(element)
        traffic_element = element.element.traffic_control_element

        xyz = self.unpack_deltas_cm(
            traffic_element.points_x_deltas_cm,
            traffic_element.points_y_deltas_cm,
            traffic_element.points_z_deltas_cm,
            traffic_element.geo_frame,
        )

        return {"xyz": xyz}

    def get_traffic_light_coords(self, element_id: str) -> dict:
        """
        Get XYZ coordinates in world ref system for a crosswalk given its id
        lru_cached for O(1) access

        Args:
            element_id (str): crosswalk element id

        Returns:
            dict: a dict with the polygon coordinates as an (Nx3) XYZ array
        """
        element = self[element_id]
        traffic_element = element.element.traffic_control_element

        xyz = self.unpack_deltas_cm(
            traffic_element.points_x_deltas_cm,
            traffic_element.points_y_deltas_cm,
            traffic_element.points_z_deltas_cm,
            traffic_element.geo_frame,
        )

        return {"xyz": xyz}

    def is_traffic_face_colour(self, element_id: str, colour: str) -> bool:
        """
        Check if the element is a traffic light face of the given colour

        Args:
            element_id (str): the id (utf-8 encode) of the element
            colour (str): the colour to check
        Returns:
            True if the element is a traffic light with the given colour
        """
        element = self[element_id]
        if not element.element.HasField("traffic_control_element"):
            return False
        traffic_el = element.element.traffic_control_element
        if (
                traffic_el.HasField(f"signal_{colour}_face")
                or traffic_el.HasField(f"signal_left_arrow_{colour}_face")
                or traffic_el.HasField(f"signal_right_arrow_{colour}_face")
                or traffic_el.HasField(f"signal_upper_left_arrow_{colour}_face")
                or traffic_el.HasField(f"signal_upper_right_arrow_{colour}_face")
                or traffic_el.HasField(f"signal_{colour}_u_turn")
        ):
            return True
        return False

    def get_traffic_face_colour(self, element_id: str):
        for color in ['green', 'yellow', 'red']:
            if self.is_traffic_face_colour(element_id, color):
                return color
        return 'unknown'

    def is_bike_only_lane(self, element_id):
        assert self.is_lane(self[element_id])
        return self[element_id].element.lane.access_restriction.type == 4

    def get_traffic_light_face_sets(self, element_id):
        """

        Args:
            element_id (str): the id (utf-8 encode) of the element
        Returns:
        """
        element = self[element_id]
        if not element.element.HasField("traffic_control_element"):
            return None
        traffic_el = element.element.traffic_control_element
        results = []
        for colour in ['red', 'green', 'yellow']:
            for face_set_name in [f"signal_{colour}_face",
                                  f"signal_left_arrow_{colour}_face",
                                  f"signal_right_arrow_{colour}_face",
                                  f"signal_upper_left_arrow_{colour}_face",
                                  f"signal_upper_right_arrow_{colour}_face",
                                  f"signal_{colour}_u_turn"]:
                if traffic_el.HasField(face_set_name):
                    results.append((getattr(traffic_el, face_set_name), face_set_name))
        return results

    def get_rules_for_traffic_light_face_set(self, traffic_light_face_set):
        """
        Args:

        Returns:
            list of tuples (lane, yield_to_lanes, yield_to_crosswalks)
            / The lane where the cars need to observe the traffic light.
            lane

            // The set of all other lanes that the lane above needs to yield to.
            yield_to_lanes

            // List of crosswalks that are not safe to ignore when this light face is on.
            // For example, on green, the intersection lane turning right needs to yield to the
            // pedestrian crosswalk for pedestrians going straight.
            yield_to_crosswalks
        """
        result = []
        for yield_rules_when_on_ in traffic_light_face_set.yield_rules_when_on:
            lane = yield_rules_when_on_.lane
            yield_to_lanes = yield_rules_when_on_.yield_to_lanes
            yield_to_crosswalks = yield_rules_when_on_.yield_to_crosswalks
            result.append({'lane': lane, 'yield_to_lanes': yield_to_lanes, 'yield_to_crosswalks': yield_to_crosswalks})
        return result

    def get_lane_parent(self, element_id: str) -> bool:
        """
        Get segment or junction, the lane belongs to

        Args:
            element_id (str): the id (utf-8 encode) of the lane element

        Returns:
            element_id of the parent
        """
        element_instance = self[element_id]
        if not element_instance.element.HasField("lane"):
            raise ValueError(f"The element {element_id} is not a lane.")
        parent_element_id = element_instance.element.lane.parent_segment_or_junction
        return parent_element_id

    @no_type_check
    def __getitem__(self, item: Union[int, str, bytes]) -> MapElement:
        if isinstance(item, str):
            return self.elements[self.ids_to_el[item]]
        elif isinstance(item, int):
            return self.elements[item]
        elif isinstance(item, bytes):
            return self.elements[self.ids_to_el[item.decode(ENCODING)]]
        else:
            raise TypeError("only str, bytes and int are allowed in API __getitem__")

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]
