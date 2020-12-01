import bisect
import gc
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import zarr
from l5kit.data import (
    ChunkedDataset,
    get_agents_slice_from_frames,
    get_frames_slice_from_scenes,
    filter_tl_faces_by_frames,
    get_tl_faces_slice_from_frames,
    filter_tl_faces_by_status,
)
from l5kit.data.zarr_dataset import (
    FRAME_ARRAY_KEY,
    AGENT_ARRAY_KEY,
    SCENE_ARRAY_KEY,
    TL_FACE_ARRAY_KEY,
)
from l5kit.dataset.select_agents import (
    TH_DISTANCE_AV,
    TH_EXTENT_RATIO,
    TH_YAW_DEGREE,
    select_agents,
)
from l5kit.geometry import rotation33_as_yaw
from pytz import timezone
from torch.utils.data import Dataset
from zarr import convenience

# WARNING: changing these values impact the number of instances selected for both train and inference!
MIN_FRAME_HISTORY = (
    10  # minimum number of frames an agents must have in the past to be picked
)
MIN_FRAME_FUTURE = (
    1  # minimum number of frames an agents must have in the future to be picked
)
MASK_AGENT_INDICES_ARRAY_KEY = "mask_agent_indices"


def generate_frame_sample_without_hist(
    state_index: int,
    frames: zarr.core.Array,
    tl_faces: zarr.core.Array,
    agents: zarr.core.Array,
    agents_from_standard_mask_only: bool = False,
    mask_agent_indices: zarr.core.Array = None,
) -> dict:
    frame = frames[state_index]
    if not agents_from_standard_mask_only:
        agent_slice = get_agents_slice_from_frames(frame)
        agents = agents[agent_slice].copy()
    else:
        masked_indices_slice = slice(*frame["mask_agent_index_interval"])
        masked_agent_indices = [
            el[0] for el in mask_agent_indices[masked_indices_slice]
        ]
        if masked_agent_indices:
            agents = agents.get_coordinate_selection(masked_agent_indices).copy()
        else:
            agents = []

    ego_centroid = frame["ego_translation"][:2]
    # try to estimate ego velocity
    if state_index > 0:
        prev_frame_candidate = frames[state_index - 1]
        prev_ego_centroid = prev_frame_candidate["ego_translation"][:2]
        translation_m = np.hypot(
            prev_ego_centroid[0] - ego_centroid[0],
            prev_ego_centroid[1] - ego_centroid[1],
        )
        if translation_m < 10:
            timestamp = datetime.fromtimestamp(frame["timestamp"] / 10 ** 9).astimezone(
                timezone("US/Pacific")
            )
            timestamp_prev = datetime.fromtimestamp(
                prev_frame_candidate["timestamp"] / 10 ** 9
            ).astimezone(timezone("US/Pacific"))
            timediff_sec = (timestamp - timestamp_prev).total_seconds()
            if timestamp > timestamp_prev and timediff_sec < 0.2:
                ego_speed = (ego_centroid - prev_ego_centroid) / timediff_sec
            else:
                ego_speed = None
        else:
            ego_speed = None
    else:
        ego_speed = None

    try:
        tl_slice = get_tl_faces_slice_from_frames(frame)  # -1 is the farthest
        frame["traffic_light_faces_index_interval"] -= tl_slice.start
        tl_faces_this = filter_tl_faces_by_frames([frame], tl_faces[tl_slice].copy())[0]
        tl_faces_this = filter_tl_faces_by_status(tl_faces_this, "ACTIVE")
    except ValueError:
        tl_faces_this = []
    return {
        "ego_centroid": ego_centroid,
        "ego_speed": ego_speed,
        "ego_yaw": rotation33_as_yaw(frame["ego_rotation"]),
        "tl_faces": tl_faces_this,
        "agents": agents,
    }


def get_agent_indices_set(
    dataset: ChunkedDataset,
    filter_agents_threshold: float,
    min_frame_histories: List,
    min_frame_future: int,
):
    agents_mask_path = Path(dataset.path) / f"agents_mask/{filter_agents_threshold}"
    if not agents_mask_path.exists():  # don't check in root but check for the path
        print(
            f"cannot find the right config in {dataset.path},\n"
            f"your cfg has loaded filter_agents_threshold={filter_agents_threshold};\n"
            "but that value doesn't have a match among the agents_mask in the zarr\n"
            "Mask will now be generated for that parameter."
        )
        select_agents(
            dataset,
            filter_agents_threshold,
            th_yaw_degree=TH_YAW_DEGREE,
            th_extent_ratio=TH_EXTENT_RATIO,
            th_distance_av=TH_DISTANCE_AV,
        )

    agents_mask = convenience.load(
        str(agents_mask_path)
    )  # note (lberg): this doesn't update root

    min_frame_history_vals = sorted(min_frame_histories)
    orig_indices_order = sorted(
        range(len(min_frame_histories)), key=lambda i: min_frame_histories[i]
    )
    results = []
    result_mask = agents_mask[:, 1] >= min_frame_future
    past_counts = agents_mask[:, 0]
    del agents_mask
    gc.collect()

    for min_frame_history_val in min_frame_history_vals:
        result_mask[past_counts < min_frame_history_val] = False
        if len(results) == 0:
            agents_indices = np.nonzero(result_mask)[0]
            results.append(set(agents_indices))
            del agents_indices
            gc.collect()
        else:
            agents_indices_removed = {
                idx for idx in results[0] if result_mask[idx] == 0
            }
            results.append(agents_indices_removed)

    if results:
        results = [results[i] for i in orig_indices_order]
    return results


class FramesDataset(Dataset):
    def __init__(
        self,
        zarr_dataset_path: str,
        cache_zarr: bool = False,
        with_history: bool = False,
        return_indices: bool = False,
        agents_from_standard_mask_only: bool = False,
    ):

        if cache_zarr:
            zarr_root = zarr.open_group(
                store=zarr.LRUStoreCache(
                    zarr.DirectoryStore(zarr_dataset_path), max_size=int(1e9)
                ),
                mode="r",
            )
        else:
            zarr_root = zarr.open_group(zarr_dataset_path, mode="r")

        self.cumulative_sizes = zarr_root[SCENE_ARRAY_KEY]["frame_index_interval"][:, 1]

        if with_history:
            raise NotImplementedError
        if agents_from_standard_mask_only:
            self.sample_function = partial(
                generate_frame_sample_without_hist,
                agents=zarr_root[AGENT_ARRAY_KEY],
                tl_faces=zarr_root[TL_FACE_ARRAY_KEY],
                agents_from_standard_mask_only=True,
                mask_agent_indices=zarr_root[MASK_AGENT_INDICES_ARRAY_KEY],
            )
        else:
            self.sample_function = partial(
                generate_frame_sample_without_hist,
                agents=zarr_root[AGENT_ARRAY_KEY],
                tl_faces=zarr_root[TL_FACE_ARRAY_KEY],
            )
        self.with_history = with_history
        self.return_indices = return_indices
        self.zarr_root = zarr_root

    def __len__(self) -> int:
        return len(self.zarr_root[FRAME_ARRAY_KEY])

    def get_frame(self, scene_index: int, state_index: int) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame
        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image, the target trajectory (position and yaw) along with their availability,
            the 2D matrix to center that agent, the agent track (-1 if ego) and the timestamp
        """
        frames_slice = get_frames_slice_from_scenes(
            self.zarr_root[SCENE_ARRAY_KEY][scene_index]
        )
        frames = self.zarr_root[FRAME_ARRAY_KEY][frames_slice]
        timestamp = frames[state_index]["timestamp"]
        data = self.sample_function(state_index, frames)

        results = {
            "timestamp": timestamp,
        }
        if self.with_history:
            results.update(
                {
                    "ego_centroid": data["ego_centroid"],
                    "ego_speed": data["ego_speed"],
                    "history_tl_faces": data["history_tl_faces"],
                    "history_agents": data["history_agents"],
                }
            )
        else:
            results.update(
                (
                    {
                        "ego_centroid": data["ego_centroid"],
                        "ego_speed": data["ego_speed"],
                        "ego_yaw": data["ego_yaw"],
                        "tl_faces": data["tl_faces"],
                        "agents": data["agents"],
                    }
                )
            )
        if self.return_indices:
            results.update({"scene_index": scene_index, "state_index": state_index})

        return results

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            if -index > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            index = len(self) + index

        scene_index = bisect.bisect_right(self.cumulative_sizes, index)

        if scene_index == 0:
            state_index = index
        else:
            state_index = index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index)

    def get_scene_indices(self, scene_idx: int) -> np.ndarray:
        scenes = self.zarr_root[SCENE_ARRAY_KEY]
        assert scene_idx < len(
            scenes
        ), f"scene_idx {scene_idx} is over len {len(scenes)}"
        return np.arange(*scenes[scene_idx]["frame_index_interval"])
