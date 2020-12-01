import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from lyft_trajectories.data_preprocessing.common.map_traffic_lights_data import (
    get_lane_center_line,
    find_closest_lane,
    get_info_per_related_lanes,
    lane_id_2_master_intersection_idx,
    get_tl_signal_current,
    get_accumulated_tl_signals,
    tl_signal_idx_2_stop_coordinates,
    get_traffic_light_coordinates,
    traffic_light_ids_all,
    lane_id_2_idx,
    ALL_WHEELS_CLASS,
)
from lyft_trajectories.utils.l5kit_modified.l5kit_modified import FramesDataset
from IPython.display import display, clear_output
import PIL
import time
from datetime import datetime
from pytz import timezone
from tqdm.auto import tqdm
import imutils
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-basename", default="validate")
parser.add_argument("--intersection-i", default=0, type=int)
parser.add_argument("--vis-inputs", action="store_true")
parser.add_argument("--vis-predictions", action="store_true")
parser.add_argument("--start-scene", default=4, type=int)
parser.add_argument("--end-scene", default=6, type=int)
parser.add_argument("--scenes-per-video", default=2, type=int)

args = parser.parse_args()

cycled_colors = plt.get_cmap("tab20b")(np.linspace(0, 1, 20))
cycled_colors_targets = plt.get_cmap("Set2")(np.linspace(0, 1, 8))

intersection_i = args.intersection_i
dataset_basename = args.dataset_basename
vis_inputs = args.vis_inputs
vis_predictions = args.vis_predictions
start_scene = args.start_scene
end_scene = args.end_scene
scenes_per_video = args.scenes_per_video

dataset_path = f"input/scenes/{dataset_basename}_filtered_min_frame_history_4_min_frame_future_1_with_mask_idx.zarr"

zarr_dataset_filtered = ChunkedDataset(dataset_path)
zarr_dataset_filtered.open()

cfg = load_config_data("input/visualisation_config.yaml")
cfg["raster_params"]["map_type"] = "py_semantic"
dm = LocalDataManager()
rast = build_rasterizer(cfg, dm)
dataset_filtered = EgoDataset(cfg, zarr_dataset_filtered, rast)
frame_dataset = FramesDataset(dataset_path)


def plot_line(
    ax,
    line_id,
    speed=None,
    completion=None,
    speed_min=0,
    speed_max=12.295,
    color=None,
    text1="",
    text2="",
):
    lane_center_line = get_lane_center_line(line_id)
    if speed is None:
        ax.scatter(lane_center_line[:, 0], lane_center_line[:, 1], s=11, color=color)
        ax.text(
            lane_center_line[:, 0].mean() + 1.7,
            lane_center_line[:, 1].mean() + 4,
            text1,
            fontsize=14,
        )
        ax.text(
            lane_center_line[:, 0].mean() - 12,
            lane_center_line[:, 1].mean() - 1.7,
            text2,
            color=color,
            fontsize=12,
        )
    else:
        speed = np.clip(speed, speed_min, speed_max)
        color = plt.get_cmap("Oranges")(speed / speed_max)
        completion_idx = int((len(lane_center_line) - 1) * completion)
        ax.scatter(lane_center_line[:, 0], lane_center_line[:, 1], s=0.2, color=color)

        ax.scatter(
            lane_center_line[completion_idx, 0],
            lane_center_line[completion_idx, 1],
            s=3,
            color=color,
        )
        speed_activation_threshold = 1.5
        ax.text(
            lane_center_line[completion_idx, 0],
            lane_center_line[completion_idx, 1] + 0.7,
            f"{speed:.0f} ({line_id})",
            color=color if speed > speed_activation_threshold else "grey",
            fontsize=14,
        )
        if completion_idx + 1 < len(lane_center_line):
            ax.scatter(
                lane_center_line[completion_idx + 1, 0],
                lane_center_line[completion_idx + 1, 1],
                s=12,
                color=color,
            )


def store_output(im, output, start_scene, end_scene, intersection, show, pred=False):
    if output is None:
        start_scene = (
            f'{"".join(["0" for _ in range(3 - len(str(start_scene)))])}{start_scene}'
        )
        end_scene = (
            f'{"".join(["0" for _ in range(3 - len(str(end_scene)))])}{end_scene}'
        )
        output = cv2.VideoWriter(
            f'outputs/vis_check_{dataset_basename}_{str(intersection) + "_" if intersection is not None else ""}scenes_{start_scene}_{end_scene}_{"pred" if pred else "inputs"}.avi',
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            10,
            (im.shape[1], im.shape[0]),
        )
    output.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    if show:
        clear_output(wait=True)
        display(PIL.Image.fromarray(im))
        time.sleep(0.01)
    return output


def run_vis(
    frame_dataset,
    dataset_filtered,
    scene_idx_bounds,
    output=None,
    vis=True,
    show=False,
    scenes_per_video=1,
    intersection=None,
    tl_events_df=None,
    tl_events_df_pred=None,
):
    tl_signals_buffer = dict()
    timestamp_prev, ego_centroid_prev = (
        datetime(1970, 1, 1).astimezone(timezone("US/Pacific")),
        np.array([-9999, -9999]),
    )
    master_intersection_idx_prev = -9999

    for start_scene in tqdm(
        range(*scene_idx_bounds, scenes_per_video),
        desc="Iterating over scenes for visualization check...",
    ):
        end_scene = start_scene + scenes_per_video
        for scene_idx in range(start_scene, end_scene):
            indexes = frame_dataset.get_scene_indices(scene_idx)

            seq_order_i = 0
            change_source_id_2_idx = dict()
            scene_start_frame_idx = np.min(indexes)
            for idx in indexes:

                centroid = frame_dataset[idx]["ego_centroid"]
                yaw = frame_dataset[idx]["ego_yaw"]

                closest_lane_id = find_closest_lane(
                    centroid, yaw, ALL_WHEELS_CLASS, intersections_only=True
                )
                if closest_lane_id is None or (
                    intersection is not None
                    and lane_id_2_master_intersection_idx[closest_lane_id]
                    != intersection
                ):
                    continue

                info_related_lanes = get_info_per_related_lanes(frame_dataset[idx])

                if len(info_related_lanes):

                    (
                        master_intersection_idx,
                        timestamp,
                        ego_centroid,
                    ) = info_related_lanes[:3]
                    lane_speeds_completions, tl_faces_info = info_related_lanes[-2:]
                    (
                        tl_signals_GO,
                        tl_signals_STOP,
                        observed_events,
                    ) = get_tl_signal_current(lane_speeds_completions, tl_faces_info)
                    tl_signals_buffer = get_accumulated_tl_signals(
                        timestamp,
                        ego_centroid,
                        master_intersection_idx,
                        tl_signals_GO,
                        tl_signals_STOP,
                        timestamp_prev,
                        ego_centroid_prev,
                        master_intersection_idx_prev,
                        tl_signals_buffer,
                    )
                    master_intersection_idx_prev = master_intersection_idx
                    timestamp_prev, ego_centroid_prev = timestamp, ego_centroid

                    # plotting
                    # to np source: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
                    if vis:
                        data_filtered = dataset_filtered[idx]
                        im_filtered = data_filtered["image"].transpose(1, 2, 0)
                        im_filtered = dataset_filtered.rasterizer.to_rgb(im_filtered)
                        im_filtered = im_filtered[::-1, :, :]

                        fig = plt.figure(figsize=(3, 3))
                        ax = fig.add_subplot(111)
                        plt.axis("off")
                        if (
                            tl_events_df_pred is not None
                            and (scene_idx, idx - scene_start_frame_idx)
                            in tl_events_df_pred.index
                        ):

                            tl_events_pred_current = tl_events_df_pred.loc[
                                [(scene_idx, idx - scene_start_frame_idx)]
                            ]
                            tl_indices = [
                                int(col_name.replace("_green_prob", ""))
                                for col_name in tl_events_df_pred.columns
                                if "_green_prob" in col_name
                            ]

                            tl_sig_idx_2_min_x, tl_sig_idx_2_min_y = dict(), dict()
                            for tl_signal_order_num, tl_signal_idx in enumerate(
                                tl_indices
                            ):
                                green_prob_val = tl_events_pred_current[
                                    f"{tl_signal_idx}_green_prob"
                                ].values[0]
                                color_vis = "green" if green_prob_val > 0.5 else "red"
                                for (
                                    tl_signal_stop_coord
                                ) in tl_signal_idx_2_stop_coordinates[tl_signal_idx]:
                                    ax.scatter(
                                        tl_signal_stop_coord[0],
                                        tl_signal_stop_coord[1],
                                        color="green"
                                        if tl_events_pred_current[
                                            f"{tl_signal_idx}_green_prob"
                                        ].values[0]
                                        > 0.5
                                        else "r",
                                        s=70,
                                    )
                                    ax.scatter(
                                        tl_signal_stop_coord[0],
                                        tl_signal_stop_coord[1],
                                        color=color_vis,
                                        s=30,
                                    )
                                    if tl_signal_idx not in tl_sig_idx_2_min_x:
                                        tl_sig_idx_2_min_x[
                                            tl_signal_idx
                                        ] = tl_signal_stop_coord[0]
                                        tl_sig_idx_2_min_y[
                                            tl_signal_idx
                                        ] = tl_signal_stop_coord[1]
                                    else:
                                        tl_sig_idx_2_min_x[tl_signal_idx] = min(
                                            tl_signal_stop_coord[0],
                                            tl_sig_idx_2_min_x[tl_signal_idx],
                                        )
                                        tl_sig_idx_2_min_y[tl_signal_idx] = min(
                                            tl_signal_stop_coord[1],
                                            tl_sig_idx_2_min_y[tl_signal_idx],
                                        )

                                text_coord = (
                                    tl_sig_idx_2_min_x[tl_signal_idx],
                                    tl_sig_idx_2_min_y[tl_signal_idx] - 4,
                                )

                                ax.text(
                                    *text_coord,
                                    f"{green_prob_val * 100 if green_prob_val > 0.5 else 100 - green_prob_val * 100:.0f}% (tte:{tl_events_pred_current[f'{tl_signal_idx}_tte_mode'].values[0]:.1f} sec)",
                                    color=color_vis,
                                    fontsize=12,
                                )
                                ax.arrow(
                                    *text_coord,
                                    tl_sig_idx_2_min_x[tl_signal_idx] - text_coord[0],
                                    tl_sig_idx_2_min_y[tl_signal_idx] - text_coord[1],
                                    color=color_vis,
                                )

                        plt.title("Predictions")
                        fig.canvas.draw()
                        data = np.fromstring(
                            fig.canvas.tostring_rgb(), dtype=np.uint8, sep=""
                        ).reshape(fig.canvas.get_width_height() + (3,))
                        plt.close()

                        im_plt = cv2.resize(
                            data, (im_filtered.shape[1], im_filtered.shape[0])
                        )
                        im_plt = cv2.resize(
                            im_plt, (im_filtered.shape[1], im_filtered.shape[0])
                        )
                        im = np.concatenate(
                            (
                                imutils.rotate(im_filtered, 180 * yaw / np.pi),
                                np.zeros(
                                    (im_filtered.shape[0], 20, im_filtered.shape[2]),
                                    dtype="uint8",
                                ),
                                im_plt,
                            ),
                            axis=1,
                        )

                        if tl_events_df is not None:
                            tl_events_current = tl_events_df.loc[
                                [(master_intersection_idx, timestamp)]
                            ]
                            rnn_events = tl_events_current["rnn_inputs_raw"].values[0]
                            tl_events = [
                                x for x in rnn_events if x[0] in traffic_light_ids_all
                            ]
                            lane_events = [
                                x for x in rnn_events if x[0] in lane_id_2_idx
                            ]

                            fig = plt.figure(figsize=(3, 3))
                            ax = fig.add_subplot(111)
                            plt.axis("off")
                            for tl_id, ohe_type in tl_events:
                                if tl_id not in change_source_id_2_idx:
                                    change_source_id_2_idx[tl_id] = len(
                                        change_source_id_2_idx
                                    )
                                color_i = change_source_id_2_idx[tl_id]
                                tl_coord = get_traffic_light_coordinates(tl_id)
                                ax.scatter(
                                    tl_coord[0],
                                    tl_coord[1],
                                    color=cycled_colors[color_i % len(cycled_colors)],
                                    s=50,
                                )
                                if ohe_type[0] == 1 and np.sum(ohe_type) == 1:
                                    ohe_based_color_check = "green"
                                elif ohe_type[1] == 1 and np.sum(ohe_type) == 1:
                                    ohe_based_color_check = "red"
                                else:
                                    ohe_based_color_check = "BUG!!!!"
                                ax.text(
                                    tl_coord[0] + 1.7,
                                    tl_coord[1] - 5,
                                    ohe_based_color_check,
                                    color=cycled_colors[color_i % len(cycled_colors)],
                                    fontsize=12,
                                )
                                seq_order_i += 1

                            for lane_id, ohe_type in lane_events:
                                if lane_id not in change_source_id_2_idx:
                                    change_source_id_2_idx[lane_id] = len(
                                        change_source_id_2_idx
                                    )
                                color_i = change_source_id_2_idx[lane_id]
                                seq_order_text = ""  # f'  seq order: {seq_order_i}'
                                if ohe_type[2] == 1 and np.sum(ohe_type) == 1:
                                    ohe_based_text_check = "idle"
                                elif ohe_type[3] == 1 and np.sum(ohe_type) == 1:
                                    ohe_based_text_check = "on move"
                                else:
                                    ohe_based_text_check = "BUG!!!!"
                                for j, lane_id_ in enumerate(lane_id.split("_")):
                                    plot_line(
                                        ax,
                                        lane_id_,
                                        color=cycled_colors[
                                            color_i % len(cycled_colors)
                                        ],
                                        text1=seq_order_text if j == 0 else "",
                                        text2=ohe_based_text_check,
                                    )
                                seq_order_i += 1

                            ax.set_xlim((centroid[0] - 85, centroid[0] + 85))
                            ax.set_ylim((centroid[1] - 85, centroid[1] + 85))

                            plt.title("Inputs")
                            fig.canvas.draw()
                            data_X = np.fromstring(
                                fig.canvas.tostring_rgb(), dtype=np.uint8, sep=""
                            ).reshape(fig.canvas.get_width_height() + (3,))
                            plt.close()
                            im_plt = cv2.resize(
                                data_X, (im_filtered.shape[1], im_filtered.shape[0])
                            )
                            im_plt = cv2.resize(
                                im_plt, (im_filtered.shape[1], im_filtered.shape[0])
                            )
                            im = np.concatenate(
                                (
                                    imutils.rotate(im_filtered, 180 * yaw / np.pi),
                                    np.zeros(
                                        (
                                            im_filtered.shape[0],
                                            20,
                                            im_filtered.shape[2],
                                        ),
                                        dtype="uint8",
                                    ),
                                    im_plt,
                                ),
                                axis=1,
                            )
                            output = store_output(
                                im, output, start_scene, end_scene, intersection, show
                            )

                        if tl_events_df is None:
                            output = store_output(
                                im,
                                output,
                                start_scene,
                                end_scene,
                                intersection,
                                show,
                                pred=True,
                            )

        if output is not None:
            output.release()
            output = None


if vis_inputs:
    events_df_path = (
        f"input/tl_events_df_{dataset_basename}_0_intersection_{intersection_i}.hdf5"
    )
    tl_events_df_trn = pd.read_hdf(events_df_path, key="data")
    run_vis(
        frame_dataset,
        dataset_filtered,
        [start_scene, end_scene],
        scenes_per_video=scenes_per_video,
        intersection=intersection_i,
        tl_events_df=tl_events_df_trn.set_index(
            ["master_intersection_idx", "timestamp"]
        ),
    )

if vis_predictions:
    pred_df_path = f"outputs/tl_predictions/tl_pred_{dataset_basename}_0_enriched_intersection_{intersection_i}.hdf5"
    tl_events_df_pred = pd.read_hdf(pred_df_path, key="data")
    run_vis(
        frame_dataset,
        dataset_filtered,
        [start_scene, end_scene],
        scenes_per_video=scenes_per_video,
        intersection=intersection_i,
        tl_events_df_pred=tl_events_df_pred.set_index(["scene_idx", "scene_frame_idx"]),
    )
