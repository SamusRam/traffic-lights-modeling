import argparse
import gc
import logging

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--events-basename")
parser.add_argument("--predictions-basename")

args = parser.parse_args()
events_basename = args.events_basename
predictions_basename = args.predictions_basename
N_INTERSECTIONS = 10
logger = logging.getLogger(__name__)
logger.info(
    f"Events observed: {events_basename}, predictions for all events: {predictions_basename}"
)

for intersection_i in range(N_INTERSECTIONS):
    logger.info(f"Processing intersection {intersection_i}")
    tl_events_all_pred = pd.read_hdf(
        f"outputs/tl_predictions/{predictions_basename}_intersection_{intersection_i}.hdf5",
        key="data",
    )
    tl_events = pd.read_hdf(
        f"input/{events_basename}_intersection_{intersection_i}.hdf5", key="data"
    )
    merged_df = tl_events_all_pred.merge(
        tl_events[["scene_idx", "frame_idx", "tl_signal_classes"]],
        how="left",
        left_on=["scene_idx", "scene_frame_idx"],
        right_on=["scene_idx", "frame_idx"],
    )
    merged_cols_set = set(merged_df.columns)
    for index, row in merged_df.iterrows():
        for tl_signal_i, derived_in_observation_class in row[
            "tl_signal_classes"
        ].items():
            derived_green_prob = 1.0 if derived_in_observation_class == 1 else 0.0
            if (
                f"{tl_signal_i}_green_prob" in merged_cols_set
            ):  # as ego sdv might be assigned to a neighbouring intersection
                merged_df[f"{tl_signal_i}_green_prob"].loc[index] = derived_green_prob
    merged_df[tl_events_all_pred.columns].to_hdf(
        f"outputs/tl_predictions/{predictions_basename}_enriched_intersection_{intersection_i}.hdf5",
        key="data",
    )
    del merged_df, tl_events, tl_events_all_pred
    gc.collect()
