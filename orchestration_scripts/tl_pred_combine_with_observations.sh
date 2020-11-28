#!/bin/bash
python -m lyft_trajectories.data_preprocessing.tl_pred_combined_with_observations.py --events-basename "tl_events_df_test_0" --predictions-basename "tl_pred_test_0"
python -m lyft_trajectories.data_preprocessing.tl_pred_combined_with_observations.py --events-basename "tl_events_df_validate_0" --predictions-basename "tl_pred_validate_0"
python -m lyft_trajectories.data_preprocessing.tl_pred_combined_with_observations.py --events-basename "tl_events_df_test_0" --predictions-basename "tl_pred_test_1"
python -m lyft_trajectories.data_preprocessing.tl_pred_combined_with_observations.py --events-basename "tl_events_df_validate_0" --predictions-basename "tl_pred_validate_1"
python -m lyft_trajectories.data_preprocessing.tl_pred_combined_with_observations.py --events-basename "tl_events_df_train_full_0" --predictions-basename "tl_pred_0"
python -m lyft_trajectories.data_preprocessing.tl_pred_combined_with_observations.py --events-basename "tl_events_df_train_full_1" --predictions-basename "tl_pred_1"
