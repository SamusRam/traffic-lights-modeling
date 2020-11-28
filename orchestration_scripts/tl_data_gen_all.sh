#!/bin/bash
python -m lyft_trajectories.data_preprocessing.tl_light_data_gen --input-name "validate"
python -m lyft_trajectories.utils.split_tl_data_per_intersection --joint-hdf-file "tl_events_df_validate_0.hdf5"
python -m lyft_trajectories.data_preprocessing.tl_light_data_gen --input-name "train_full" --timestamp-max "2020-01-01"
python -m lyft_trajectories.utils.split_tl_data_per_intersection --joint-hdf-file "tl_events_df_train_full_0.hdf5"
python -m lyft_trajectories.data_preprocessing.tl_light_data_gen --input-name "train_full" --timestamp-min "2020-01-01" --fold-i 1
python -m lyft_trajectories.data_preprocessing.tl_light_data_gen --input-name "train"
python -m lyft_trajectories.utils.group_tl_event_inputs --dataset-names "tl_events_df_train_full_1.hdf5" "tl_events_df_train_0.hdf5" --output-name "tl_events_df_train_full_1.hdf5"
python -m lyft_trajectories.utils.split_tl_data_per_intersection --joint-hdf-file "tl_events_df_train_full_1.hdf5"
python -m lyft_trajectories.data_preprocessing.tl_light_data_gen --input-name "test"
python -m lyft_trajectories.utils.split_tl_data_per_intersection --joint-hdf-file "tl_events_df_test_0.hdf5"
