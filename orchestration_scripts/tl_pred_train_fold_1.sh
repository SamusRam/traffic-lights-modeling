#!/bin/bash
for i in {0..9}; do
 python -m lyft_trajectories.model.tl_light_predictor_intersection --fold-i 1 --trn-dataset-names "tl_events_df_train_full_1_intersection_${i}.hdf5" --val-file-name "tl_events_df_validate_0_intersection_${i}.hdf5" --intersection-i "${i}"
done
