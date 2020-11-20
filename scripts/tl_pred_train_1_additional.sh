#!/bin/bash
for ((i=9; i>=6; i--)); do
 python tl_light_predictor_intersection.py --fold-i 1 --different-gpu --trn-dataset-names "tl_events_df_train_full_1_intersection_${i}.hdf5" --val-file-name "tl_events_df_validate_0_intersection_${i}.hdf5" --intersection-i "${i}" > "../logs/tl_train_fold_1_intersection_${i}.log"
done
