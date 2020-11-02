#!/bin/bash
for i in {0..9}
do
    python tl_light_predictor_intersection.py --trn-dataset-names "train_full_p1_intersection_${i}.hdf5" --val-file-name "tl_events_df_val_intersection_${i}.hdf5" --intersection-i "${i}" > "../logs/tl_train_fold_0_intersection_${i}.log"
done