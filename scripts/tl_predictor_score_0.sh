#!/bin/bash
for i in {0..9}
do
    python tl_light_predictor_intersection.py --trn-dataset-names "tl_events_df_train_full_0_intersection_${i}.hdf5" --predict --intersection-i "${i}" --batch-size 512 > "../logs/tl_fold_0_intersection_${i}_prediction.log"
done