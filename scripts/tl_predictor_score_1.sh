#!/bin/bash
for i in {0..9}
do
  if [ "${i}" != 3 ] && [ "${i}" != 9 ]; then
    python tl_light_predictor_intersection.py --fold-i 1 --trn-dataset-names "tl_events_df_train_full_1_intersection_${i}.hdf5" --predict --intersection-i "${i}" --batch-size 512 > "../logs/tl_fold_1_intersection_${i}_prediction.log"
  fi
done