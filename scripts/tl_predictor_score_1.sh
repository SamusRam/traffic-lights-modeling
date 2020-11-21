#!/bin/bash
for i in {0..9}
do
  python tl_light_predictor_intersection.py --fold-i 1 --trn-dataset-names "tl_events_df_validate_0_intersection_${i}.hdf5" --predict --intersection-i "${i}" --batch-size 512 --prediction-id 'validate' > "../logs/tl_val_fold_1_intersection_${i}_prediction.log"
  python tl_light_predictor_intersection.py --fold-i 1 --trn-dataset-names "tl_events_df_test_0_intersection_${i}.hdf5" --predict --intersection-i "${i}" --batch-size 512 --prediction-id 'test' > "../logs/tl_test_fold_1_intersection_${i}_prediction.log"
  python tl_light_predictor_intersection.py --fold-i 1 --trn-dataset-names "tl_events_df_train_full_1_intersection_${i}.hdf5" --predict --intersection-i "${i}" --batch-size 512 > "../logs/tl_fold_1_intersection_${i}_prediction.log"
done