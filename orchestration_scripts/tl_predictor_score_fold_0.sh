#!/bin/bash
for i in {0..9}
do
  python -m lyft_trajectories.model.tl_light_predictor_intersection --dataset-names "tl_events_df_validate_0_intersection_${i}.hdf5" --predict --intersection-i "${i}" --batch-size 512 --prediction-id 'validate'
  python -m lyft_trajectories.model.tl_light_predictor_intersection --dataset-names "tl_events_df_test_0_intersection_${i}.hdf5" --predict --intersection-i "${i}" --batch-size 512 --prediction-id 'test'
  python -m lyft_trajectories.model.tl_light_predictor_intersection --dataset-names "tl_events_df_train_full_0_intersection_${i}.hdf5" --predict --intersection-i "${i}" --batch-size 512
done
