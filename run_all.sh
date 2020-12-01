#!/bin/bash
python -m lyft_trajectories.utils.dataset_enrichment --add-standard-mask-indices --dataset-path scenes/validate.zarr --min-frame-history 4
python -m lyft_trajectories.utils.dataset_enrichment --add-standard-mask-indices --dataset-path scenes/train.zarr --min-frame-history 4
python -m lyft_trajectories.utils.dataset_enrichment --add-standard-mask-indices --dataset-path scenes/train_full.zarr --min-frame-history 4
python -m lyft_trajectories.utils.dataset_enrichment --add-standard-mask-indices --dataset-path scenes/test.zarr --min-frame-history 4 --min-frame-future 0

bash orchestration_scripts/tl_data_gen_all.sh

bash orchestration_scripts/tl_pred_train_fold_0.sh
bash orchestration_scripts/tl_pred_train_fold_1.sh

bash orchestration_scripts/tl_predictor_score_fold_0.sh
bash orchestration_scripts/tl_predictor_score_fold_0.sh

bash orchestration_scripts/tl_pred_combine_with_observations.sh

python -m lyft_trajectories.test.vis_check --vis-inputs --vis-predictions
