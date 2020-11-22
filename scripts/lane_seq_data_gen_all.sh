#!/bin/bash
python lane_seq_data_gen.py --zarr-basename "validate" --traffic-light-predictions-base-name "tl_pred_validate_*_enriched"
python lane_seq_data_gen.py --zarr-basename "test" --traffic-light-predictions-base-name "tl_pred_test_*_enriched"
python lane_seq_data_gen.py --zarr-basename "train_full" --timestamp-max "2020-01-01" --traffic-light-predictions-base-name "tl_pred_0_enriched" --splitting-dates "2019-11-03" "2019-11-17" "2019-11-28" "2019-12-14"
python lane_seq_data_gen.py --fold-i 1 --zarr-basename "train_full" --timestamp-min "2020-01-01" --traffic-light-predictions-base-name "tl_pred_1_enriched" --splitting-dates "2020-01-17" "2020-02-03" "2020-02-17" "2020-02-27"
