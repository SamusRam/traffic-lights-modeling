#!/bin/bash
python lane_seq_data_gen.py --zarr-basename "validate" --traffic-light-predictions-base-name "tl_pred_validate_*_enriched" > "../logs/validate_lane_seq_data_gen.log"
python lane_seq_data_gen.py --zarr-basename "test" --traffic-light-predictions-base-name "tl_pred_test_*_enriched" > "../logs/test_lane_seq_data_gen.log"
python lane_seq_data_gen.py --zarr-basename "train_full" --timestamp-max "2020-01-01" --traffic-light-predictions-base-name "tl_pred_0_enriched" > "../logs/train_full_0_lane_seq_data_gen.log"
python lane_seq_data_gen.py --fold-i 1 --zarr-basename "train_full" --timestamp-min "2020-01-01" --traffic-light-predictions-base-name "tl_pred_1_enriched" > "../logs/train_full_1_lane_seq_data_gen.log"
