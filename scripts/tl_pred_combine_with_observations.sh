#!/bin/bash
python tl_pred_combined_with_observations.py --events-basename "tl_events_df_test_0" --predictions-basename "tl_pred_test_0" > "../logs/tl_test_0_pred_event_combi.log"
python tl_pred_combined_with_observations.py --events-basename "tl_events_df_validate_0" --predictions-basename "tl_pred_validate_0" > "../logs/tl_validate_0_pred_event_combi.log"
python tl_pred_combined_with_observations.py --events-basename "tl_events_df_test_0" --predictions-basename "tl_pred_test_1" > "../logs/tl_test_1_pred_event_combi.log"
python tl_pred_combined_with_observations.py --events-basename "tl_events_df_validate_0" --predictions-basename "tl_pred_validate_1" > "../logs/tl_validate_1_pred_event_combi.log"
python tl_pred_combined_with_observations.py --events-basename "tl_events_df_train_full_0" --predictions-basename "tl_pred_0" > "../logs/tl_train_0_pred_event_combi.log"
python tl_pred_combined_with_observations.py --events-basename "tl_events_df_train_full_1" --predictions-basename "tl_pred_1" > "../logs/tl_train_1_pred_event_combi.log"
