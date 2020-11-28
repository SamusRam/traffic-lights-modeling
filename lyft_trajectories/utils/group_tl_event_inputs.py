import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-names", nargs="*", action="append")
parser.add_argument("--output-name", default="")
args = parser.parse_args()

dataset_names = args.dataset_names[0]
print(dataset_names)
output_name = args.output_name

input_paths = [f"input/{name}" for name in dataset_names]
tl_events_df_concat = pd.concat([pd.read_hdf(path, key="data") for path in input_paths])
tl_events_df_concat.to_hdf(os.path.join("input", output_name), key="data")
