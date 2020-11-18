import os
import pandas as pd
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--joint-hdf-file')
args = parser.parse_args()

input_name = args.joint_hdf_file

tl_events_df_joint = pd.read_hdf(os.path.join('../input', input_name), key='data')
for intersection_idx in tqdm(tl_events_df_joint['master_intersection_idx'].unique()):
    tl_events_df_intersection = tl_events_df_joint[tl_events_df_joint['master_intersection_idx'] == intersection_idx]
    tl_events_df_intersection.to_hdf(os.path.join('../input', f'{os.path.splitext(input_name)[0]}_intersection_{intersection_idx}.hdf5'), key='data')