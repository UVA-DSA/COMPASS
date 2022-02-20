from __future__ import division
from __future__ import print_function

import json
import pdb
import math

all_params = json.load(open('config.json'))
dataset_name = all_params['dataset_name']
val_type = all_params["experiment_setup"]["val_type"]
tcn_model_params = all_params[dataset_name]["tcn_params"]
input_size = all_params[dataset_name]["input_size"]
num_class=all_params[dataset_name]["gesture_class_num"]

raw_feature_dir = all_params[dataset_name]["raw_feature_dir"]
test_trial=all_params[dataset_name]["test_trial"]
train_trial = all_params[dataset_name]["train_trial"]
sample_rate = all_params[dataset_name]["sample_rate"]
gesture_class_num = all_params[dataset_name]["gesture_class_num"]
data_transform_path = all_params[dataset_name]["data_transform_path"]
# for parameter tuning
validation_trial = all_params[dataset_name]["validation_trial"]
validation_trial_train = all_params[dataset_name]["validation_trial_train"]
#[2,3,4,5,6] #"Desk" or [2,3,4,5]"JIGSAWS"
LOCS =all_params[dataset_name]["locs"]