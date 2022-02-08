# Tune learning rate, batch size weight decay, and number of epochs
# Make sure to run preprocess.py <set> <var> <labeltype>
# before tune.py with the same arguments
"""

@author: Zongyu-zoey-li & kch4fk
"""
from config import input_size, num_class, raw_feature_dir, validation_trial, validation_trial_train, sample_rate,dataset_name
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from functools import partial
from lstm_model import LSTM_Layer
from train_test_cross import train_model_parameter, test_model
import numpy as np
import torch
import os
import sys
import pdb
from utils import get_cross_val_splits
from data_loading import RawFeatureDataset
from logger import Logger
import utils
import torch
import torch.nn as nn

# Process arguments from command line
# returns set, input variables, and labeltype
def processArguments(args):
    # Get arguments from command line
    try:
        set=args[1]
        # Check if valid set
        if set not in ["DESK", "JIGSAWS"]: # , "ROSMA", "All"]:
            print("Please choose set: DESK, JIGSAWS") #, ROSMA, All")
            sys.exit()
    except:
        print("Please choose set: DESK, JIGSAWS") #, ROSMA, All")
        sys.exit()

    # Get orientation or velocity from command line
    try:
        var = args[2]
        # Check if valid var
        if var not in ["velocity", "orientation", "all"]:
            print("Please choose input variable: velocity orientation all")
            sys.exit()
    except:
        print("Please choose input variable: velocity orientation all")
        sys.exit()

    # Get MP or gesture from command line
    try:
        labeltype = args[3]
        # Check if valid labeltype
        if labeltype not in ["MP", "gesture"]:
            print("Please choose label type: MP gesture")
            sys.exit()
    except:
        print("Please choose label type: MP gesture")
        sys.exit()

    # return arguments
    return set, var, labeltype


def tuneParams(rate, size, decay, num_samples=1, max_num_epochs=50):

    # For parameter tuning:
    if rate == 0:
        config = {"learning_rate":tune.loguniform(1e-5,1e-3), "batch_size":1, "weight_decay": tune.loguniform(1e-4,1e-2)}
    else:
        config = {"learning_rate":rate, "batch_size":size, "weight_decay": decay}

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_model_parameter,type='tcn',input_size=input_size,\
             num_class=num_class,num_epochs=max_num_epochs,dataset_name=dataset_name,\
                 sample_rate=sample_rate),
        resources_per_trial={"cpu": 10, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)


    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    return rate, size, decay


if __name__ == "__main__":

    # Process arguments from command line and get set, var, and labeltype
    set, var, labeltype = processArguments(sys.argv)

    # loadConfig() not needed here assuming preprocess.py was run
    # immediately before this which correctly updates config.json
    # and sets up the training data and pkl files for this tuning


    # Number of CPU and GPU resources are hard coded in main_tcn, make
    # sure to change if running on a different computer
    # First, tune learning rate, batch size, and weight decay
    rate, size, decay = tuneParams(0, 0, 0, num_samples=100, max_num_epochs=60)
    # Then, pass returned config to next tuning loop to tune number of epochs
    tuneParams(rate, size, decay, num_samples=1, max_num_epochs=60)
