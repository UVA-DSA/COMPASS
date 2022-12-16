# Tune learning rate, batch size weight decay, and number of epochs
# Make sure to run preprocess.py <set> <var> <labeltype> <valtype>
# before tune.py with the same arguments

from config import input_size, val_type, num_class, raw_feature_dir, validation_trial, validation_trial_train, sample_rate,dataset_name
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from functools import partial
from lstm_model import LSTM_Layer
from train_test_val import train_model_parameter, test_model
import numpy as np
import torch
import os
import sys
import pdb
from utils import get_cross_val_splits, get_cross_val_splits_LOUO
from data_loading import RawFeatureDataset
from logger import Logger
import utils
import torch
import torch.nn as nn
import json

from preprocess import processArguments
# import tensorflow as tf

# # add to the top of your code under import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config....)

global val_type

# Modify config.json values based on best config found
def updateJSONtcnparams(dataset_name, batch_size, epoch, learning_rate, weight_decay):
    # dataset_name passed as argument from command line
    print("Updating tcn params in config for " + dataset_name + " with " + var + " and " + labeltype + "...")

    # Load saved parameters from json
    all_params = json.load(open('config.json'))


    # Make changes to params:
    # Update tcn params
    all_params[dataset_name]["tcn_params"]["config"]["batch_size"] = batch_size
    all_params[dataset_name]["tcn_params"]["config"]["epoch"] = epoch
    all_params[dataset_name]["tcn_params"]["config"]["learning_rate"] = learning_rate
    all_params[dataset_name]["tcn_params"]["config"]["weight_decay"] = weight_decay

    # Write updated params to config.json
    with open('config.json', 'w') as jF:
        json.dump(all_params, jF, indent=4, sort_keys=True)









def tuneParams(rate, size, decay, num_samples=1, max_num_epochs=50):
    #print(rate)
    #print(size)
    #print(decay)
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
        resources_per_trial={"cpu": 14, "gpu": 1},
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

    #print(best_trial.config)
    rate = best_trial.config['learning_rate']
    #print(rate)
    size = best_trial.config['batch_size']
    decay = best_trial.config['weight_decay']

    return rate, size, decay


if __name__ == "__main__":
    global valtype
    # Process arguments from command line and get set, var, and labeltype
    set, var, labeltype, valtype = processArguments(sys.argv)

    # loadConfig() not needed here assuming preprocess.py was run
    # immediately before this which correctly updates config.json
    # and sets up the training data and pkl files for this tuning


    # 7/22/22 By-passing tuning and using hyperparameter values determined
    # using a gridsearch for learning_rate and weight_decay on
    # JIGSAWS gesture velocity LOSO/LOUO
    # hardcode batch_size=1 and epoch=60
    if valtype == "LOSO":
        updateJSONtcnparams(set, 1, 60, 0.00005, 0.01) # based on JIGSAWS gesture velocity LOSO
        #updateJSONtcnparams(set, 1, 60, 0.0001, 0.0001)  # based on JIGSAWS MPbaseline velocity LOSO
    elif valtype == "LOUO":
        updateJSONtcnparams(set, 1, 60, 0.00005, 0.0005) # based on JIGSAWS gesture velocity LOUO
        #updateJSONtcnparams(set, 1, 60, 0.0001, 0.001) # based on JIGSAWS MPbaseline velocity LOUO
    elif valtype == "LOTO":
        updateJSONtcnparams(set, 1, 60, 0.0001, 0.001)   # based on JIGSAWS gesture velocity LOTO
    '''
    # Number of CPU and GPU resources are hard coded in main_tcn, make
    # sure to change if running on a different computer
    # First, tune learning rate, batch size, and weight decay
    rate, size, decay = tuneParams(0, 0, 0, num_samples=100, max_num_epochs=30)

    # Hard coded for now...
    epoch = 50
    # Update json
    updateJSONtcnparams(set, size, epoch, rate, decay)

    # Then, pass returned config to next tuning loop to tune number of epochs
    tuneParams(rate, size, decay, num_samples=1, max_num_epochs=60)
    '''
