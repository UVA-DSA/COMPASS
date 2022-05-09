from __future__ import division
from __future__ import print_function
# Kay Hutchinson 2/4/2022
# File for preprocessing the data and save it to csv files in the
# preprocessed folder. Then, encodes labels and saves to .pkl file.

# 4/25/22 Adding x,y coordinates for left and right grippers and needle
# based on Cogito labels. This means that S, NP, and KT will have different
# numbers of variables and different variables from the kinematics files and
# additional if conditions need to be added to this file to handle that.

# 5/6/22 Adding other MP label options: baseline, combined, left, and right
# baseline is original context to MP labels
# combined is [Touch,Grasp] -> Grasp and [Release,Untouch] -> Release combined
# left and right are the combined transcript but split by relevant side and
#     filled in with Idle(*) for that hand
# TO DO:
# add different MP paths to combined, left, and right

import glob
import sys
import os
import numpy as np
import pandas as pd
import pdb
import json
import math

from sklearn import preprocessing
import pickle

# Process arguments from command line
# returns set, input variables, labeltype, and valtype
def processArguments(args):
    # Get arguments from command line
    try:
        set=args[1]
        # Check if valid set
        if set not in ["DESK", "JIGSAWS", "All-5a","All-5b", "S", "NP", "KT", "PoaP", "PaS", "SNP", "PTPaS"]: # , "ROSMA", "All"]:
            print("Please choose set: DESK, JIGSAWS, All-5a, All-5b, S, NP, KT, PoaP, Pas, SNP, PTPaS") #, ROSMA, All")
            sys.exit()
    except:
        print("Please choose set: DESK, JIGSAWS, All-5a, All-5b, S, NP, KT, PoaP, Pas, SNP, PTPaS") #, ROSMA, All")
        sys.exit()

    # Get orientation or velocity from command line
    try:
        var = args[2]
        # Check if valid var
        if var not in ["velocity", "orientation", "all", "vis", "vis2"]:
            print("Please choose input variable: velocity orientation all vis vis2")
            sys.exit()
    except:
        print("Please choose input variable: velocity orientation all vis vis2")  # add vis2
        sys.exit()

    # Get MP or gesture from command line
    try:
        labeltype = args[3]
        # Check if valid labeltype
        if labeltype not in ["gesture", "MPbaseline", "MPcombined", "MPexchange", "MPleft", "MPright"]:
            print("Please choose label type: gesture MPbaseline MPcombined MPexchange MPleft MPright")
            sys.exit()
    except:
        print("Please choose label type: gesture MPbaseline MPcombined MPexchange MPleft MPright")
        sys.exit()

    # Get LOSO or LOUO from command line
    try:
        valtype = args[4]
        # Check if valid crossval
        if valtype not in ["LOSO", "LOUO"]:
            print("Please choose label type: LOSO LOUO")
            sys.exit()
    except:
        print("Please choose label type: LOSO LOUO")
        sys.exit()



    # return arguments
    return set, var, labeltype, valtype


# Load config parameters
def loadConfig(dataset_name, var, labeltype, valtype):
    # dataset_name passed as argument from command line
    print("Loading config for " + dataset_name + " with " + var + " and " + labeltype + "...")

    # Load saved parameters from json
    all_params = json.load(open('config.json'))

    # Get tcn model params and sizes, and input size and number of label classes
    tcn_model_params = all_params[dataset_name]["tcn_params"]

    # Calculate input size based on set and var
    if (var == "velocity"):
        input_size = 14
    elif (var == "orientation"):
        input_size = 16
    elif (var == "all"):
        input_size = 22
    elif (var == "vis") and (dataset_name == "S"):
        input_size = 20
    elif (var == "vis") and (dataset_name == "NP"):
        input_size = 20
    elif (var == "vis") and (dataset_name == "KT"):
        input_size = 18
    elif (var == "vis2") and (dataset_name == "NP"):
        input_size = 28
    elif (var == "vis2") and (dataset_name == "KT"):
        input_size = 58  # fill in, same as vis right now
    else:
        print("Please specify input size.")
        sys.exit()

    # Kernel size is shortest average label duration based on set and label type
    # 5/9/2022 changing to dictionary look up; rounded down to an odd number based on stats.py
    kernel_size_gesture_dict = {"DESK": 29, "JIGSAWS": 89, "S": 61, "NP": 97, "KT": 85}
    kernel_size_MPbaseline_dict = {"DESK": 0, "JIGSAWS": 23, "All-5a": 0,\
        "All-5b": 0, "S": 21, "NP": 23, "KT": 17, "PoaP": 0, "PaS": 0, "SNP": 23, "PTPaS": 0}
    kernel_size_MPcombined_dict = {"DESK": 0, "JIGSAWS": 27, "All-5a": 0,\
        "All-5b": 0, "S": 25, "NP": 25, "KT": 21, "PoaP": 0, "PaS": 0, "SNP": 25, "PTPaS": 0}
    kernel_size_MPexchange_dict = {"DESK": 0, "JIGSAWS": 31, "All-5a": 0,\
        "All-5b": 0, "S": 35, "NP": 25, "KT": 21, "PoaP": 0, "PaS": 0, "SNP": 33, "PTPaS": 0}
    kernel_size_MPleft_dict = {"DESK": 0, "JIGSAWS": 25, "All-5a": 0,\
        "All-5b": 0, "S": 25, "NP": 21, "KT": 15, "PoaP": 0, "PaS": 0, "SNP": 25, "PTPaS": 0}
    kernel_size_MPright_dict = {"DESK": 0, "JIGSAWS": 29, "All-5a": 0,\
        "All-5b": 0, "S": 25, "NP": 15, "KT": 23, "PoaP": 0, "PaS": 0, "SNP": 27, "PTPaS": 0}
    kernel_size_dict = {"gesture": kernel_size_gesture_dict, "MPbaseline": kernel_size_MPbaseline_dict, \
        "MPcombined": kernel_size_MPcombined_dict, "MPexchange": kernel_size_MPexchange_dict, \
        "MPleft": kernel_size_MPleft_dict, "MPright": kernel_size_MPright_dict}

    # Get kernel_size
    kernel_size = kernel_size_dict[labeltype][dataset_name]
    # Exit if not a valid combination of model settings
    if kernel_size == 0:
        print("Kernel size not specified yet")
        sys.exit()

    # Path to processed files in 'preprocessed' folder
    # raw_feature_dir contains a list of paths to the preprocessed folder of
    # the each task that will be included in the training set
    raw_feature_dir = all_params[dataset_name]["raw_feature_dir"]

    # Number of label classes
    # 5/9/2022 updating to dictionary look up
    # Determined using stats.py
    gesture_class_num_gesture_dict = {"DESK": 7, "JIGSAWS": 14, "S": 10, "NP": 10, "KT": 6}
    gesture_class_num_MPbaseline_dict = {"DESK": 0, "JIGSAWS": 6, "All-5a": 0,\
        "All-5b": 0, "S": 6, "NP": 6, "KT": 5, "PoaP": 0, "PaS": 0, "SNP": 6, "PTPaS": 0}
    gesture_class_num_MPcombined_dict = {"DESK": 0, "JIGSAWS": 6, "All-5a": 0,\
        "All-5b": 0, "S": 6, "NP": 6, "KT": 5, "PoaP": 0, "PaS": 0, "SNP": 6, "PTPaS": 0}
    gesture_class_num_MPexchange_dict = {"DESK": 0, "JIGSAWS": 7, "All-5a": 0,\
        "All-5b": 0, "S": 7, "NP": 7, "KT": 6, "PoaP": 0, "PaS": 0, "SNP": 7, "PTPaS": 0}
    gesture_class_num_MPleft_dict = {"DESK": 0, "JIGSAWS": 7, "All-5a": 0,\
        "All-5b": 0, "S": 7, "NP": 6, "KT": 6, "PoaP": 0, "PaS": 0, "SNP": 7, "PTPaS": 0}
    gesture_class_num_MPright_dict = {"DESK": 0, "JIGSAWS": 7, "All-5a": 0,\
        "All-5b": 0, "S": 7, "NP": 7, "KT": 6, "PoaP": 0, "PaS": 0, "SNP": 7, "PTPaS": 0}
    gesture_class_num_dict = {"gesture": gesture_class_num_gesture_dict, "MPbaseline": gesture_class_num_MPbaseline_dict, \
        "MPcombined": gesture_class_num_MPcombined_dict, "MPexchange": gesture_class_num_MPexchange_dict,\
        "MPleft": gesture_class_num_MPleft_dict, "MPright": gesture_class_num_MPright_dict}
    # Get gesture_class_num
    gesture_class_num = gesture_class_num_dict[labeltype][dataset_name]
    # Exit if not a valid combination of model settings
    if gesture_class_num == 0:
        print("Number of classes not specified yet")
        sys.exit()

    num_class = gesture_class_num

    # Sets for cross validation
    test_trial=all_params[dataset_name]["test_trial"]
    train_trial = all_params[dataset_name]["train_trial"]
    sample_rate = all_params[dataset_name]["sample_rate"]

    # For parameter tuning
    validation_trial = all_params[dataset_name]["validation_trial"]
    validation_trial_train = [2,3,4,5,6]

    return all_params, tcn_model_params, input_size, kernel_size, num_class, raw_feature_dir,\
     test_trial, train_trial, sample_rate, gesture_class_num, validation_trial, validation_trial_train

# Modify config.json values based on command line arguements and
# processed results from loadConfig
def updateJSON(dataset_name, var, labeltype, valtype, input_size, kernel_size, num_class):
    # dataset_name passed as argument from command line
    print("Updating config for " + dataset_name + " with " + var + " and " + labeltype + "...")

    # Load saved parameters from json
    all_params = json.load(open('config.json'))


    # Make changes to params:
    # Update dataset_name
    all_params["dataset_name"] = dataset_name
    # Update input size
    all_params[dataset_name]["input_size"] = input_size
    # Update num classes
    all_params[dataset_name]["gesture_class_num"] = num_class


    # Update LOSO/LOUO trial lists
    all_params["experiment_setup"]["val_type"] = valtype
    # Sets for cross validation
    if valtype == "LOSO":
        if dataset_name == "DESK":
            all_params[dataset_name]["test_trial"] = [1,2,3,4,5,6]
            all_params[dataset_name]["train_trial"] = [[2,3,4,5,6],[1,3,4,5,6],[1,2,4,5,6],[1,2,3,5,6],[1,2,3,4,6],[1,2,3,4,5]]
            all_params[dataset_name]["validation_trial"] = 1
            all_params[dataset_name]["validation_trial_train"] = [2,3,4,5,6]

        elif dataset_name in ["JIGSAWS", "S", "NP", "KT", "SNP"]:
            all_params[dataset_name]["test_trial"] = [1,2,3,4,5]
            all_params[dataset_name]["train_trial"] = [[2,3,4,5],[1,3,4,5],[1,2,4,5],[1,2,3,5],[1,2,3,4]]
            all_params[dataset_name]["validation_trial"] = 2
            all_params[dataset_name]["validation_trial_train"] = [2,3,4,5]

        elif dataset_name == "All-5a":
            all_params[dataset_name]["test_trial"] = [1,2,3,4,5]
            all_params[dataset_name]["train_trial"] = [[2,3,4,5],[1,3,4,5],[1,2,4,5],[1,2,3,5],[1,2,3,4]]
            all_params[dataset_name]["validation_trial"] = 1
            all_params[dataset_name]["validation_trial_train"] = [2,3,4,5]

        elif dataset_name == "All-5b":
            all_params[dataset_name]["test_trial"] = [1,2,3,4,5,6]
            all_params[dataset_name]["train_trial"] = [[2,3,4,5,6],[1,3,4,5,6],[1,2,4,5,6],[1,2,3,5,6],[1,2,3,4,6],[1,2,3,4,5]]
            all_params[dataset_name]["validation_trial"] = 1
            all_params[dataset_name]["validation_trial_train"] = [2,3,4,5,6]

        elif dataset_name in ["PoaP", "PaS"]:
            all_params[dataset_name]["test_trial"] = [1,2,3,4,5,6]
            all_params[dataset_name]["train_trial"] = [[2,3,4,5,6],[1,3,4,5,6],[1,2,4,5,6],[1,2,3,5,6],[1,2,3,4,6],[1,2,3,4,5]]
            all_params[dataset_name]["validation_trial"] = 1
            all_params[dataset_name]["validation_trial_train"] = [2,3,4,5,6]

        elif dataset_name == "PTPaS":
            print("test/train/val splits not defined yet")
            sys.exit()


    elif valtype == "LOUO":
        if dataset_name == "DESK":
            all_params[dataset_name]["test_trial"] = [1,2,3,4,5,6,7,8]
            all_params[dataset_name]["train_trial"] = [[2,3,4,5,6,7,8],[1,3,4,5,6,7,8],[1,2,4,5,6,7,8],[1,2,3,5,6,7,8],[1,2,3,4,6,7,8],[1,2,3,4,5,7,8],[1,2,3,4,5,6,8],[1,2,3,4,5,6,7]]
            all_params[dataset_name]["validation_trial"] = 1
            all_params[dataset_name]["validation_trial_train"] = [2,3,4,5,6,7,8]

        elif dataset_name  in ["JIGSAWS", "S", "NP", "KT", "SNP"]:
            all_params[dataset_name]["test_trial"] = [2,3,4,5,6,7,8,9]
            all_params[dataset_name]["train_trial"] = [[3,4,5,6,7,8,9],[2,4,5,6,7,8,9],[2,3,5,6,7,8,9],[2,3,4,6,7,8,9],[2,3,4,5,7,8,9],[2,3,4,5,6,8,9],[2,3,4,5,6,7,9],[2,3,4,5,6,7,8]]
            all_params[dataset_name]["validation_trial"] = 2
            all_params[dataset_name]["validation_trial_train"] = [2,3,4,5,6,7,8,9]

        elif dataset_name == "All-5a":
            all_params[dataset_name]["test_trial"] = [2,3,4,5,6,7,8,9]
            all_params[dataset_name]["train_trial"] = [[3,4,5,6,7,8,9],[2,4,5,6,7,8,9],[2,3,5,6,7,8,9],[2,3,4,6,7,8,9],[2,3,4,5,7,8,9],[2,3,4,5,6,8,9],[2,3,4,5,6,7,9],[2,3,4,5,6,7,8]]
            all_params[dataset_name]["validation_trial"] = 2
            all_params[dataset_name]["validation_trial_train"] = [3,4,5,6,7,8,9]

        elif dataset_name == "All-5b":
            all_params[dataset_name]["test_trial"] = [1,2,3,4,5,6,7,8]
            all_params[dataset_name]["train_trial"] = [[2,3,4,5,6,7,8],[1,3,4,5,6,7,8],[1,2,4,5,6,7,8],[1,2,3,5,6,7,8],[1,2,3,4,6,7,8],[1,2,3,4,5,7,8],[1,2,3,4,5,6,8],[1,2,3,4,5,6,7]]
            all_params[dataset_name]["validation_trial"] = 1
            all_params[dataset_name]["validation_trial_train"] = [2,3,4,5,6,7,8]

        elif dataset_name in ["PoaP", "PaS"]:
            all_params[dataset_name]["test_trial"] = [1,2,3,4,5,6,7,8,9,10,11,12]
            all_params[dataset_name]["train_trial"] = [[2,3,4,5,6,7,8,9,10,11,12],[1,3,4,5,6,7,8,9,10,11,12],[1,2,4,5,6,7,8,9,10,11,12],[1,2,3,5,6,7,8,9,10,11,12],[1,2,3,4,6,7,8,9,10,11,12],[1,2,3,4,5,7,8,9,10,11,12],[1,2,3,4,5,6,8,9,10,11,12],[1,2,3,4,5,6,7,9,10,11,12],[1,2,3,4,5,6,7,8,10,11,12],[1,2,3,4,5,6,7,8,9,11,12],[1,2,3,4,5,6,7,8,9,10,12],[1,2,3,4,5,6,7,8,9,10,11]]
            all_params[dataset_name]["validation_trial"] = 1
            all_params[dataset_name]["validation_trial_train"] = [2,3,4,5,6,7,8,9,10,11,12]

        elif dataset_name == "PTPaS":
            print("test/train/val splits not defined yet")
            sys.exit()


    # Update tcn params
    all_params[dataset_name]["tcn_params"]["model_params"]["class_num"] = num_class
    all_params[dataset_name]["tcn_params"]["model_params"]["encoder_params"]["input_size"] = input_size
    all_params[dataset_name]["tcn_params"]["model_params"]["encoder_params"]["kernel_size"] = kernel_size
    all_params[dataset_name]["tcn_params"]["model_params"]["decoder_params"]["kernel_size"] = kernel_size

    # LOCS
    if var == "velocity":
        LOCS=[  "PSML_position_x", "PSML_position_y", "PSML_position_z", \
                "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z", \
                "PSML_gripper_angle", \
                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z", \
                "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z", \
                "PSMR_gripper_angle"]
    elif var == "orientation":
        LOCS=[  "PSML_position_x", "PSML_position_y", "PSML_position_z", \
                "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w",\
                "PSML_gripper_angle", \
                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z", \
                "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w",\
                "PSMR_gripper_angle"]
    elif var == "all":
        LOCS=[  "PSML_position_x", "PSML_position_y", "PSML_position_z",\
                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w",\
                "PSML_gripper_angle", \
                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w",\
                "PSMR_gripper_angle"]
    elif (var == "vis") and (dataset_name == "S"):
        LOCS=[  "PSML_position_x", "PSML_position_y", "PSML_position_z",\
                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                "PSML_gripper_angle", \
                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                "PSMR_gripper_angle", \
                "L_Gripper_X", "L_Gripper_Y", "R_Gripper_X", "R_Gripper_Y", "Needle_X", "Needle_Y"]
    elif (var == "vis") and (dataset_name == "NP"):
        LOCS=[  "PSML_position_x", "PSML_position_y", "PSML_position_z",\
                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                "PSML_gripper_angle", \
                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                "PSMR_gripper_angle", \
                "L_Gripper_X", "L_Gripper_Y", "R_Gripper_X", "R_Gripper_Y", "Needle_X", "Needle_Y"]
    elif (var == "vis") and (dataset_name == "KT"):
        LOCS=[  "PSML_position_x", "PSML_position_y", "PSML_position_z",\
                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                "PSML_gripper_angle", \
                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                "PSMR_gripper_angle", \
                "L_Gripper_X", "L_Gripper_Y", "R_Gripper_X", "R_Gripper_Y"]
    elif (var == "vis2") and (dataset_name == "NP"):
        LOCS=[  "PSML_position_x", "PSML_position_y", "PSML_position_z",\
                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                "PSML_gripper_angle", \
                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                "PSMR_gripper_angle", \
                "L_Gripper_X", "L_Gripper_Y", "R_Gripper_X", "R_Gripper_Y", "Needle_X", "Needle_Y", \
                "R_4_X", "R_4_Y", "R_5_X", "R_5_Y", "R_6_X", "R_6_Y", "R_7_X", "R_7_Y"]
    elif (var == "vis2") and (dataset_name == "KT"):
        LOCS=[  "PSML_position_x", "PSML_position_y", "PSML_position_z",\
                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                "PSML_gripper_angle", \
                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                "PSMR_gripper_angle", \
                "L_Gripper_X", "L_Gripper_Y", "R_Gripper_X", "R_Gripper_Y", \
                "Thread_TL_1_X", "Thread_TL_1_Y", \
                "Thread_TL_2_X", "Thread_TL_2_Y", \
                "Thread_TL_3_X", "Thread_TL_3_Y", \
                "Thread_TL_4_X", "Thread_TL_4_Y", \
                "Thread_TL_5_X", "Thread_TL_5_Y", \
                "Thread_TR_1_X", "Thread_TR_1_Y", \
                "Thread_TR_2_X", "Thread_TR_2_Y", \
                "Thread_TR_3_X", "Thread_TR_3_Y", \
                "Thread_TR_4_X", "Thread_TR_4_Y", \
                "Thread_TR_5_X", "Thread_TR_5_Y", \
                "Thread_BL_1_X", "Thread_BL_1_Y", \
                "Thread_BL_2_X", "Thread_BL_2_Y", \
                "Thread_BL_3_X", "Thread_BL_3_Y", \
                "Thread_BL_4_X", "Thread_BL_4_Y", \
                "Thread_BL_5_X", "Thread_BL_5_Y", \
                "Thread_BR_1_X", "Thread_BR_1_Y", \
                "Thread_BR_2_X", "Thread_BR_2_Y", \
                "Thread_BR_3_X", "Thread_BR_3_Y", \
                "Thread_BR_4_X", "Thread_BR_4_Y", \
                "Thread_BR_5_X", "Thread_BR_5_Y"]
    all_params[dataset_name]["locs"] = LOCS

    # Update pickle file name
    pklFile = dataset_name + "_TRANSFORM_" + var + "_" + labeltype + ".pkl"
    all_params[dataset_name]["data_transform_path"] = os.path.join(os.getcwd(), dataset_name, pklFile)

    # Write updated params to config.json
    with open('config.json', 'w') as jF:
        json.dump(all_params, jF, indent=4, sort_keys=True)

    # Get updated tcn model params
    with open('config.json', 'r'):
        tcn_model_params = all_params[dataset_name]["tcn_params"]

    # return updated tcn params and LOCS
    return tcn_model_params, LOCS

# Preprocess files and save into preprocessed folder
def preprocess(set, var, labeltype, raw_feature_dir):
    print("Adding labels to selected kinematic data...")
    # raw_feature_dir contains a list of paths to the preprocessed folder of
    # the each task that will be included in the training set
    for sub in raw_feature_dir:

        # List all transcription file paths of labels
        if labeltype == "MPbaseline":
            ges_dir_all=glob.glob(os.path.join("/".join(sub.split('/')[0:-1]),"motion_primitives_baseline/*"))
        elif labeltype == "MPcombined":
            ges_dir_all=glob.glob(os.path.join("/".join(sub.split('/')[0:-1]),"motion_primitives_combined/*"))
        elif labeltype == "MPexchange":
            ges_dir_all=glob.glob(os.path.join("/".join(sub.split('/')[0:-1]),"motion_primitives_exchange/*"))
        elif labeltype == "MPleft":
            ges_dir_all=glob.glob(os.path.join("/".join(sub.split('/')[0:-1]),"motion_primitives_L/*"))
        elif labeltype == "MPright":
            ges_dir_all=glob.glob(os.path.join("/".join(sub.split('/')[0:-1]),"motion_primitives_R/*"))
        elif labeltype == "gesture":
            ges_dir_all=glob.glob(os.path.join("/".join(sub.split('/')[0:-1]),"gestures/*"))
        else:
            print("Please specify path to labels.")

        # List all kinematic file paths
        # velkinematics currently contains position, velocity, orientation, and gripper angle
        kine_dir_all = glob.glob(os.path.join("/".join(sub.split('/')[0:-1]),"velkinematics/*"))

        # For each transcription file
        for ges_dir in ges_dir_all:  #[0:1]:
            print("\t Preprocessing: " + ges_dir)

            # Read in label transcript
            if labeltype == "gesture":
                tg = pd.read_table(ges_dir, sep="\s+", header=None)
            else: # labeltype == "MPbaseline":
                tg = pd.read_table(ges_dir)

            #print(tg)

            # Get kin file associated with transcript
            if labeltype == "MPbaseline":
                kin_dir = ges_dir.replace("motion_primitives_baseline", "velkinematics").replace(".txt", ".csv")
            elif labeltype == "MPcombined":
                kin_dir = ges_dir.replace("motion_primitives_combined", "velkinematics").replace(".txt", ".csv")
            elif labeltype == "MPexchange":
                kin_dir = ges_dir.replace("motion_primitives_exchange", "velkinematics").replace(".txt", ".csv")
            elif labeltype == "MPleft":
                kin_dir = ges_dir.replace("motion_primitives_L", "velkinematics").replace(".txt", ".csv")
            elif labeltype == "MPright":
                kin_dir = ges_dir.replace("motion_primitives_R", "velkinematics").replace(".txt", ".csv")
            elif labeltype == "gesture":
                kin_dir = ges_dir.replace("gestures", "velkinematics").replace(".txt", ".csv")

            # Skip transcript and loop if no kinematic file is found
            if len(kin_dir)==0:continue

            # Read in kin data
            tb = pd.read_csv(kin_dir, sep="\t|,")
            #print(tb)

            # For JIGSAWS, only take PSM side
            if set == "JIGSAWS":
                tb = tb.loc[:, "PSML_position_x":]

            # Depending on chosen var, take only certain columns
            if var == "velocity":
                tb = tb.loc[:, ["PSML_position_x", "PSML_position_y", "PSML_position_z", \
                                "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z", \
                                "PSML_gripper_angle", \
                                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z", \
                                "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z", \
                                "PSMR_gripper_angle"]]
            elif var == "orientation":
                tb = tb.loc[:, ["PSML_position_x", "PSML_position_y", "PSML_position_z", \
                                "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w",\
                                "PSML_gripper_angle", \
                                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z", \
                                "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w",\
                                "PSMR_gripper_angle"]]
            elif var == "all":
                tb = tb.loc[:, ["PSML_position_x", "PSML_position_y", "PSML_position_z",\
                                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                                "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w",\
                                "PSML_gripper_angle", \
                                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                                "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w",\
                                "PSMR_gripper_angle"]]
            elif (var == "vis") and (set == "S"):
                tb = tb.loc[:, ["PSML_position_x", "PSML_position_y", "PSML_position_z",\
                                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                                "PSML_gripper_angle", \
                                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                                "PSMR_gripper_angle", \
                                "L_Gripper_X", "L_Gripper_Y", "R_Gripper_X", "R_Gripper_Y", "Needle_X", "Needle_Y"]]
            elif (var == "vis") and (set == "NP"):
                tb = tb.loc[:, ["PSML_position_x", "PSML_position_y", "PSML_position_z",\
                                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                                "PSML_gripper_angle", \
                                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                                "PSMR_gripper_angle", \
                                "L_Gripper_X", "L_Gripper_Y", "R_Gripper_X", "R_Gripper_Y", "Needle_X", "Needle_Y"]]
            elif (var == "vis") and (set == "KT"):
                tb = tb.loc[:, ["PSML_position_x", "PSML_position_y", "PSML_position_z",\
                                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                                "PSML_gripper_angle", \
                                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                                "PSMR_gripper_angle", \
                                "L_Gripper_X", "L_Gripper_Y", "R_Gripper_X", "R_Gripper_Y"]]
            elif (var == "vis2") and (set == "NP"):
                tb = tb.loc[:, ["PSML_position_x", "PSML_position_y", "PSML_position_z",\
                                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                                "PSML_gripper_angle", \
                                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                                "PSMR_gripper_angle", \
                                "L_Gripper_X", "L_Gripper_Y", "R_Gripper_X", "R_Gripper_Y", "Needle_X", "Needle_Y", \
                                "R_4_X", "R_4_Y", "R_5_X", "R_5_Y", "R_6_X", "R_6_Y", "R_7_X", "R_7_Y"]]
            elif (var == "vis2") and (set == "KT"):
                tb = tb.loc[:, ["PSML_position_x", "PSML_position_y", "PSML_position_z",\
                                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                                "PSML_gripper_angle", \
                                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                                "PSMR_gripper_angle", \
                                "L_Gripper_X", "L_Gripper_Y", "R_Gripper_X", "R_Gripper_Y", \
                                "Thread_TL_1_X", "Thread_TL_1_Y", \
                                "Thread_TL_2_X", "Thread_TL_2_Y", \
                                "Thread_TL_3_X", "Thread_TL_3_Y", \
                                "Thread_TL_4_X", "Thread_TL_4_Y", \
                                "Thread_TL_5_X", "Thread_TL_5_Y", \
                                "Thread_TR_1_X", "Thread_TR_1_Y", \
                                "Thread_TR_2_X", "Thread_TR_2_Y", \
                                "Thread_TR_3_X", "Thread_TR_3_Y", \
                                "Thread_TR_4_X", "Thread_TR_4_Y", \
                                "Thread_TR_5_X", "Thread_TR_5_Y", \
                                "Thread_BL_1_X", "Thread_BL_1_Y", \
                                "Thread_BL_2_X", "Thread_BL_2_Y", \
                                "Thread_BL_3_X", "Thread_BL_3_Y", \
                                "Thread_BL_4_X", "Thread_BL_4_Y", \
                                "Thread_BL_5_X", "Thread_BL_5_Y", \
                                "Thread_BR_1_X", "Thread_BR_1_Y", \
                                "Thread_BR_2_X", "Thread_BR_2_Y", \
                                "Thread_BR_3_X", "Thread_BR_3_Y", \
                                "Thread_BR_4_X", "Thread_BR_4_Y", \
                                "Thread_BR_5_X", "Thread_BR_5_Y"]]

            # For each line in the label transcript, get start frame, end frame,
            # and gesture label
            for i in range(tg.shape[0]):
                if (set == "DESK") and (labeltype == "gesture"):
                    start_ = int(tg.iloc[i,0])
                    end_ = int(tg.iloc[i,1])
                    label = tg.iloc[i,2]
                    #gesture = label
                elif (set in ["JIGSAWS", "S", "NP", "KT"]) and (labeltype == 'gesture'):
                    start_ = int(tg.iloc[i,0])
                    end_ = int(tg.iloc[i,1])
                    label = tg.iloc[i,2]
                else: # labeltype == "MPbaseline":
                    line = tg.iloc[i,0].split(" ")
                    start_ = int(line[0])  #int(tg.iloc[i,0])  #line[0])
                    end_ = int(line[1])   #int(tg.iloc[i,1])   #line[1])
                    # Only use MP as label, no context
                    label = line[2].split("(")[0]   #tg.iloc[i,2].split("(")[0]  #line[2].split("(")


                # Create array of labels of the size of the duration of the MP
                fill = [label]*int(end_-start_+1)
                # Add array of labels to kin data
                # Get number of rows in kin file and don't try to write outside tb
                nrows = tb.shape[0]
                if end_ >= nrows:
                    end_ = nrows-1
                    fill = [label]*int(end_-start_+1)
                tb.loc[start_:end_,"Y"]=fill

            save_dir = os.path.join(os.path.join("/".join(sub.split('/')[0:-1]),'preprocessed'),kin_dir.split('/')[-1])

            # Save dataframe to csv
            tb.to_csv(save_dir,index=None)
    print("Files saved to preprocessed folder.")


# Encode labels with one hot encoding and create pickle file
def encode(set, var, labeltype, raw_feature_dir):
    print("Encoding labels...")
    # all_g contains all labels of the data
    all_g =[]
    # for each task
    for sub in raw_feature_dir:
        # get list of all trials in preprocessed folder
        paths = glob.glob(os.path.join(sub,'*'))

        # for each trial
        for p in paths:
            # read in preprocessed file created above
            tb = pd.read_csv(p)

            # append all labels to all_g array
            all_g.extend(tb['Y'])

    # convert to np array
    all_g = np.array(all_g)
    #print(all_g)

    # remove 'nan' at end of labels
    g_total=[g for g in all_g if g!='nan' ]
    #print(g_total)

    # Get list of unique labels and number of classes
    #print(set(g_total))
    unique_g = np.unique(np.array(g_total))
    print("Found " + str(len(unique_g)) + " label classes: " + ' '.join(unique_g))

    # Encode labels
    le = preprocessing.LabelEncoder()
    le.fit(g_total)
    #z=le.transform(list(g_total))
    #print(list(le.classes_))

    # Save to pkl file, should be same as data_transform_path from updateJSON function
    pklFile = set + "_TRANSFORM_" + var + "_" + labeltype + ".pkl"
    pklFile = os.path.join(os.getcwd(), set, pklFile)
    print("Encoded labels written to " + pklFile)
    #with open(pklFile,'wb') as f:
        #pickle.dump(le,f)
    pickle.dump(le, open(pklFile, 'wb'))

    # Return encoded label classes
    return le




# MAIN -------------------------------------------------------------------------
if __name__ == "__main__":

    # Process arugments from command line and get set, var, and labeltype
    set, var, labeltype, valtype = processArguments(sys.argv)
    # Load model parameters from config.json
    all_params, tcn_model_params, input_size, kernel_size, num_class, raw_feature_dir,\
     test_trial, train_trial, sample_rate, gesture_class_num, validation_trial, validation_trial_train = loadConfig(set, var, labeltype, valtype)

    # Many other files look to config.json for parameters, so need to update
    # it based on set, var, and labeltype
    tcn_model_params, LOCS = updateJSON(set, var, labeltype, valtype, input_size, kernel_size, num_class)

    # Preprocess files and save into preprocessed folder
    preprocess(set, var, labeltype, raw_feature_dir)

    # Encode labels from preprocessed data files
    encode(set, var, labeltype, raw_feature_dir)

    #print(gesture_class_num)
