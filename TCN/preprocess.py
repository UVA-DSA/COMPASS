from __future__ import division
from __future__ import print_function
# Kay Hutchinson 2/4/2022
# File for preprocessing the data and save it to csv files in the
# preprocessed folder. Then, encodes labels and saves to .pkl file.

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


# Load config parameters
def loadConfig(dataset_name, var, labeltype):
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
    else:
        print("Please specify input size.")


    # Path to processed files in 'preprocessed' folder
    # raw_feature_dir contains a list of paths to the preprocessed folder of
    # the each task that will be included in the training set
    raw_feature_dir = all_params[dataset_name]["raw_feature_dir"]

    # Number of label classes
    #gesture_class_num = all_params[dataset_name]["gesture_class_num"]
    if labeltype == "MP":
        gesture_class_num = 8  # actually 6 I think...
    elif (dataset_name == "JIGSAWS") and (labeltype == "gesture"):
        gesture_class_num = 14
    elif (dataset_name == "DESK") and (labeltype == "gesture"):
        gesture_class_num = 7
    else:
        print("Please specify number of label classes.")
        sys.exit()
    num_class = gesture_class_num

    # Sets for cross validation
    test_trial=all_params[dataset_name]["test_trial"]
    train_trial = all_params[dataset_name]["train_trial"]
    sample_rate = all_params[dataset_name]["sample_rate"]

    # For parameter tuning
    validation_trial = all_params[dataset_name]["validation_trial"]
    validation_trial_train = [2,3,4,5,6]

    return all_params, tcn_model_params, input_size, num_class, raw_feature_dir,\
     test_trial, train_trial, sample_rate, gesture_class_num, validation_trial, validation_trial_train

# Modify config.json values based on command line arguements and
# processed results from loadConfig
def updateJSON(dataset_name, var, labeltype, input_size, num_class):
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

    # Update tcn params
    all_params[dataset_name]["tcn_params"]["model_params"]["class_num"] = num_class
    all_params[dataset_name]["tcn_params"]["model_params"]["encoder_params"]["input_size"] = input_size

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
    else:
        LOCS=[  "PSML_position_x", "PSML_position_y", "PSML_position_z",\
                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w",\
                "PSML_gripper_angle", \
                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w",\
                "PSMR_gripper_angle"]
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
        if labeltype == "MP":
            ges_dir_all=glob.glob(os.path.join("/".join(sub.split('/')[0:-1]),"motion_primitives/*"))
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
            if labeltype == "MP":
                tg = pd.read_table(ges_dir)
            elif labeltype == "gesture":
                tg = pd.read_table(ges_dir, sep="\s+", header=None)
            #print(tg)

            # Get kin file associated with transcript
            if labeltype == "MP":
                kin_dir = ges_dir.replace("motion_primitives", "velkinematics").replace(".txt", ".csv")
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
            # else take all vars
            else:
                tb = tb.loc[:, ["PSML_position_x", "PSML_position_y", "PSML_position_z",\
                                "PSML_velocity_x","PSML_velocity_y","PSML_velocity_z", \
                                "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w",\
                                "PSML_gripper_angle", \
                                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",\
                                "PSMR_velocity_x","PSMR_velocity_y","PSMR_velocity_z", \
                                "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w",\
                                "PSMR_gripper_angle"]]

            # For each line in the label transcript, get start frame, end frame,
            # and gesture label
            for i in range(tg.shape[0]):
                if labeltype == "MP":
                    line = tg.iloc[i,0].split(" ")
                    start_ = int(line[0])  #int(tg.iloc[i,0])  #line[0])
                    end_ = int(line[1])   #int(tg.iloc[i,1])   #line[1])
                    # Only use MP as label, no context
                    label = line[2].split("(")[0]   #tg.iloc[i,2].split("(")[0]  #line[2].split("(")
                elif (set == "DESK") and (labeltype == "gesture"):
                    start_ = int(tg.iloc[i,0])
                    end_ = int(tg.iloc[i,1])
                    label = tg.iloc[i,2]
                    #gesture = label
                elif set == "JIGSAWS":
                    start_ = int(tg.iloc[i,0])
                    end_ = int(tg.iloc[i,1])
                    label = tg.iloc[i,2]

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

    # Save to pkl file, should be same as data_transform_path from updateJSON function
    pklFile = set + "_TRANSFORM_" + var + "_" + labeltype + ".pkl"
    pklFile = os.path.join(os.getcwd(), set, pklFile)
    print("Encoded labels written to " + pklFile)
    with open(pklFile,'wb') as f:
        pickle.dump(le,f)




# MAIN -------------------------------------------------------------------------
if __name__ == "__main__":

    # Process arugments from command line and get set, var, and labeltype
    set, var, labeltype = processArguments(sys.argv)
    # Load model parameters from config.json
    all_params, tcn_model_params, input_size, num_class, raw_feature_dir,\
     test_trial, train_trial, sample_rate, gesture_class_num, validation_trial, validation_trial_train = loadConfig(set, var, labeltype)

    # Many other files look to config.json for parameters, so need to update
    # it based on set, var, and labeltype
    tcn_model_params, LOCS = updateJSON(set, var, labeltype, input_size, num_class)

    # Preprocess files and save into preprocessed folder
    preprocess(set, var, labeltype, raw_feature_dir)

    # Encode labels from preprocessed data files
    encode(set, var, labeltype, raw_feature_dir)

    #print(gesture_class_num)

    # Location of dataset folder
    dir=os.getcwd()
