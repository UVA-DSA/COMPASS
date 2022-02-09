from __future__ import division
from __future__ import print_function

# 2/4/2022 Trying to automate training so we have to change as few vars
# as possible when changing set, var, and labeltype

import os
import sys
import glob
import numpy as np
from ray.tune.session import checkpoint_dir
import torch
import torch.nn as nn
from random import randrange
from data_loading import RawFeatureDataset
from lstm_model import LSTM_Layer
import pandas as pd
from logger import Logger
import utils
import pdb
import json

from calculate_mean_cv import analyze

# for parameter tuning LSTM/TCN
from utils import get_cross_val_splits
import ray
from ray import tune
from tcn_model import EncoderDecoderNet

# Import from config, should be consistent with updates from preprocess.py
from config import tcn_model_params, dataset_name, gesture_class_num

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



def train_model(config,type,train_dataset,val_dataset,input_size, num_class,num_epochs,
                loss_weights=None,
                trained_model_file=None,
                log_dir=None, checkpoint_dir=None):

    if type =='lstm':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTM_Layer(input_size=input_size, num_class=num_class,hidden_size=config["hidden_size"], num_layers=config["num_layers"],device=device)
        model.to(device)
    if type =='tcn':
        model = EncoderDecoderNet(**tcn_model_params['model_params'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)


    df = pd.DataFrame(index=np.arange(0,num_epochs),  columns=('t_accuracy','t_edit_score','t_loss','t_f_scores_10','t_f_scores_25','t_f_scores_50','t_f_scores_75',\
    'v_accuracy','v_edit_score','v_loss','v_f_scores_10','v_f_scores_25','v_f_scores_50','v_f_scores_75'))
        #breakpoint()
    loss_weights = utils.get_class_weights(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=config["batch_size"], shuffle=True)


    model.train()

    if loss_weights is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
                        weight=torch.Tensor(loss_weights).to(device), #.cuda()
                        ignore_index=-1)



    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"],
                                            weight_decay=config["weight_decay"])

    # if checkpoint_dir:
    #     model_state, optimizer_state = torch.load(
    #         os.path.join(checkpoint_dir, "checkpoint"))
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    step = 1
    for epoch in range(num_epochs):
        print(epoch)
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader):

            feature = data['feature'].float()
            feature = feature.to(device) #.cuda()

            gesture = data['gesture'].long()
            gesture = gesture.view(-1)
            gesture = gesture.to(device) #.cuda()

            # Forward
            out = model(feature)
            flatten_out = out.view(-1, out.shape[-1])
            #breakpoint()

            loss = criterion(input=flatten_out, target=gesture)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if i % 10 == 9:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

            # Logging
            #if log_dir is not None:
                #logger.scalar_summary('loss', loss.item(), step)

            step += 1

        if trained_model_file is not None:
            if not os.path.exists(trained_model_file):
                os.makedirs(trained_model_file, exist_ok=True)
            file_dir = os.path.join(trained_model_file,"checkpoint_{}.pth".format(epoch))
            torch.save(model.state_dict(), file_dir)

        if log_dir is not None:
            if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
            train_result = test_model(model, train_dataset, loss_weights)
            t_accuracy, t_edit_score, t_loss, t_f_scores = train_result

            val_result = test_model(model, val_dataset, loss_weights)
            v_accuracy, v_edit_score, v_loss, v_f_scores = val_result
            df.loc[epoch] = [t_accuracy, t_edit_score,t_loss, t_f_scores[0], t_f_scores[1], t_f_scores[2], t_f_scores[3],\
                v_accuracy, v_edit_score,v_loss, v_f_scores[0], v_f_scores[1], v_f_scores[2], v_f_scores[3]]
    df.to_csv(os.path.join(log_dir,'train_test_result.csv'))



def train_model_parameter( config, type,input_size, num_class,num_epochs,dataset_name,sample_rate,
                loss_weights=None,
                trained_model_file=None,
                log_dir=None, checkpoint_dir=None):

    if type =='lstm':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTM_Layer(input_size=input_size, num_class=num_class,hidden_size=config["hidden_size"], num_layers=config["num_layers"],device=device)
        model.to(device)
    if type =='tcn':
        model = EncoderDecoderNet(**tcn_model_params['model_params'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    paths = get_cross_val_splits(validation = True)

    train_trail_list = paths["train"]
    test_trail_list = paths["test"]
    train_dataset = RawFeatureDataset(dataset_name,
                                        train_trail_list,
                                        feature_type="sensor",
                                        sample_rate=sample_rate,
                                        sample_aug=False,
                                        normalization=[None, None])
    #breakpoint()

    test_norm = [train_dataset.get_means(), train_dataset.get_stds()]
    val_dataset = RawFeatureDataset(dataset_name,
                                        test_trail_list,
                                        feature_type="sensor",
                                        sample_rate=sample_rate,
                                        sample_aug=False,
                                        normalization=test_norm)

    loss_weights = utils.get_class_weights(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=config["batch_size"], shuffle=True)


    model.train()

    if loss_weights is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
                        weight=torch.Tensor(loss_weights).to(device), #.cuda()
                        ignore_index=-1)

    # Logger
    if log_dir is not None:
        logger = Logger(log_dir)

    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"],
                                            weight_decay=config["weight_decay"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    step = 1
    for epoch in range(num_epochs):
        print(epoch)
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader):

            feature = data['feature'].float()
            feature = feature.to(device) #.cuda()

            gesture = data['gesture'].long()
            gesture = gesture.view(-1)
            gesture = gesture.to(device) #.cuda()
            #print(feature.shape)
            #print(gesture.shape)

            # Forward
            out = model(feature)
           # print(out.shape)
            flatten_out = out.view(-1, out.shape[-1])
           # print(flatten_out.shape)

            #breakpoint()

            loss = criterion(input=flatten_out, target=gesture)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if i % 10 == 9:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

            # Logging
            if log_dir is not None:
                logger.scalar_summary('loss', loss.item(), step)

            step += 1

        train_result = test_model(model, train_dataset, loss_weights)
        t_accuracy, t_edit_score, t_loss, t_f_scores = train_result

        val_result = test_model(model, val_dataset, loss_weights)
        v_accuracy, v_edit_score, v_loss, v_f_scores = val_result

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            if epoch ==num_epochs -1:
                path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(epoch))
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=v_loss , accuracy=v_accuracy,edit_score=v_edit_score,F1=v_f_scores)
        print("Finished Training")
        if log_dir is not None:
            train_result = test_model(model, train_dataset, loss_weights)
            t_accuracy, t_edit_score, t_loss, t_f_scores = train_result

            val_result = test_model(model, val_dataset, loss_weights)
            v_accuracy, v_edit_score, v_loss, v_f_scores = val_result

            logger.scalar_summary('t_accuracy', t_accuracy, epoch)
            logger.scalar_summary('t_edit_score', t_edit_score, epoch)
            logger.scalar_summary('t_loss', t_loss, epoch)
            logger.scalar_summary('t_f_scores_10', t_f_scores[0], epoch)
            logger.scalar_summary('t_f_scores_25', t_f_scores[1], epoch)
            logger.scalar_summary('t_f_scores_50', t_f_scores[2], epoch)
            logger.scalar_summary('t_f_scores_75', t_f_scores[3], epoch)

            logger.scalar_summary('v_accuracy', v_accuracy, epoch)
            logger.scalar_summary('v_edit_score', v_edit_score, epoch)
            logger.scalar_summary('v_loss', v_loss, epoch)
            logger.scalar_summary('v_f_scores_10', v_f_scores[0], epoch)
            logger.scalar_summary('v_f_scores_25', v_f_scores[1], epoch)
            logger.scalar_summary('v_f_scores_50', v_f_scores[2], epoch)
            logger.scalar_summary('v_f_scores_75', v_f_scores[3], epoch)

        if trained_model_file is not None:
            torch.save(model.state_dict(), trained_model_file)



def test_model(model, test_dataset, loss_weights=None, plot_naming=None):

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    if loss_weights is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
                        weight=torch.Tensor(loss_weights).to(device), #.cuda()
                        ignore_index=-1)

    #Test the Model
    total_loss = 0
    preditions = []
    gts=[]

    with torch.no_grad():
        for i, data in enumerate(test_loader):

            feature = data['feature'].float()
            feature = feature.to(device) #.cuda()

            gesture = data['gesture'].long()
            gesture = gesture.view(-1)
            gesture = gesture.to(device) #.cuda()

            # Forward
            out = model(feature)
            out = out.squeeze(0)

            loss = criterion(input=out, target=gesture)

            total_loss += loss.item()

            pred = out.data.max(1)[1]

            trail_len = (gesture.data.cpu().numpy()!=-1).sum()
            gesture = gesture[:trail_len]
            pred = pred[:trail_len]

            preditions.append(pred.cpu().numpy())
            gts.append(gesture.data.cpu().numpy())

            # Plot   [Errno 2] No such file or directory: './graph/JIGSAWS/Suturing/sensor_run_1_split_1_seq_0.png' 12/7/21
            # if plot_naming:
            #     graph_file = os.path.join(graph_dir, '{}_seq_{}'.format(
            #                                     plot_naming, str(i)))

            #     utils.plot_barcode(gt=gesture.data.cpu().numpy(),
            #                        pred=pred.cpu().numpy(),
            #                        visited_pos=None,
            #                        show=False, save_file=graph_file)

    bg_class = 0 if dataset_name != 'JIGSAWS' else None

    avg_loss = total_loss / len(test_loader.dataset)
    edit_score = utils.get_edit_score_colin(preditions, gts,
                                            bg_class=bg_class)
    accuracy = utils.get_accuracy_colin(preditions, gts)
    #accuracy = utils.get_accuracy(preditions, gts)

    f_scores = []
    for overlap in [0.1, 0.25, 0.5, 0.75]:
        f_scores.append(utils.get_overlap_f1_colin(preditions, gts,
                                        n_classes=gesture_class_num,
                                        bg_class=bg_class,
                                        overlap=overlap))

    model.train()
    return accuracy, edit_score, avg_loss, f_scores






######################### Main Process #########################
def cross_validate(dataset_name,net_name):
    '''

    '''
    # Update after running parameter tuning
    if net_name =='tcn':
        num_epochs = 30 # about 25 mins for 5 fold cross validation
        #config = {'learning_rate': 0.0003042861945575232, 'batch_size': 1, 'weight_decay': 0.00012035748692105724} #EPOCH=30 tcn
        # DESK MPs best config
        #config = {'learning_rate': 0.000303750997737948, 'batch_size': 1, 'weight_decay': 0.0003482923872868488}
        # DESK surgemes best confg
        config = {'learning_rate': 3.963963013042929e-05, 'batch_size': 1, 'weight_decay': 0.00027159256985286403}
    if net_name=='lstm':
        num_epochs = 60
        config =  {'hidden_size': 128 , 'learning_rate': 0.000145129 ,  'num_layers': 3 ,'batch_size': 1, 'weight_decay':0.00106176 } # Epoch =60 lstm



    # Get trial splits
    print("Getting cross validation splits")
    cross_val_splits = utils.get_cross_val_splits()
    #breakpoint()


    # Cross-Validation Result
    #result = []

    # Cross Validation
    for idx, data in enumerate(cross_val_splits):
        #breakpoint()
        # Dataset
        train_dir, test_dir,name = data['train'], data['test'],data['name']
        print("Loading training data")

        # Useful debugging note:
        #print("If KeyError: ['PSML_var', ...]... occurs here, it may be because the data_loading.py file doesn't have the updated LOCS.")
        #print("Try running train_test_val.py again with the same set, var, and labeltype configuration.")

        train_dataset = RawFeatureDataset(dataset_name,
                                        train_dir,
                                        feature_type="sensor",
                                        sample_rate=sample_rate,
                                        sample_aug=False,
                                        normalization=[None, None])
        #breakpoint()
        print("Normalizing training data")
        test_norm = [train_dataset.get_means(), train_dataset.get_stds()]
        test_dataset = RawFeatureDataset(dataset_name,
                                         test_dir,
                                         feature_type="sensor",
                                         sample_rate=sample_rate,
                                         sample_aug=False,
                                         normalization=test_norm)

        print("Getting loss weights")
        loss_weights = utils.get_class_weights(train_dataset)
        # make directories
        path = os.getcwd()
        trained_model_dir=  os.path.join(path,dataset_name,net_name,name) # contain name of the testing set
        os.makedirs(trained_model_dir, exist_ok=True)
        log_dir = os.path.join(trained_model_dir,'log')
        checkpoint_dir = os.path.join(trained_model_dir,'checkpoints')

        print("Training model")
        train_model(config,net_name,train_dataset,test_dataset,input_size, num_class,num_epochs,
                loss_weights=loss_weights,
                trained_model_file=trained_model_dir,
                log_dir=log_dir, checkpoint_dir=checkpoint_dir)

        #acc, edit, _, f_scores = test_model(model, test_dataset,
                                         #   loss_weights=loss_weights)


# MAIN -------------------------------------------------------------------------
if __name__ == "__main__":

    # Process arugments from command line and get set, var, and labeltype
    set, var, labeltype = processArguments(sys.argv)

    # Some functions require dataset_name, which returned as set from processArguments
    global dataset_name
    dataset_name = set

    # Load model parameters from config.json
    all_params, tcn_model_params, input_size, num_class, raw_feature_dir,\
     test_trial, train_trial, sample_rate, gesture_class_num, validation_trial, validation_trial_train = loadConfig(set, var, labeltype)

    # Many other files look to config.json for parameters, so need to update
    # it based on set, var, and labeltype (should now be redundant with the
    # same updates from the same function, but run in the preprocess.py script)
    tcn_model_params, LOCS = updateJSON(set, var, labeltype, input_size, num_class)



    # Train, test, and cross validate
    cross_validate(set,'tcn') #, sample_rate, input_size, num_class)

    # Call analyze() from calculate_mean_cv and print results
    #analyze()  doesn't work yet b/c hard coded file paths!
