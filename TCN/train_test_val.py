from __future__ import division
from __future__ import print_function

# 2/4/2022 Trying to automate training so we have to change as few vars
# as possible when changing set, var, and labeltype

import os
import sys
import glob
import time
from datetime import datetime
import numpy as np
from ray.tune.trainable.session import checkpoint_dir
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
import shutil
import pdb
import pickle

#from calculate_mean_cv import analyze

# for parameter tuning LSTM/TCN
from utils import get_cross_val_splits, get_cross_val_splits_LOUO, get_cross_val_splits_LOUO_multi
import ray
from ray import tune
from tcn_model import EncoderDecoderNet

# Import from config, should be consistent with updates from preprocess.py
from config import tcn_model_params, dataset_name, gesture_class_num, data_transform_path, val_type
from preprocess import processArguments, loadConfig, updateJSON, preprocess, encode


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
            #torch.save(model.state_dict(), file_dir)  # 7/22/22 turned off saving to save memory

        if log_dir is not None:
            if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
            train_result = test_model(model, train_dataset, loss_weights,name = "train",log_dir =log_dir )
            t_accuracy, t_edit_score, t_loss, t_f_scores = train_result

            val_result = test_model(model, val_dataset, loss_weights,name="test",log_dir =log_dir, epoch=epoch)
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

    # Get paths for LOSO or LOUO val setup
    if val_type == "LOSO":
        paths = get_cross_val_splits(validation = True)
    elif val_type == "LOUO":
        paths = get_cross_val_splits_LOUO(validation=True)

    train_trail_list = paths["train"]
    test_trail_list = paths["test"]


    import fnmatch
    if  dataset_name=="All-5a":
        test_dir_5a=[dir  for dir in test_trail_list if fnmatch.fnmatch(dir,"*[Suturing,Needle_Passing,Knot_Tying]*")]

        other_train = [dir  for dir in test_trail_list if not fnmatch.fnmatch(dir,"*[Suturing,Needle_Passing,Knot_Tying]*")]
        train_trail_list.extend(other_train)
        test_trail_list = test_dir_5a

    if  dataset_name=="All-5b":
        test_dir_5b = [dir  for dir in test_trail_list if fnmatch.fnmatch(dir,"*Peg_Transfer*")]
        other_train = [dir  for dir in test_trail_list if not fnmatch.fnmatch(dir,"*Peg_Transfer*")]
        train_trail_list.extend(other_train)
        test_trail_list = test_dir_5b

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



def test_model(model, test_dataset, loss_weights=None, log_dir =None, name = 'default',plot_naming=None, epoch='default'):

    if log_dir!=None:
        test_data_file = 'tcn_{}.npy'.format(name)

        np.save(os.path.join(log_dir, test_data_file),
                                            test_dataset)

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

    #import json,pickle
    #all_params = json.load(open('config.json'))
    transform_path = data_transform_path  #all_params[dataset_name]["data_transform_path"]
    with open(transform_path, 'rb') as f:
            label_transform =  pickle.load(f)
            #print(list(label_transform.classes_))
    #Test the Model
    total_loss = 0
    preditions = []
    gts=[]


    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # Access name of kinematic file and pull out the trial name (task, subject, and trial number)
            trialName = data['name'][0].split("/")[-1].split(".")[0]
            naming=data['name'][0].split("/")[-1][:-4]
            #print(trialName)

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
            if log_dir!=None and name !='train':
                model_conv_pred = label_transform.inverse_transform(pred.cpu().numpy())
            #breakpoint()

                model_conv_gt = label_transform.inverse_transform(gesture.data.cpu().numpy())
                test_data_naming_pred_gt = '{}_{}_pred_gt.npy'.format(name,naming)
                #np.save(os.path.join(log_dir, test_data_naming_pred_gt), [data['feature'][:,:trail_len,:].float(),model_conv_pred,model_conv_gt])
                # Updated saving code
                if not os.path.exists(os.path.join(log_dir, 'epoch_{}'.format(epoch))):
                    os.makedirs(os.path.join(log_dir, 'epoch_{}'.format(epoch)), exist_ok=True)
                np.save(os.path.join(log_dir, 'epoch_{}'.format(epoch),test_data_naming_pred_gt), [data['feature'][:,:trail_len,:].float(),model_conv_pred,model_conv_gt])

            # Call inverse_transform on the preditions to get the original labels
            #print(np.shape(preditions))
            #predictions = list(le.inverse_transform(np.transpose(preditions)))
            #print(predictions)

            #sys.exit()

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
def cross_validate(dataset_name,net_name, logDir):
    '''
    '''

    # Load num_epochs, learning_rate, batch_size, and weight_decay from config.json
    numepochs = tcn_model_params["config"]["epoch"]
    learningrate = tcn_model_params["config"]["learning_rate"]
    batchsize = tcn_model_params["config"]["batch_size"]
    weightdecay = tcn_model_params["config"]["weight_decay"]

    print("Config parameters:")
    print("num_epochs: "+ str(numepochs))
    #print(learningrate)
    #print(type(learningrate))
    print("learning_rate: " + str(learningrate))
    print("batch_size: " + str(batchsize))
    print("weight_decay: " + str(weightdecay))

    #print(type(learningrate))
    #print(type(batchsize))

    # Update after running parameter tuning
    if net_name =='tcn':
        num_epochs = numepochs # about 25 mins for 5 fold cross validation
        config = {'learning_rate': learningrate, 'batch_size': batchsize, 'weight_decay': weightdecay}

    if net_name=='lstm':
        num_epochs = 60
        config =  {'hidden_size': 128 , 'learning_rate': 0.000145129 ,  'num_layers': 3 ,'batch_size': 1, 'weight_decay':0.00106176 } # Epoch =60 lstm

    

    # Get trial splits
    print("Getting cross validation splits")
    if valtype == "LOSO":
        cross_val_splits = utils.get_cross_val_splits()
        print("cross_val_splits", cross_val_splits)
    elif (valtype == "LOUO") and (dataset_name == "PTPaS"):
        cross_val_splits = utils.get_cross_val_splits_LOUO_multi()
    elif (valtype == "LOUO") and (dataset_name == "SNP" or dataset_name == "JIGSAWS" or dataset_name == "ROSMA"):
        cross_val_splits = utils.get_cross_val_splits_LOUO()
    elif (valtype == "LOUO") and (dataset_name == "All"):
        cross_val_splits = utils.get_cross_val_splits_LOUO_all()
    elif (valtype == "LOTO") and (dataset_name in ["SNP", "JIGSAWS", "ROSMA", "PTPaS", "All"]):
        #print("Using LOTO cross validation folds")
        cross_val_splits = utils.get_cross_val_splits_LOTO()
    elif valtype == "random":
        print("Using 5 random fold cross valiation")
        cross_val_splits = utils.get_cross_val_splits_random()

    # cross_val_splits = utils.get_cross_val_splits_LOUO() #utils.get_cross_val_splits()
    #breakpoint()

    # Cross-Validation Result
    #result = []

    # Cross Validation
    for idx, data in enumerate(cross_val_splits):
        #breakpoint()
        # Dataset
        train_dir, test_dir,name = data['train'], data['test'],data['name']
        print("Loading training data")
        print("test_dir right after loading", test_dir)

        import fnmatch
        if  dataset_name=="All-5a":
            test_dir_5a=[dir  for dir in test_dir if fnmatch.fnmatch(dir,"*[Suturing,Needle_Passing,Knot_Tying]*")]

            other_train = [dir  for dir in test_dir if not fnmatch.fnmatch(dir,"*[Suturing,Needle_Passing,Knot_Tying]*")]
            train_dir.extend(other_train)
            test_dir = test_dir_5a

        if  dataset_name=="All-5b":
            test_dir_5b = [dir  for dir in test_dir if fnmatch.fnmatch(dir,"*Peg_Transfer*")]
            other_train = [dir  for dir in test_dir if not fnmatch.fnmatch(dir,"*Peg_Transfer*")]
            train_dir.extend(other_train)
            test_dir = test_dir_5b

        # Useful debugging note:
        #print("If KeyError: ['PSML_var', ...]... occurs here, it may be because the data_loading.py file doesn't have the updated LOCS.")
        #print("Try running train_test_val.py again with the same set, var, and labeltype configuration.")

        print("train_dir", train_dir)
        train_dataset = RawFeatureDataset(dataset_name,
                                        train_dir,
                                        feature_type="sensor",
                                        sample_rate=sample_rate,
                                        sample_aug=False,
                                        normalization=[None, None])
        #breakpoint()
        print("Normalizing training data")
        test_norm = [train_dataset.get_means(), train_dataset.get_stds()]
        # print("test_norm:", test_norm)
        # print("test_dir", test_dir)
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
        trained_model_dir=  os.path.join(logDir, net_name, name)  #os.path.join(path,dataset_name,net_name,name) # contain name of the testing set
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
    global valtype, dataset_name
    # Process arugments from command line and get set, var, and labeltype
    set, var, labeltype, valtype = processArguments(sys.argv)

    # Some functions require dataset_name, which returned as set from processArguments
    dataset_name = set

    # Load model parameters from config.json
    all_params, tcn_model_params, input_size, kernel_size, num_class, raw_feature_dir,\
     test_trial, train_trial, sample_rate, gesture_class_num, validation_trial, validation_trial_train = loadConfig(set, var, labeltype, valtype)

    # Many other files look to config.json for parameters, so need to update
    # it based on set, var, and labeltype (should now be redundant with the
    # same updates from the same function, but run in the preprocess.py script)
    #tcn_model_params, LOCS = updateJSON(set, var, labeltype, valtype, input_size, kernel_size, num_class)


    # Create folder for results
    dir = os.getcwd()
    resultsDir = os.path.join(dir, "Results")
    # Create folder named by current time and config
    now = datetime.now()
    timeNow = now.strftime("%m_%d_%Y_%H%M")
    logFolder = set +"_"+ var +"_"+ labeltype +"_"+ valtype +"_"+ timeNow
    logDir =  os.path.join(resultsDir, logFolder)
    if not os.path.exists(logDir):
        os.mkdir(logDir)

    print("Results will be stored in: " + logDir)
    # Copy config file over first
    # path to config file
    configPath = os.path.join(dir, "config.json")
    shutil.copy2(configPath, logDir)


    # Encode labels again and get the encoder so inverse_transform can be used
    # Encode labels from preprocessed data files
    le = encode(set, var, labeltype, raw_feature_dir)
    print(list(le.classes_))

    # Train, test, and cross validate
    cross_validate(set,'tcn', logDir) #'tcn') #, sample_rate, input_size, num_class)

    # Call analyze() from calculate_mean_cv and print results
    #analyze()  doesn't work yet b/c hard coded file paths!
