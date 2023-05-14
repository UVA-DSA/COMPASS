from __future__ import division
from __future__ import print_function
from curses import raw

import os
import torch
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('AGG')    # Show Plot Disabled
import matplotlib.pyplot as plt
import random
import string
import glob
import sys

import pdb

################## Random ##########################

def set_global_seeds(seed, use_cudnn=True):
    torch.backends.cudnn.enabled = use_cudnn   # Too slow
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

def generate_random_str(size, chs=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chs) for _ in range(size))

################### Trail List ######################

def get_all_trail():
    from config import split_info_dir
    all_trail_file = os.path.join(split_info_dir, 'all.txt')

    with open(all_trail_file) as file:
        trail_list = file.readlines()
    trail_list = [t.strip() for t in trail_list]

    return trail_list


# Random k-fold cross validation
def get_cross_val_splits_random(k=5):
    import random
    def partition(list, n):
        random.shuffle(list)
        return [list[i::n] for i in range(n)]

    # load from config
    from config import raw_feature_dir, validation_trial, validation_trial_train, test_trial, train_trial
    cross_val_splits = []
    all_files = []
    for i in raw_feature_dir:
        all_files.extend(glob.glob(os.path.join(i, "*")))

    test_folds = partition(all_files, k)

    for i in range(k):
        test_dir = test_folds[i]
        train_dir = [t for t in all_files if t not in test_dir]
        cross_val_splits.append({'train': train_dir,
                                'test': test_dir,
                                'name': 'test_{}'.format(i)})
    return cross_val_splits



# LOTO for evaluating models trained with data from multiple tasks where each
# fold is a different task
def get_cross_val_splits_LOTO(validation = False):
    # load from config
    from config import raw_feature_dir, validation_trial, validation_trial_train, test_trial, train_trial
    # for validation set and hyperparameter tuning
    if validation ==True:
        print("Exiting... LOTO validation sets not set up")
        sys.exit()
        cross_val_splits=[]
        test_dir = []
        train_dir = []
        if len(raw_feature_dir)!=1:
            for i in raw_feature_dir:
                print(os.path.join(i,'*{}_*'.format(validation_trial)))
                #test = glob.glob(os.path.join(i,'*{}_*'.format(validation_trial)))
                test = glob.glob(os.path.join(i,'*S'+test_num+'_*'))#.format(test_num)))
                test_dir.extend(test)

                train = glob.glob(os.path.join(i, "*"))
                train = [t for t in train if t not in test]
                train_dir.extend(train)

            return {'train':train_dir,'test':test_dir,'name':'tune'}
        else:
            i = raw_feature_dir[0]
            print(os.path.join(i,'*{}_*'.format(validation_trial)))
            #test = glob.glob(os.path.join(i,'*{}_*'.format(validation_trial)))
            test = glob.glob(os.path.join(i,'*S'+test_num+'_*'))#.format(test_num)))
            test_dir.extend(test)

            train = glob.glob(os.path.join(i, "*"))
            train = [t for t in train if t not in test]
            train_dir.extend(train)

            return {'train':train_dir,'test':test_dir,'name':'tune'}


    # for training
    else:
        cross_val_splits = []
        # for each fold (each task_subject combination)
        for idx, test_num in enumerate(test_trial):
            train_dir = []
            test_dir = []
            if len(raw_feature_dir)!=1:
                for i in raw_feature_dir:
                    # list files for testing
                    #test = glob.glob(os.path.join(i,'{}_*'.format(test_num)))
                    test = glob.glob(os.path.join(i,test_num+'_S*'))#.format(test_num)))
                    test_dir.extend(test)

                    # list all other files minus test files
                    train = glob.glob(os.path.join(i, "*"))
                    train = [t for t in train if t not in test]
                    train_dir.extend(train)

            else:
                i = raw_feature_dir[0]
                #test = glob.glob(os.path.join(i,'*{}_*'.format(test_num)))
                test = glob.glob(os.path.join(i,test_num+'_S*'))#.format(test_num)))
                test_dir.extend(test)

                train = glob.glob(os.path.join(i, "*"))
                train = [t for t in train if t not in test]
                train_dir.extend(train)

            #breakpoint()
            # add fold sets to cross_val_splits
            cross_val_splits.append({'train': train_dir,
                                    'test': test_dir,
                                    'name': 'test_{}'.format(test_num)})

        return cross_val_splits






# LOUO for evaluating model with data from all tasks where each
# fold is a different subject and considering that datasets have subjects
# that performed different tasks
def get_cross_val_splits_LOUO_all(validation = False):
    # load from config
    from config import raw_feature_dir, validation_trial, validation_trial_train, test_trial, train_trial
    # for validation set and hyperparameter tuning
    if validation ==True:
        print("Exiting... not folds not defined yet")
        sys.exit()
        cross_val_splits=[]
        test_dir = []
        train_dir = []
        if len(raw_feature_dir)!=1:
            for i in raw_feature_dir:
                print(os.path.join(i,'*{}_*'.format(validation_trial)))
                #test = glob.glob(os.path.join(i,'*{}_*'.format(validation_trial)))
                test = glob.glob(os.path.join(i,'*S'+test_num+'_*'))#.format(test_num)))
                test_dir.extend(test)
                # tri=[j for j in validation_trial_train]
                # a = "*["+",".join(['{}']*len(validation_trial_train))+"]_*"
                # train = glob.glob(os.path.join(i,a.format(*tri)))
                #     #train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
                #      #   train_trial[idx][2],train_trial[idx][3])))
                #
                # print(os.path.join(i,a.format(*tri)))
                # train_dir.extend(train)
                train = glob.glob(os.path.join(i, "*"))
                train = [t for t in train if t not in test]
                train_dir.extend(train)

            return {'train':train_dir,'test':test_dir,'name':'tune'}
        else:
            i = raw_feature_dir[0]
            print(os.path.join(i,'*{}_*'.format(validation_trial)))
            #test = glob.glob(os.path.join(i,'*{}_*'.format(validation_trial)))
            test = glob.glob(os.path.join(i,'*S'+test_num+'_*'))#.format(test_num)))
            test_dir.extend(test)
            # tri=[j for j in validation_trial_train]
            # a = "*["+",".join(['{}']*len(validation_trial_train))+"]_*"
            # train = glob.glob(os.path.join(i,a.format(*tri)))
            #         #train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
            #          #   train_trial[idx][2],train_trial[idx][3])))
            # print(os.path.join(i,a.format(*tri)))
            # #breakpoint()
            # train_dir.extend(train)
            train = glob.glob(os.path.join(i, "*"))
            train = [t for t in train if t not in test]
            train_dir.extend(train)

            return {'train':train_dir,'test':test_dir,'name':'tune'}


    # for training
    else:
        cross_val_splits = []

        # list of all trials
        all_dir = []
        for i in raw_feature_dir:
            task_trials = glob.glob(os.path.join(i, "*"))
            all_dir.extend(task_trials)

        # get path to datasets
        data_dir = os.path.dirname(os.path.dirname(i))


        # Tasks in each dataset:
        JIGSAWS_tasks = ["Suturing", "Needle_Passing", "Knot_Tying"]
        DESK_tasks = ["Peg_Transfer"]
        ROSMA_tasks = ["Post_and_Sleeve", "Pea_on_a_Peg"]

        # Subjects for each dataset:
        JIGSAWS_subjects = ["02","03","04","05","06","07","08","09"]
        DESK_subjects = ["01","02","03","04","05","06","07","08"]
        ROSMA_subjects = ["01","02","03","04","05","06","07","08","09","10","11","12"]

        # create each fold based on dataset and subject combination
        # starting with JIGSAWS
        for subject in JIGSAWS_subjects:
            test_dir = []
            train_dir = []
            # for each task, look for the task_subject files and create folds
            for task in JIGSAWS_tasks:
                #print(data_dir, task+"_S"+subject+'_*')
                test = glob.glob(os.path.join(data_dir, task, "preprocessed", task+"_S"+subject+'_*'))
                test_dir.extend(test)
            # remove test files from all_dir to get the train files
            train = [t for t in all_dir if t not in test_dir]
            train_dir.extend(train)
            # add fold to cross_val_splits
            cross_val_splits.append({'train': train_dir,
                                    'test': test_dir,
                                    'name': 'test_{}'.format("JIGSAWS_"+subject)})

        # for DESK
        for subject in DESK_subjects:
            test_dir = []
            train_dir = []
            # for each task, look for the task_subject files and create folds
            for task in DESK_tasks:
                #print(data_dir, task+"_S"+subject+'_*')
                test = glob.glob(os.path.join(data_dir, task, "preprocessed", task+"_S"+subject+'_*'))
                test_dir.extend(test)
            # remove test files from all_dir to get the train files
            train = [t for t in all_dir if t not in test_dir]
            train_dir.extend(train)
            # add fold to cross_val_splits
            cross_val_splits.append({'train': train_dir,
                                    'test': test_dir,
                                    'name': 'test_{}'.format("DESK_"+subject)})

        # for ROSMA
        for subject in ROSMA_subjects:
            test_dir = []
            train_dir = []
            # for each task, look for the task_subject files and create folds
            for task in ROSMA_tasks:
                #print(data_dir, task+"_S"+subject+'_*')
                test = glob.glob(os.path.join(data_dir, task, "preprocessed", task+"_S"+subject+'_*'))
                test_dir.extend(test)
            # remove test files from all_dir to get the train files
            train = [t for t in all_dir if t not in test_dir]
            train_dir.extend(train)
            # add fold to cross_val_splits
            cross_val_splits.append({'train': train_dir,
                                    'test': test_dir,
                                    'name': 'test_{}'.format("ROSMA_"+subject)})

        '''
        # Check that folds are correct by printing
        for fold in range(len(cross_val_splits)):
            print(cross_val_splits[fold]["name"])
            tests = cross_val_splits[fold]["test"]
            # for k in range(len(tests)):
            #     print(tests[k].split("/")[-1])
            # trains = cross_val_splits[fold]["train"]
            # for k in range(len(trains)):
            #     print(trains[k].split("/")[-1])
            print(len(cross_val_splits[fold]["test"]))
            print(len(cross_val_splits[fold]["train"]))
            print(str(int(len(cross_val_splits[fold]["test"])) + int(len(cross_val_splits[fold]["train"]))))
        sys.exit()
        '''


        return cross_val_splits




# LOUO for evaluating models trained with data from multiple tasks where each
# fold is a different task_subject (instead of just different subject)
def get_cross_val_splits_LOUO_multi(validation = False):
    # load from config
    from config import raw_feature_dir, validation_trial, validation_trial_train, test_trial, train_trial
    # for validation set and hyperparameter tuning
    if validation ==True:
        print("Exiting... not sure if LOUO_multi validation sets are correct")
        sys.exit()
        cross_val_splits=[]
        test_dir = []
        train_dir = []
        if len(raw_feature_dir)!=1:
            for i in raw_feature_dir:
                print(os.path.join(i,'*{}_*'.format(validation_trial)))
                #test = glob.glob(os.path.join(i,'*{}_*'.format(validation_trial)))
                test = glob.glob(os.path.join(i,'*S'+test_num+'_*'))#.format(test_num)))
                test_dir.extend(test)
                # tri=[j for j in validation_trial_train]
                # a = "*["+",".join(['{}']*len(validation_trial_train))+"]_*"
                # train = glob.glob(os.path.join(i,a.format(*tri)))
                #     #train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
                #      #   train_trial[idx][2],train_trial[idx][3])))
                #
                # print(os.path.join(i,a.format(*tri)))
                # train_dir.extend(train)
                train = glob.glob(os.path.join(i, "*"))
                train = [t for t in train if t not in test]
                train_dir.extend(train)

            return {'train':train_dir,'test':test_dir,'name':'tune'}
        else:
            i = raw_feature_dir[0]
            print(os.path.join(i,'*{}_*'.format(validation_trial)))
            #test = glob.glob(os.path.join(i,'*{}_*'.format(validation_trial)))
            test = glob.glob(os.path.join(i,'*S'+test_num+'_*'))#.format(test_num)))
            test_dir.extend(test)
            # tri=[j for j in validation_trial_train]
            # a = "*["+",".join(['{}']*len(validation_trial_train))+"]_*"
            # train = glob.glob(os.path.join(i,a.format(*tri)))
            #         #train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
            #          #   train_trial[idx][2],train_trial[idx][3])))
            # print(os.path.join(i,a.format(*tri)))
            # #breakpoint()
            # train_dir.extend(train)
            train = glob.glob(os.path.join(i, "*"))
            train = [t for t in train if t not in test]
            train_dir.extend(train)

            return {'train':train_dir,'test':test_dir,'name':'tune'}


    # for training
    else:
        cross_val_splits = []
        # for each fold (each task_subject combination)
        for idx, test_num in enumerate(test_trial):
            train_dir = []
            test_dir = []
            if len(raw_feature_dir)!=1:
                for i in raw_feature_dir:
                    # list files for testing
                    #test = glob.glob(os.path.join(i,'{}_*'.format(test_num)))
                    #test = glob.glob(os.path.join(i,'*S'+test_num+'_*'))#.format(test_num)))
                    test = glob.glob(os.path.join(i,test_num+'_*'))
                    test_dir.extend(test)

                    # #breakpoint()
                    # tri=[j for j in train_trial[idx]]
                    # a = "*["+",".join(['{}']*len(train_trial[idx]))+"]_*"
                    # train = glob.glob(os.path.join(i,a.format(*tri)))
                    # #train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
                    #  #   train_trial[idx][2],train_trial[idx][3])))
                    # print(os.path.join(i,a.format(*tri)))
                    #
                    # train_dir.extend(train)

                    # list all other files minus test files
                    train = glob.glob(os.path.join(i, "*"))
                    train = [t for t in train if t not in test]
                    train_dir.extend(train)

            else:
                i = raw_feature_dir[0]
                #test = glob.glob(os.path.join(i,'*{}_*'.format(test_num)))
                #test = glob.glob(os.path.join(i,'*S'+test_num+'_*'))#.format(test_num)))
                test = glob.glob(os.path.join(i,test_num+'_*'))
                test_dir.extend(test)
                #breakpoint()
                # tri=[j for j in train_trial[idx]]
                # a = "*["+",".join(['{}']*len(train_trial[idx]))+"]_*"
                # train = glob.glob(os.path.join(i,a.format(*tri)))
                #     #train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],train_trial[idx][2],train_trial[idx][3])))
                #      #
                # print(os.path.join(i,a.format(*tri)))
                # #breakpoint()
                #
                # train_dir.extend(train)
                train = glob.glob(os.path.join(i, "*"))
                train = [t for t in train if t not in test]
                train_dir.extend(train)

            #breakpoint()
            # add fold sets to cross_val_splits
            cross_val_splits.append({'train': train_dir,
                                    'test': test_dir,
                                    'name': 'test_{}'.format(test_num)})

        return cross_val_splits


def get_cross_val_splits_LOUO(validation = False):
    from config import raw_feature_dir, validation_trial, validation_trial_train, test_trial, train_trial
    if validation ==True:
        cross_val_splits=[]
        test_dir = []
        train_dir = []
        if len(raw_feature_dir)!=1:
            for i in raw_feature_dir:
                print(os.path.join(i,'*{}_*'.format(validation_trial)))
                test = glob.glob(os.path.join(i,'*{}_*'.format(validation_trial)))
                test_dir.extend(test)
                # tri=[j for j in validation_trial_train]
                # a = "*["+",".join(['{}']*len(validation_trial_train))+"]_*"
                # train = glob.glob(os.path.join(i,a.format(*tri)))
                #     #train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
                #      #   train_trial[idx][2],train_trial[idx][3])))
                #
                # print(os.path.join(i,a.format(*tri)))
                # train_dir.extend(train)
                train = glob.glob(os.path.join(i, "*"))
                train = [t for t in train if t not in test]
                train_dir.extend(train)

            return {'train':train_dir,'test':test_dir,'name':'tune'}
        else:
            i = raw_feature_dir[0]
            print(os.path.join(i,'*{}_*'.format(validation_trial)))
            test = glob.glob(os.path.join(i,'*{}_*'.format(validation_trial)))
            test_dir.extend(test)
            # tri=[j for j in validation_trial_train]
            # a = "*["+",".join(['{}']*len(validation_trial_train))+"]_*"
            # train = glob.glob(os.path.join(i,a.format(*tri)))
            #         #train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
            #          #   train_trial[idx][2],train_trial[idx][3])))
            # print(os.path.join(i,a.format(*tri)))
            # #breakpoint()
            # train_dir.extend(train)
            train = glob.glob(os.path.join(i, "*"))
            train = [t for t in train if t not in test]
            train_dir.extend(train)

            return {'train':train_dir,'test':test_dir,'name':'tune'}

    else:
        cross_val_splits = []
        for idx, test_num in enumerate(test_trial):
            train_dir = []
            test_dir = []
            if len(raw_feature_dir)!=1:
                for i in raw_feature_dir:
                    #test = glob.glob(os.path.join(i,'*{}_*'.format(test_num)))
                    test = glob.glob(os.path.join(i,'*S'+test_num+'_*'))#.format(test_num)))
                    test_dir.extend(test)
                    #breakpoint()
                    # tri=[j for j in train_trial[idx]]
                    # a = "*["+",".join(['{}']*len(train_trial[idx]))+"]_*"
                    # train = glob.glob(os.path.join(i,a.format(*tri)))
                    # #train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
                    #  #   train_trial[idx][2],train_trial[idx][3])))
                    # print(os.path.join(i,a.format(*tri)))
                    #
                    # train_dir.extend(train)
                    train = glob.glob(os.path.join(i, "*"))
                    train = [t for t in train if t not in test]
                    train_dir.extend(train)
            else:
                i = raw_feature_dir[0]
                #test = glob.glob(os.path.join(i,'*{}_*'.format(test_num)))
                # test = glob.glob(os.path.join(i,'*S'+test_num+'_*'))#.format(test_num)))
                test = glob.glob(os.path.join(i,'*S'+str(test_num)+'_*'))#.format(test_num)))
                test_dir.extend(test)
                #breakpoint()
                # tri=[j for j in train_trial[idx]]
                # a = "*["+",".join(['{}']*len(train_trial[idx]))+"]_*"
                # train = glob.glob(os.path.join(i,a.format(*tri)))
                #     #train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],train_trial[idx][2],train_trial[idx][3])))
                #      #
                # print(os.path.join(i,a.format(*tri)))
                # #breakpoint()
                #
                # train_dir.extend(train)
                train = glob.glob(os.path.join(i, "*"))
                train = [t for t in train if t not in test]
                train_dir.extend(train)

            #breakpoint()
            cross_val_splits.append({'train': train_dir,
                                    'test': test_dir,
                                    'name': 'test_{}'.format(test_num)})

        return cross_val_splits


def get_cross_val_splits(validation = False):
    from config import raw_feature_dir, validation_trial, validation_trial_train, test_trial, train_trial
    print("beginning of get_cross_val_splits")
    print("test_trial", test_trial)
    print("train_trial", train_trial)
    if validation ==True:
        cross_val_splits=[]
        test_dir = []
        train_dir = []
        if len(raw_feature_dir)!=1:
            for i in raw_feature_dir:
                print(os.path.join(i,'*{}.*'.format(validation_trial)))
                test = glob.glob(os.path.join(i,'*{}.*'.format(validation_trial)))
                test_dir.extend(test)
                train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(validation_trial_train[0],\
                    validation_trial_train[1],validation_trial_train[2],validation_trial_train[3])))
                train_dir.extend(train)

            return {'train':train_dir,'test':test_dir,'name':'tune'}
        else:
            i = raw_feature_dir[0]
            print(os.path.join(i,'*{}.*'.format(validation_trial)))
            test = glob.glob(os.path.join(i,'*{}.*'.format(validation_trial)))
            test_dir.extend(test)
            train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(validation_trial_train[0],\
                validation_trial_train[1],validation_trial_train[2],validation_trial_train[3])))
            train_dir.extend(train)

            return {'train':train_dir,'test':test_dir,'name':'tune'}

    else:
        cross_val_splits = []
        for idx, test_num in enumerate(test_trial):
            train_dir = []
            test_dir = []
            if len(raw_feature_dir)!=1:
                for i in raw_feature_dir:
                    # test = glob.glob(os.path.join(i,'*{}.*'.format(test_num)))
                    # test_dir.extend(test)
                    # #breakpoint()
                    # train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
                    #     train_trial[idx][2],train_trial[idx][3])))
                    # print(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
                    #     train_trial[idx][2],train_trial[idx][3])))

                    # train_dir.extend(train)
                    # test = glob.glob(os.path.join(i,'*T'+test_num+'_*'))
                    test = glob.glob(os.path.join(i,'*T'+test_num+'*'))
                    test_dir.extend(test)

                    train = glob.glob(os.path.join(i, "*"))
                    train = [t for t in train if t not in test]
                    train_dir.extend(train)
            else:  # will follow this path for Sara's capstone
                i = raw_feature_dir[0]
                print("i", i)
                # test = glob.glob(os.path.join(i,'*{}.*'.format(test_num)))
                # test_dir.extend(test)
                # #breakpoint()
                # train = glob.glob(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
                #     train_trial[idx][2],train_trial[idx][3])))
                # print(os.path.join(i,'*[{},{},{},{}].*'.format(train_trial[idx][0],train_trial[idx][1],\
                #     train_trial[idx][2],train_trial[idx][3])))

                # train_dir.extend(train)
                # print(os.path.join(i,'*T'+str(test_num)+'_*'))
                # test = glob.glob(os.path.join(i,'*T'+str(test_num)+'_*'))
                # test = glob.glob(os.path.join(i,'Peg_Transfer_S01_T'+test_num+'*'))
                test = glob.glob(os.path.join(i,'*T'+str(test_num)+'*'))
                print("test", test)
                test_dir.extend(test)
                
                train = glob.glob(os.path.join(i, "*"))
                train = [t for t in train if t not in test]
                train_dir.extend(train)

            print("train_dir in utils.py", train_dir)
            print("test_dir in utils.py", test_dir)
            #breakpoint()
            cross_val_splits.append({'train': train_dir,
                                    'test': test_dir,
                                    'name': 'test_{}'.format(test_num)})

        return cross_val_splits


################### Load File ######################

def get_tcn_model_file(naming):
    from config import tcn_model_dir
    tcn_model_file = os.path.join(tcn_model_dir, naming)
    if not os.path.exists(tcn_model_file):
        os.makedirs(tcn_model_file,exist_ok=True)
    tcn_model_file = os.path.join(tcn_model_file, 'model.pkl')
    return tcn_model_file

def get_tcn_log_sub_dir(naming):
    from config import tcn_log_dir
    sub_dir = os.path.join(tcn_log_dir, naming)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir,exist_ok=True)
    return sub_dir

def clear_dir(dir):
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        try:
            shutil.rmtree(filepath)
        except NotADirectoryError:
            os.remove(filepath)

# To be improved
def set_up_dirs():
    from config import (result_dir, tcn_log_dir, tcn_model_dir,
                        tcn_feature_dir, trpo_model_dir, graph_dir)

    for i in [result_dir, tcn_log_dir, tcn_model_dir,
              tcn_feature_dir, trpo_model_dir, graph_dir]:
        if not os.path.exists(i):
            os.makedirs(i,exist_ok=True)

# To be improved
def clean_up():
    from config import (result_dir, tcn_log_dir, tcn_model_dir,
                        tcn_feature_dir, trpo_model_dir, graph_dir)

    for i in [result_dir, tcn_log_dir, tcn_model_dir,
              tcn_feature_dir, trpo_model_dir, graph_dir]:
        clear_dir(i)


################## Gesture Statistics ####################

def get_class_counts(dataset):  # RAW
    from config import gesture_class_num
    class_num = gesture_class_num

    counts = [0 for i in range(class_num)]

    for data in dataset:
        gesture = data['gesture']
        gesture = gesture[gesture!=-1]

        for i in range(class_num):
            counts[i] += (gesture==i).sum()

    return counts

def get_class_weights(dataset):  # RAW
    from config import gesture_class_num
    class_num = gesture_class_num

    counts = get_class_counts(dataset)

    if 0 in counts:
        return None

    weights = [1/i for i in counts]
    w_sum = sum(weights)
    for i in range(class_num):
        weights[i] = weights[i] * class_num / w_sum

    return weights


def get_transition_matrix(dataset): # TCN
    from config import gesture_class_num

    class_num = gesture_class_num + 1  # Including Init
    matrix = np.zeros((class_num, class_num))  # 10: Init

    for data in dataset:
        gesture = data['label']

        last = class_num - 1  #init
        for i in range(len(gesture)):
            current = int(gesture[i])
            matrix[last][current] += 1
            last = current

    return matrix.astype(int)


def get_normalized_transition_matrix(dataset): # TCN
    from config import gesture_class_num

    class_num = gesture_class_num + 1   # Including Init
    matrix = get_transition_matrix(dataset).astype(float)

    for i in range(class_num):
        matrix[i][i] = 0
        matrix[i] = matrix[i] / (matrix[i].sum() + 1e-20)

    return matrix

def get_gesture_durations(datasets): # TCN   # Multiple dataset possible
    from config import gesture_class_num

    class_num = gesture_class_num
    durations = [[] for i in range(class_num)]

    if type(datasets) != list:
        raise Exception('Input should be put into an array!')

    for dataset in datasets:
        for data in dataset:
            gesture = data['label']

            count = 1
            for i in range(1, len(gesture)):
                if gesture[i-1] == gesture[i]:
                    count += 1
                else:
                    durations[gesture[i-1]].append(count)
                    count = 1

            durations[gesture[i-1]].append(count)

    return durations

def get_duration_statistics(dataset): # TCN

    durations = get_gesture_durations([dataset])

    mus = [np.array(i).mean() for i in durations]
    sigmas = [np.array(i).std() for i in durations]

    # Empty durations handled: Caution!!!
    mus = [0 if np.isnan(i) else i  for i in mus]
    sigmas = [1 if np.isnan(i) else i  for i in sigmas]

    return np.array([mus, sigmas])

def get_min_length(datasets):  # TCN          # Multiple dataset possible

    durations = get_gesture_durations(datasets)

    # Empty durations handled: Caution!!!
    durations = [i if i else [float('inf')]  for i in durations]

    mins = [np.array(i).min() for i in durations]
    min_min = np.array(mins).min()

    return float(min_min)

def get_min_mean_length(datasets):  # TCN     # Multiple dataset possible

    durations = get_gesture_durations(datasets)

    # Empty durations handled: Caution!!!
    durations = [i if i else [float('inf')]  for i in durations]

    means = [np.array(i).mean() for i in durations]
    min_mean = np.array(means).min()

    return min_mean

def get_mean_mean_length(datasets):  # TCN     # Multiple dataset possible

    durations = get_gesture_durations(datasets)

    # Empty durations handled: Caution!!!
    durations = [i if i else [float('inf')]  for i in durations]

    means = [np.array(i).mean() for i in durations]
    mean_mean = np.array(means).mean()

    return mean_mean


################## Visualization ####################

def visualize_result(result):
    result_string = []
    last = ''
    for i in range(result.size):
        label = str(get_reverse_mapped_gesture_label(result[i]))
        if label != last:
            result_string.append(label)
            last = label

    result_string = '-'.join(result_string)

    return result_string


def plot_trail(ls, pred=None, ys=None, show=True, save_file=None):

    fig = plt.figure()
    xs = np.arange(len(ls))
    plt.plot(xs, ls, 'b')
    if ys is not None:
        plt.plot(xs, ys, 'r')
    if pred is not None:
        plt.plot(xs, pred, 'g')
    if save_file is not None:
        fig.savefig(save_file)
    if show:
        plt.show()

    plt.close(fig)


def plot_barcode(gt=None, pred=None, visited_pos=None,
                 show=True, save_file=None):
    from config import gesture_class_num

    if gesture_class_num <= 10:
        color_map = plt.cm.tab10
    else:
        color_map = plt.cm.tab20

    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                interpolation='nearest', vmin=0, vmax=gesture_class_num-1)

    fig = plt.figure(figsize=(18, 4))

    # a horizontal barcode
    if gt is not None:
        ax1 = fig.add_axes([0, 0.65, 1, 0.2], **axprops)
        ax1.set_title('Ground Truth')
        ax1.imshow(gt.reshape((1, -1)), **barprops)

    if pred is not None:
        ax2 = fig.add_axes([0, 0.35, 1, 0.2], **axprops)
        ax2.set_title('Predicted')
        ax2.imshow(pred.reshape((1, -1)), **barprops)

    if visited_pos is not None:
        ax3 = fig.add_axes([0, 0.15, 1, 0.1], **axprops)
        ax3.set_title('Steps of Agent')
        ax3.set_xlim(0, len(gt))
        ax3.plot(visited_pos, np.ones_like(visited_pos), 'ro', markersize=1)

    if save_file is not None:
        fig.savefig(save_file, dpi=400)
    if show:
        plt.show()

    plt.close(fig)


################## Metrics ####################

def get_result_string(result):
    from itertools import groupby

    result_string = ''
    for i in range(result.size):
        result_string += str(int(result[i]))  # No negtive allowed

    result_string = ''.join(i for i, _ in groupby(result_string))
    return result_string


# levenstein
def get_edit_score(out, gt):
    import editdistance

    if type(out) == list:
        tmp = [get_edit_score(out[i], gt[i]) for i in range(len(out))]
        return np.mean(tmp)
    else:
        gt_string = get_result_string(gt)
        out_string = get_result_string(out)
        max_len = max(len(gt_string), len(out_string))
        edit_score = 1 - editdistance.eval(gt_string, out_string) / max_len
        return edit_score * 100

def get_accuracy(out, gt):
    if type(out) == list:
        return np.mean(np.concatenate(out)==np.concatenate(gt)) * 100
    else:
        return np.mean(out==gt) * 100


################## Colin Lea ####################

from numba import jit, int64, boolean
#@jit("float64(int64[:], int64[:], boolean)")
def levenstein_(p,y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i,0] = i
    for i in range(n_col+1):
        D[0,i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1]==p[i-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j]+1,
                             D[i,j-1]+1,
                             D[i-1,j-1]+1)

    if norm:
        score = (1 - D[-1,-1]/max(m_row, n_col) ) * 100
    else:
        score = D[-1,-1]

    return score

def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
    return Yi_split

def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    intervals = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
    return intervals

def get_edit_score_colin(P, Y, norm=True, bg_class=None, **kwargs):
    if type(P) == list:
        tmp = [get_edit_score_colin(P[i], Y[i], norm, bg_class)
                 for i in range(len(P))]
        return np.mean(tmp)
    else:
        P_ = segment_labels(P)
        Y_ = segment_labels(Y)
        if bg_class is not None:
            P_ = [c for c in P_ if c!=bg_class]
            Y_ = [c for c in Y_ if c!=bg_class]
        return levenstein_(P_, Y_, norm)

def get_accuracy_colin(P, Y, **kwargs):  # Average acc
    def acc_(p,y):
        return np.mean(p==y)*100
    if type(P) == list:
        return np.mean([np.mean(P[i]==Y[i]) for i in range(len(P))])*100
    else:
        return acc_(P,Y)


def get_overlap_f1_colin(P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
    def overlap_(p,y, n_classes, bg_class, overlap):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels!=bg_class]
            true_labels = true_labels[true_labels!=bg_class]
            pred_intervals = pred_intervals[pred_labels!=bg_class]
            pred_labels = pred_labels[pred_labels!=bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
            union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
            IoU = (intersection / union)*(pred_labels[j]==true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1


        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()

        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 = 2 * (precision*recall) / (precision+recall)  #RuntimeWarning: invalid value encountered in double_scalars

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1*100

    if type(P) == list:
        return np.mean([overlap_(P[i],Y[i], n_classes, bg_class, overlap) for i in range(len(P))])
    else:
        return overlap_(P, Y, n_classes, bg_class, overlap)
