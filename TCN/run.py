# Kay Hutchinson 2/18/2022
# Script to automate TCN experiments for MICCAI
# Runs preprocess.py, tune.py, and train_test_val.py for combinations
# of dataset, labeltype, and valtype and saves results.

import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
import time
from datetime import datetime
import shutil


# Model options
sets = ["DESK", "JIGSAWS", "All"]
vars = ["velocity", "orientation", "all"]
labeltypes = ["MP", "gesture"]
valtypes = ["LOSO", "LOUO"]


# Create folder for results
dir = os.getcwd()
resultsDir = os.path.join(dir, "Results")


# Iterate through each combination of settings
for set in sets[0:2]:
    for var in vars[0:1]:
        for labeltype in labeltypes:
            for valtype in valtypes:

                # Create folder named by current time and config
                now = datetime.now()
                timeNow = now.strftime("%m_%d_%Y_%H%M")
                logFolder = set +"_"+ var +"_"+ labeltype +"_"+ valtype +"_"+ timeNow
                logDir =  os.path.join(resultsDir, logFolder)
                os.mkdir(logDir)

                print("Results will be stored in: " + logDir)
                # Copy config file over first
                # path to config file
                configPath = os.path.join(dir, "config.json")
                shutil.copy2(configPath, logDir)



                # Preprocess
                # File to pipe outputs to:
                print("Preprocessing " + set + " " + var + " " + labeltype + " " + valtype)
                prepOut = os.path.join(logDir, "prep.txt")
                preprocessTask = "python preprocess.py " + set + " " + var + " " + labeltype + " " + valtype + " > " + prepOut
                #print(preprocessTask)
                subprocess.call(preprocessTask, shell=True)

                # Tune
                # File to pipe outputs to
                print("Tuning " + set + " " + var + " " + labeltype + " " + valtype)
                tuneOut = os.path.join(logDir, "tune.txt")
                tuneTask = "python tune.py " + set + " " + var + " " + labeltype + " " + valtype + " > " + tuneOut
                #print(tuneTask)
                subprocess.call(tuneTask, shell=True)

print("Done!")



























# EOF
