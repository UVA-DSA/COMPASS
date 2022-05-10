# Kay Hutchinson 2/10/2022
#
# Get stats on the durations of gestures, surgemes, and motion primitives


# Imports
import os
import sys
import glob
import shutil
import csv
from csv import reader, writer
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from preprocess import processArguments


baseDir = os.path.dirname(os.getcwd())
sysDir = os.path.join(baseDir, "Datasets", "dV")
allTasks = ["Suturing", "Needle_Passing", "Knot_Tying", "Peg_Transfer", "Pea_on_a_Peg", "Post_and_Sleeve"]
labeltypes = ["gesture", "MPbaseline", "MPcombined", "MPexchange", "MPleft", "MPright", "MPleftX", "MPrightX"]




# For a given label type, look at all label transcripts and calculate mean and
# stdev of each label type
def getStats(tasks, labeltype, durations, instances):
    # For each task
    for task in tasks:
        print(task)
        taskDir = os.path.join(sysDir, task, labeltype)
        # For each label transcript
        for file in os.listdir(taskDir):
            fileIn = os.path.join(taskDir, file)
            # Read in label transcript
            if labeltype == "gestures":
                tg = pd.read_csv(fileIn, header=None)
            else: #if labeltype == "motion_primitives":
                tg = pd.read_table(fileIn) #, sep="\s")

            #print(tg)

            # For each line, get start and end frames and class label
            for i in range(tg.shape[0]):
                line = tg.iloc[i,0].replace("\t", " ")
                line = line.split()

                start = int(line[0])
                end = int(line[1])
                labels = ''.join(line[2:])
                label = labels.split("(")[0]
                # Calculate duration and update durations and instances dictionaries
                dur = end - start
                #durations[label] = durations[label].append(dur)
                durations[label].append(dur)
                instances[label] = instances[label] + 1

    #print(durations)
    #print(instances)

    # Calculate stats
    for i in durations:
        if instances[i] != 0:
            print(i)
            print("Number of examples: " + str(len(durations[i])))
            print("Average Duration: " + str(sum(durations[i])/len(durations[i])))
            print("Standard Deviation: " + str(np.std(durations[i])))
            print("Max Duration: " + str(max(durations[i])))
            print("Min Duration: " + str(min(durations[i])))
            print("\n")
            #print(i + " " + str(durations[i]/instances[i]))

            # Plot durations as a histogram
            # plt.hist(durations[i])
            # plt.title("Durations of " + i + " in " + set)
            # plt.xlabel("Duration (# samples)")
            # plt.ylabel("Number of instances")
            # plt.show()





# Main
if __name__ == "__main__":
    global set
    # Process arugments from command line and get set, var, and labeltype
    set, var, labeltype, valtype = processArguments(sys.argv)
    if labeltype == "gesture":
        labeltype = "gestures"
    elif labeltype == "MPbaseline":
        labeltype = "motion_primitives_baseline"
    elif labeltype == "MPcombined":
        labeltype = "motion_primitives_combined"
    elif labeltype == "MPexchange":
        labeltype = "motion_primitives_exchange"
    elif labeltype == "MPleft":
        labeltype = "motion_primitives_L"
    elif labeltype == "MPright":
        labeltype = "motion_primitives_R"
    elif labeltype == "MPleftX":
        labeltype = "motion_primitives_LX"
    elif labeltype == "MPrightX":
        labeltype = "motion_primitives_RX"

    # Choose which set of tasks to analyze based on set
    if set == "All":
        tasks = allTasks
    elif set == "JIGSAWS":
        tasks = allTasks[0:3]
    elif set == "DESK":
        tasks = [allTasks[3]]
    elif set == "ROSMA":
        tasks = allTasks[4:6]
    elif (set == "All-5a") or (set == "All-5b"):
        tasks = allTasks[0:4]

    elif set == "S":
        tasks = [allTasks[0]]
    elif set == "NP":
        tasks = [allTasks[1]]
    elif set == "KT":
        tasks = [allTasks[2]]
    elif set == "PoaP":
        tasks = [allTasks[4]]
    elif set == "PaS":
        tasks = [allTasks[5]]
    elif set == "SNP":
        tasks = allTasks[0:2]
    elif set == "PTPaS":
        tasks = [allTasks[3], allTasks[5]]
    #print(tasks)

    # Dictionaries to store number of instances of label class and total duration
    durationsG = {"G1": [], "G2": [], "G3": [], "G4": [], "G5": [], "G6": [], "G7": [], "G8": [], "G9": [], "G10": [], "G11": [], "G12": [], "G13": [], "G14": [], "G15": []}
    durationsS = {"S1": [], "S2": [], "S3": [], "S4": [], "S5": [], "S6": [], "S7": []}
    durationsMP = {"Grasp": [], "Release": [], "Touch": [], "Untouch": [], "Pull": [], "Push": [], "Idle": [], "Exchange": []}
    instancesG = {"G1": 0, "G2": 0, "G3": 0, "G4": 0, "G5": 0, "G6": 0, "G7": 0, "G8": 0, "G9": 0, "G10": 0, "G11": 0, "G12": 0, "G13": 0, "G14": 0, "G15": 0}
    instancesS = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0}
    instancesMP = {"Grasp": 0, "Release": 0, "Touch": 0, "Untouch": 0, "Pull": 0, "Push": 0, "Idle": 0, "Exchange": 0}

    # Choose which set of dictionaries to use based on set and labeltype
    if (labeltype == "gestures") and (set == "JIGSAWS"):
        durations = durationsG
        instances = instancesG
    elif (labeltype == "gestures") and (set == "S"):
        durations = durationsG
        instances = instancesG
    elif (labeltype == "gestures") and (set == "NP"):
        durations = durationsG
        instances = instancesG
    elif (labeltype == "gestures") and (set == "KT"):
        durations = durationsG
        instances = instancesG
    elif (labeltype == "gestures") and (set == "DESK"):
        durations = durationsS
        instances = instancesS
    else: #labeltype == "motion_primitives":
        durations = durationsMP
        instances = instancesMP
    #else:
        #print("Invalid combination of arguments. Please note ROSMA doesn't have gesture labels.")
        #sys.exit()

    getStats(tasks, labeltype, durations, instances)
    print("Done!")


# EOF
