# Kay Hutchinson 2/18/2022
# Translate predicted MPs to gestures, and get accuracy against original gestures


import os
import sys
import glob
import numpy as np
import pandas as pd
import utils
import textdistance


# Model options
sets = ["DESK", "JIGSAWS", "All"]
vars = ["velocity", "orientation", "all"]
labeltypes = ["gesture", "MP"]
valtypes = ["LOSO", "LOUO"]

# Path to dataset dir
dataDir = "/home/student/Documents/Research/MICCAI_2022/COMPASS-main/Datasets/dV"



# Translate MPs to gestures (need task specific)
# takes in list of MPs and file name, returns list of gestures
# both lists are len(MP) long to facilitate accuracy calculation
def translate(name, MPlist):
    #print("Translating...")

    #print(task)
    # Pull out task from file name
    name = name.lstrip("test_").lstrip("train_")
    trial = name.rstrip("_pred_gt.npy")
    task = trial.split("_")[:-2]
    task = '_'.join(task)
    #print(task)

    # Convert list of MPs to transcript before translation
    MPtranscript = listToTranscript(MPlist)

    # Translate based on task, gestures is a list of gestures in the form <start, end, label>
    if task == "Suturing":
        gestures = translateSuturing(MPtranscript)
    elif task == "Needle_Passing":
        gestures = translateNeedlePassing(MPtranscript)
    elif task == "Knot_Tying":
        gestures = translateKnotTying(MPtranscript)
    elif task == "Peg_Transfer":
        gestures = translatePegTransfer(MPtranscript)
    else:
        print("Can't translate " + trial)


    # Expand label into list of len(MP) with a translated gesture label for each row
    #print(gestures)
    gestureList = transcriptToList(gestures)
    #print(gestureList)

    #print("----")
    #print(MPs)
    #print(gestures)


    return gestureList


# List of labels is n samples long and of the form: [G1, G1, G2, G3, ...] length is number of kinematic samples
# Transcript is of the form: <start, end, label> where start and end are sample numbers
# Sequence of labels is list of labels from the transcript, no frame numbers

# Group all rows with the same MP and return as a new df as <start, end, MP>
def group(dfMP):
    # Find start and end indices of each group of rows with the same context label
    dfMP['subgroup'] = (dfMP['Y'] != dfMP['Y'].shift(1)).cumsum()
    #print(dfMP)

    # Create 'subgroup' column indicating groups of consecutive rows with same MP label
    dfGrouped = dfMP.groupby('subgroup').apply(lambda x: (x['Sample'].iloc[0], x['Sample'].iloc[-1], x['Y'].iloc[0]))
    #print(dfGrouped)

    # Split list from lambda function into columns again
    dfGrouped = pd.DataFrame(dfGrouped.tolist(), index=dfGrouped.index)
    #print(dfGrouped)

    return dfGrouped

# Convert list of labels to a transcript (intermediate step uses dataframes)
def listToTranscript(list):
    dfMP = pd.DataFrame(list, columns=["Y"])
    dfMP.insert(0, 'Sample', range(0, len(list)))

    # Group MPs in a dataframe with start, end, and MP label
    mps = group(dfMP)

    # convert MPs dataframe to list
    transcript = mps.values.tolist()

    return transcript

# Convert transcript to list of labels
def transcriptToList(transcript):
    list = []
    #print(transcript)
    # For each label, fill in the list with that label from start to end sample number
    for t in transcript:
        fill = [t[2]]*(int(t[1])-int(t[0]))
        list[int(t[0]):int(t[1])] = fill

    return list

# Convert transcript to sequence (one way conversion)
def transcriptToSequence(transcript):
    sequence = []
    for i in transcript:
        sequence.append(i[2])
    return sequence


# Load original gesture labels given task, returns list of gesture labels
def loadGs(name):
    # Pull out task from file name
    name = name.lstrip("test_").lstrip("train_")
    trial = name.rstrip("_pred_gt.npy")
    task = trial.split("_")[:-2]
    task = '_'.join(task)
    gestureDir = os.path.join(dataDir, task, "gestures")
    filePath = os.path.join(gestureDir, trial+".txt")

    # Read in labels to list
    labels = []
    with open(filePath) as f:
        lines = f.readlines()
        #print(lines)
        for line in lines:
            labels.append(line.rstrip("\n").split("\t"))  # for DESK
            #labels.append(line.rstrip("\n").rstrip(" ").split(" "))    # for JIGSAWS
    #print(labels)

    # Expand label into list with original gestures for each sample
    gestureList = transcriptToList(labels)
    #print(gestureList)

    return gestureList


# Get edit score between two transcripts
def get_edit_score_kay(pred, gt):
    predSeq = transcriptToSequence(pred)
    gtSeq = transcriptToSequence(gt)

    # Use levenshtein from textdistance package to get distance
    distance = textdistance.levenshtein.distance(predSeq, gtSeq)
    #print(distance)

    # Calculate edit score
    e = (1 - distance/max(len(predSeq), len(gtSeq)))*100
    return e




# Translate Peg_Transfer MPs to gestures
# Both input and output are of the form <start, end, label>
def translatePegTransfer(MPlist):

    # List of gestures
    gestureList = []

    # Context states of each gesture
    S1 = ["Touch"]
    S2 = ["Grasp"]
    S3 = ["Untouch"]
    S4 = ["Touch"]
    S5 = ["Grasp", "Release"]
    S6 = ["Untouch"]
    S7 = ["Touch", "Release", "Untouch"]
    prevG = "S1"

    # Dictionary of gestures
    gestures = {"S1":S1, "S2":S2, "S3":S3, "S4":S4, "S5":S5, "S6":S6, "S7":S7}
    #print(gestures["S1"])

    # Previous and next gestures
    bS1 = ["S7"];                    aS1 = ["S2"];
    bS2 = ["S1"];                    aS2 = ["S3"];
    bS3 = ["S2"];                    aS3 = ["S4"];
    bS4 = ["S3"];                    aS4 = ["S5"];
    bS5 = ["S4"];                    aS5 = ["S6"];
    bS6 = ["S5"];                    aS6 = ["S7"];
    bS7 = ["S6"];                    aS7 = ["S1"];

    # Dictionaries of before and after Sestures
    befores = {"S1":bS1, "S2":bS2, "S3":bS3, "S4":bS4, "S5":bS5, "S6":bS6, "S7":bS7};
    afters = {"S1":aS1, "S2":aS2, "S3":aS3, "S4":aS4, "S5":aS5, "S6":aS6, "S7":aS7};

    # Initialize currG and prevMPs
    currG = prevG
    prevMP = MPlist[0][2]

    # Previous frame number
    prevF = 0

    # Process lines of MPlist
    for line in MPlist:
        # Get current state
        frame = line[0]
        currMP = line[2]
        #print(currMP)
        # For each MP, determine gesture
        currGs = []


        # Check next gesture's states
        for a in afters[prevG]:
            if currMP in gestures[a]:
                #print("a")
                if a not in currGs:
                    currGs.append(a)

        # Check if still in current gesture
        if currMP in gestures[prevG]:
            #print("Same " + prevG)
            #print(gestures[prevG])
            if prevG not in currGs:
                currGs.append(prevG)
            #continue


        # If nothing yet, check going backwards in grammar graph
        if len(currGs) == 0:
            for b in befores[prevG]:
                if currMP in gestures[b]:
                    #print("b")
                    if b not in currGs:
                        currGs.append(b)

        # If nothing found, assume same gesture
        if len(currGs)>1:
            currG = currGs[0]
        elif len(currGs)==1:
            currG = currGs[0]
        else:
            currG = prevG

        # Print line if new gesture found
        if currG != prevG:
            newG = [prevF, frame, prevG]
            gestureList.append(newG)
            prevF = int(frame)
            #print(prevG)

        # Update prevC and prevG
        prevG = currG
        prevMP = currMP

    # Assume last gesture is S7 and fill in end of gesture list
    if int(prevF) < int(frame):
        newG = [prevF, frame, "S7"]
        gestureList.append(newG)

        # Print currGs
        #print(currMP)
        #print(currGs)
        #print("\n")

    return gestureList


# Translate Suturing MPs to gestures
# Both input and output are of the form <start, end, label>
def translateSuturing(MPlist):

    # List of gestures
    gestureList = []

    # Context states of each gesture
    G1 = ["Untouch", "Touch", "Grasp"]
    G2 = ["Touch"]
    G3 = ["Push", "Push"]
    G4 = ["Release", "Untouch", "Touch", "Grasp"]
    G5 = ["Pull", "Touch"]
    G6 = ["Touch", "Grasp", "Release", "Untouch", "Pull"]
    G8 = ["Pull", "Touch", "Grasp", "Release", "Untouch"]
    G9 = ["Touch", "Grasp", "Release", "Untouch"]
    G11 = ["Release", "Untouch", "Touch"]


    # Dictionary of gestures
    gestures = {"G1":G1, "G2":G2, "G3":G3, "G4":G4, "G5":G5, "G6":G6, "G8":G8, "G9":G9, "G11":G11}
    #print(gestures["G1"])


    # Previous and next gestures
    bG1 = [];                         aG1 = ["G2", "G5"];
    bG2 = ["G1", "G4", "G5", "G8"];   aG2 = ["G3"];
    bG3 = ["G2"];                     aG3 = ["G6", "G8"]; #["G6"]; #["G6", "G8"];
    bG4 = ["G6", "G9"];               aG4 = ["G2", "G8"];
    bG5 = ["G1"];                     aG5 = ["G2", "G8"];
    bG6 = ["G3"];                     aG6 = ["G4", "G9", "G11"];
    bG8 = ["G3", "G4", "G5"];         aG8 = ["G2"];
    bG9 = ["G6"];                     aG9 = ["G4", "G11"];
    bG11 = ["G6"];                    aG11 = [];

    # Dictionaries of before and after gestures
    befores = {"G1":bG1, "G2":bG2, "G3":bG3, "G4":bG4, "G5":bG5, "G6":bG6, "G8":bG8, "G9":bG9, "G11":bG11};
    afters = {"G1":aG1, "G2":aG2, "G3":aG3, "G4":aG4, "G5":aG5, "G6":aG6, "G8":aG8, "G9":aG9, "G11":aG11};


    # Initialize gesture to either G1 or G5
    firstLine = MPlist[0]
    fMP = firstLine[2]
    #print(fMP)
    if fMP in gestures["G1"]:
        prevG = "G1"
    elif fMP in gestures["G5"]:
        prevG = "G5"
    else:
        prevG = "G1"
    currG = prevG
    prevMP = fMP


    # Previous frame number
    prevF = 0

    # Process lines of MPlist
    for line in MPlist:
        # Get current state
        frame = line[0]
        currMP = line[2]
        #print(currMP)
        # For each MP, determine gesture
        currGs = []

        # Check if still in current gesture
        if currMP in gestures[prevG]:
            #print("Same " + prevG)
            #print(gestures[prevG])
            if prevG not in currGs:
                currGs.append(prevG)
            #continue

        # Check next gesture's states
        for a in afters[prevG]:
            if currMP in gestures[a]:
                #print("a")
                if a not in currGs:
                    currGs.append(a)

        # If nothing yet, check going backwards in grammar graph
        if len(currGs) == 0:
            for b in befores[prevG]:
                if currMP in gestures[b]:
                    #print("b")
                    if b not in currGs:
                        currGs.append(b)

        # If nothing found, assume same gesture
        if len(currGs)>1:
            currG = currGs[0]
        elif len(currGs)==1:
            currG = currGs[0]
        else:
            currG = prevG

        # Print line if new gesture found
        if currG != prevG:
            newG = [prevF, frame, prevG]
            gestureList.append(newG)
            prevF = int(frame+1)
            #print(prevG)

        # Update prevC and prevG
        prevG = currG
        prevMP = currMP


        # Print currGs
        #print(currMP)
        #print(currGs)
        #print("\n")

    return gestureList




# Analyze a single test/train file with ending "_pred_gt.npy"
def analyzeTrial(trialPath):
    # Pass task and predicted MPs to translate function
    task = trialPath.split("/")[-1]

    # Load .npy file
    [data,pred,gt]=np.load(trialPath,allow_pickle=True)
    got_acc = utils.get_accuracy_colin(pred, gt)
    print(pred)
    print(gt)
    sys.exit()
    #got_edit = utils.get_edit_score_colin(pred, gt)



    # Translate lists to transcripts for edit distance
    predT = listToTranscript(pred)
    gtT = listToTranscript(gt)

    got_edit = get_edit_score_kay(predT, gtT)

    # print(got_acc)
    print(predT)
    print(gtT)
    # print(got_edit)
    print("MPs: \t\tAccuracy: " + str(got_acc) + " \tEdit Score: " + str(got_edit))

    # Translate predicted MPs to gestures based on task
    predGs = translate(task, pred)
    #print(predGs)

    # Load original gesture labels
    gtGs = loadGs(task)

    #print(listToTranscript(predGs[0:len(gtGs)]))
    #print(listToTranscript(gtGs))

    # Get accuracy wrt translated; crop predicted labels to gt since gt shorter
    # and accuracy goes over len of predicted labels
    got_acc_pred = utils.get_accuracy_colin(predGs[0:len(gtGs)], gtGs)

    # Repeat for translated
    predGsT = listToTranscript(predGs)
    gtGsT = listToTranscript(gtGs)
    got_edit_pred = get_edit_score_kay(predGsT, gtGsT)

    # print(got_acc_pred)
    # print(predGsT)
    # print(gtGsT)
    # print(got_edit_pred)
    print("Gestures: \tAccuracy: " + str(got_acc_pred) + " \tEdit Score: " + str(got_edit_pred))


    print("\n")




# Based on calculate_mean_cv, take in path to train_test_result.csv and return
# average performance
def avgPerformance(foldDir):
    result = []

    # Read csv
    tb = pd.read_csv(os.path.join(foldDir, "train_test_result.csv"))

    # Average results
    vals = [tb['v_accuracy'].rolling(window=5).mean().iloc[-1], \
    tb['v_edit_score'].rolling(window=5).mean().iloc[-1], \
    tb['v_loss'].rolling(window=5).mean().iloc[-1], \
    tb['v_f_scores_10'].rolling(window=5).mean().iloc[-1], \
    tb['v_f_scores_25'].rolling(window=5).mean().iloc[-1], \
    tb['v_f_scores_50'].rolling(window=5).mean().iloc[-1], \
    tb['v_f_scores_75'].rolling(window=5).mean().iloc[-1]]
    result.append(vals)

    result = np.array(result)
    print(result)

    return result




# Analyze a model given a path to its results folder
def analyzeModel(folderDir):
    logDir = os.path.join(folderDir, "tcn")
    logs = os.listdir(logDir)

    # For each fold,
    for log in logs:
        # path to log folder for this fold's results
        foldDir = os.path.join(logDir, log, "log")

        print(foldDir)

        # Calculate average accuracy from train_test_result.csv
        avgPerf = avgPerformance(foldDir)
        #print(avgPerf)


    #print(logs)














# MAIN -------------------------------------------------------------------------
if __name__ == "__main__":

    # Directories
    baseDir = os.getcwd()
    resultsDir = os.path.join(baseDir, "Results", "results_02_20_22")

    # Get list of folders in the results folder (contains folders for results of tuning and training)
    folders = os.listdir(resultsDir)
    #print(folders)


    # Test path to npy to translate
    #testPath = "/home/student/Documents/Research/MICCAI_2022/COMPASS-main/TCN/Results/results_02_20_22/DESK_velocity_gesture_LOSO_02_21_2022_0218/tcn/test_5/log/train_Peg_Transfer_S07_T03_pred_gt.npy"
    #analyzeTrial(testPath)


    #testLogPath = "/home/student/Documents/Research/MICCAI_2022/COMPASS-main/TCN/Results/results_02_20_22/JIGSAWS_velocity_MP_LOSO_02_21_2022_0806/tcn/test_5/log"
    #testLogPath = "/home/student/Documents/Research/MICCAI_2022/COMPASS-main/TCN/Results/results_02_20_22/DESK_velocity_MP_LOSO_02_21_2022_0056/tcn/test_6/log"
    testLogPath = "/home/student/Documents/Research/MICCAI_2022/COMPASS-main/TCN/Results/results_02_20_22/DESK_velocity_gesture_LOSO_02_21_2022_0218/tcn/test_6/log"
    files = os.listdir(testLogPath)
    for file in files:
        if file.endswith(".npy"):
            if file.startswith("train_Peg_Transfer") or file.startswith("test_Peg_Transfer"):
            #if file.startswith("train_Suturing") or file.startswith("test_Suturing"):
                filePath = os.path.join(testLogPath, file)
                print(file)
                analyzeTrial(filePath)






'''
    # Find folders of results by set name
    for folder in folders:
        if any(x in sets for x in folder.split("_")):
            #print(folder)
            # folder path
            folderDir = os.path.join(resultsDir, folder)

            # If there is a folder containing log files for a trained model, continue, else pass
            if "tcn" in os.listdir(folderDir):
                #print("found tcn folder")
                #print(folderDir)

                analyzeModel(folderDir)






            else:
                #print("no tcn folder found")
                continue

'''



        #print(folder.split("_"))
        #if folder.split("_").contains("DESK"):
            #print(folder)

    #filePath = "/home/student/Downloads/COMPASS-main/TCN/Results/DESK_velocity_MP_LOSO_02_18_2022_1503/tcn/test_1/log/tcn_test_prediction.npy"

    #[data,pred,gt]=np.load('/home/student/Downloads/COMPASS-main/TCN/Results/DESK_velocity_MP_LOSO_02_18_2022_1648/tcn/test_1/log/test_Peg_Transfer_S01_T01_pred_gt.npy',allow_pickle=True)
    #print(data)
    #print(pred)
    #print(gt)
