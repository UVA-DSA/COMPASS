# Kay Hutchinson
# 1/18/2022
# Translate MPs to DESK peg transfer gestures

# Run in terminal: python3 motion_primitives_to_gestures_peg_transfer.py

import textdistance
import os, glob
import numpy as np

'''
Motion Primitives in an ideal trial (right to left)
S1----------------------------
    Touch(R, Ball/Block/Sleeve)
S2----------------------------
    Grasp(R, Ball/Block/Sleeve)
S3---------------------------
    Untouch(Ball/Block/Sleeve, Pole)
S4---------------------------
    Touch(L, Ball/Block/Sleeve)
S5---------------------------
    Grasp(L, Ball/Block/Sleeve)
    Release(R, Ball/Block/Sleeve)
S6---------------------------
    Untouch(R, Ball/Block/Sleeve)
S7---------------------------
    Touch(Ball/Block/Sleeve, Pole)
    Release(L, Ball/Block/Sleeve)
    Untouch(L, Ball/Block/Sleeve)
S1--------------------------
    ...
'''

'''
# Context states of each gesture (right to left)
S1  = ["Touch(R, Ball/Block/Sleeve)"]
S2 = ["Grasp(R, Ball/Block/Sleeve)"]
S3 = ["Untouch(Ball/Block/Sleeve, Pole)"]
S4 = ["Touch(L, Ball/Block/Sleeve)"]
S5 = ["Grasp(L, Ball/Block/Sleeve)", "Release(R, Ball/Block/Sleeve)"]
S6 = ["Untouch(R, Ball/Block/Sleeve)"]
S7 = ["Touch(Ball/Block/Sleeve, Pole)", "Release(L, Ball/Block/Sleeve)", "Untouch(L, Ball/Block/Sleeve)"]


# Context states of each gesture (left to right)
S1  = ["Touch(L, Ball/Block/Sleeve)"]
S2 = ["Grasp(L, Ball/Block/Sleeve)"]
S3 = ["Untouch(Ball/Block/Sleeve, Pole)"]
S4 = ["Touch(R, Ball/Block/Sleeve)"]
S5 = ["Grasp(R, Ball/Block/Sleeve)", "Release(L, Ball/Block/Sleeve)"]
S6 = ["Untouch(L, Ball/Block/Sleeve)"]
S7 = ["Touch(Ball/Block/Sleeve, Pole)", "Release(R, Ball/Block/Sleeve)", "Untouch(R, Ball/Block/Sleeve)"]


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
'''

# Directories
baseDir = os.getcwd()
# Transcript directories
taskDir = os.path.join(baseDir, "Datasets", "dV", "Peg_Transfer")
mpDir = os.path.join(taskDir, "motion_primitives")
gestureDir = os.path.join(taskDir, "gestures")
DESKDir = "/home/kay/Documents/Research/DESK_Dataset/Peg_Transfer/Da vinci/"


# For a given trial, translate MPs to gestures
# e.g. trial = "Peg_Transfer_S01_T01"
# returns list of gestures and saves translation to file
def translate(trial):
    # List of gestures
    gestureList = []

    # Read context transcript into cT
    mpTranscriptFilePath = os.path.join(mpDir, trial+".txt")
    with open(mpTranscriptFilePath) as mpT:
        lines = mpT.readlines()

    # Initialize gesture to either right or left transfers
    firstLine = lines[1]
    fL = firstLine.split()[2:]
    fG = ' '.join(fL)
    #print(fG)

    # If right to left
    if fG == "Touch(R, Ball/Block/Sleeve)":
        print("Right to left")
        # Context states of each gesture (right to left)
        S1  = ["Touch(R, Ball/Block/Sleeve)"]
        S2 = ["Grasp(R, Ball/Block/Sleeve)"]
        S3 = ["Untouch(Ball/Block/Sleeve, Pole)"]
        S4 = ["Touch(L, Ball/Block/Sleeve)"]
        S5 = ["Grasp(L, Ball/Block/Sleeve)", "Release(R, Ball/Block/Sleeve)"]
        S6 = ["Untouch(R, Ball/Block/Sleeve)"]
        S7 = ["Touch(Ball/Block/Sleeve, Pole)", "Release(L, Ball/Block/Sleeve)", "Untouch(L, Ball/Block/Sleeve)"]
        prevG = "S1"

    # else if left to right
    elif fG == "Touch(L, Ball/Block/Sleeve)":
        print("Left to right")
        # Context states of each gesture (left to right)
        S1  = ["Touch(L, Ball/Block/Sleeve)"]
        S2 = ["Grasp(L, Ball/Block/Sleeve)"]
        S3 = ["Untouch(Ball/Block/Sleeve, Pole)"]
        S4 = ["Touch(R, Ball/Block/Sleeve)"]
        S5 = ["Grasp(R, Ball/Block/Sleeve)", "Release(L, Ball/Block/Sleeve)"]
        S6 = ["Untouch(L, Ball/Block/Sleeve)"]
        S7 = ["Touch(Ball/Block/Sleeve, Pole)", "Release(R, Ball/Block/Sleeve)", "Untouch(R, Ball/Block/Sleeve)"]
        prevG = "S1"
    # Else default right to left
    else:
        print("Not sure which way.")
        S1  = ["Touch(R, Ball/Block/Sleeve)"]
        S2 = ["Grasp(R, Ball/Block/Sleeve)"]
        S3 = ["Untouch(Ball/Block/Sleeve, Pole)"]
        S4 = ["Touch(L, Ball/Block/Sleeve)"]
        S5 = ["Grasp(L, Ball/Block/Sleeve)", "Release(R, Ball/Block/Sleeve)"]
        S6 = ["Untouch(R, Ball/Block/Sleeve)"]
        S7 = ["Touch(Ball/Block/Sleeve, Pole)", "Release(L, Ball/Block/Sleeve)", "Untouch(L, Ball/Block/Sleeve)"]
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
    prevMPs = fG

    # Previous frame number
    prevF = 0

    # Read context transcript into mpT
    outFile = os.path.join(gestureDir, trial+".txt")
    with open(outFile, "w") as out:
        # Process lines
        for line in lines[1:]:
            # Get current state
            frame = line.split()[0]
            myLine = line.split()[2:]
            currMPs = ' '.join(myLine)

            # Handle case with multiple MPs in a line
            MPs = []
            for i in currMPs.split(") "):
                if i[-1] != ")":
                    s = i+")"    # reattach ")" after split
                else:
                    s = i
                MPs.append(s)
            ##print(MPs)

            '''
            # Check each gestures' states and print the gestures that contain the current state
            for k in gestures:
                nG = gestures[k]
                if currMP in nG:
                    print(k)
            #print("\n")
            '''

            #print("Previous: " + prevG)
            #print("Current MPs: " + ' '.join(MPs))
            # For each MP, if there are multiple, determine gesture
            currGs = []
            for mp in MPs:
                #print("Testing: " + mp)
                # Check if MP still in same gesture; skip and go to next iteration
                if mp in gestures[prevG]:
                    #print("Same")
                    if prevG not in currGs:
                        currGs.append(prevG)
                    continue

                # Check next gestures' states
                for a in afters[prevG]:
                    ##print(a)
                    if mp in gestures[a]:
                        #print("Forwards: " + a)
                        if a not in currGs:
                            currGs.append(a)

            # If nothing yet, check going backwards in grammar graph
            if len(currGs) == 0:
                for mp in MPs:
                    for b in befores[prevG]:
                        if mp in gestures[b]:
                            if b not in currGs:
                                currGs.append(b)
                                #print("Backwards: " + b)

            '''
            print("Previous G: " + prevG)
            #print("Curr G: " + ' '.join(currGs))
            print("Curr MP: " + ' '.join(MPs))
            print("Curr G: " + ' '.join(currGs)  +"\n")
            '''

            # Choose currG; go with first one, since higher probability, this is definitely a weak link in the algorithm
            if len(currGs) > 1:
                '''
                print("Help! Multiple gestures found.")
                #print(currGs)
                print("Previous G: " + prevG)
                print("Curr G: " + ' '.join(currGs))
                print("Previous MP: " + ''.join(prevMPs))
                print("Curr MP: " + ' '.join(MPs))
                print("\n")
                '''
                ##prevG = currGs[0]
                currG = currGs[0]
            elif len(currGs) == 1:
                ##prevG = currGs[0]
                currG = currGs[0]
            elif len(currGs) == 0:
                currG = prevG


            # Print line if new gesture found
            if currG != prevG:
                ##print(frame, currC, currG)
                # Write prev gesture to new txt file
                out.write("\t".join([str(prevF), str(frame), str(prevG)]) + "\n")
                #print("\t".join([str(prevF), str(frame), str(prevG)]))
                gestureList.append(prevG)
                prevF = int(frame)+1

            # Update prevC and prevG
            prevG = currG
            prevMPs = currMPs

        # Assume last gesture is S7 and fill in end of gesture transcript
        if int(prevF) < int(frame):
            #print("\t".join([str(prevF), str(frame), "S7"]))
            out.write("\t".join([str(prevF), str(frame), "S7"]) + "\n")
            gestureList.append("S7")

        #print("\n")
        # Close files
        out.close()
        mpT.close()

    return gestureList


# Read in ground truth gesture labels from DESK Dataset
# surgeme transcripts are: "S1/S1_T1_left_color_annot.txt"
def getGroundTruth(trial):
    # List of gestures
    goldenList = []

    # Convert trial name into DESK file name
    trialsplit = trial.split("_")
    task = "_".join(trialsplit[:-2])
    subject = int(trialsplit[-2][1:])
    trialNum = int(trialsplit[-1][1:])


    # concatenate into file path
    DESKfileName = "S" + str(subject) + "_T" + str(trialNum) + "_*_color_annot.txt"
    DESKfilePath = os.path.join(DESKDir, "S"+str(subject))
    DESKfiles = glob.glob(DESKfilePath + "/" + DESKfileName)
    #print(DESKfiles)

    # get ground truth sequences and return
    for i in range(len(DESKfiles)):
        # temp list
        temp = []
        # Read in ground truth gesture transcript
        with open(DESKfiles[i]) as DT:
            lines = DT.readlines()
        # Pull out only gesture sequence and put into temp
        for line in lines:
            l = line.split()
            temp.append(l[2])
        DT.close()
        goldenList.append(temp)

    return goldenList



def getDistance(translated, truth):
    # Drop "G" from the lists
    t1 = []
    t2 = []
    for i in translated:
        t1.append(i[1:])
    for i in truth:
        t2.append(i[1:])
    distance = textdistance.levenshtein.distance(translated, truth)
    return distance




# 1/6/2022
# MAIN -------------------------------------------------------------------------


# Get list of trials
trials = os.listdir(mpDir)

# Set up to take average edit distance
editDistances = []

# Loop through all trials   (only do one for now...)
for trial in sorted(trials):
    trialName = trial.rstrip(".txt")
    #print(trialName)

    # Translate context to gestures for this trial
    gestureList = translate(trialName)

    # Read ground truth labels from JIGSAWS
    goldenList = getGroundTruth(trialName)

    print(trialName)
    #print(gestureList)
    #print(goldenList)

    # Get Levenshtein distance between lists
    for i in range(len(goldenList)):
        #print(goldenList[i])
        d = getDistance(gestureList, goldenList[i])
        # Calculate Edit distance
        e = (1 - d/max(len(gestureList), len(goldenList[i])))*100
        print(gestureList)
        print(goldenList[i])
        print(e)
        print("\n")
        editDistances.append(e)

print("\n")
print(np.average(editDistances))
print(np.std(editDistances))



# EOF
