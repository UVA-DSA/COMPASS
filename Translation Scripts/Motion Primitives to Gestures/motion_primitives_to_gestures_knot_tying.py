# Kay Hutchinson
# 1/14/2022
# Translate MPs to JIGSAWS knot tying gestures

# Run in terminal: python3 motion_primitives_to_gestures_knot_tying.py

import textdistance
import os, glob
import numpy as np

'''
Motion Primitives in an ideal trial
G1----------------------------
    Touch(R, Thread)
    Grasp(R, Thread)
G12---------------------------
    Touch(L, Thread)
    Grasp(L, Thread)
G13--------------------------
    Release(R, Thread)
    Untouch(R, Thread)
    Pull(L, Thread)             <- C-loop
G14--------------------------
    Touch(R, Thread)
    Grasp(R, Thread)
G15--------------------------
    Pull(L, Thread) Pull(R, Thread)
G11-------------------------
    Release(L, Thread)
    Untouch(L, Thread)
    Release(R, Thread)
    Untouch(R, Thread)
'''

# Context states of each gesture
G1  = ["Touch(R, Thread)", "Grasp(R, Thread)"]
G11 = ["Release(L, Thread)", "Untouch(L, Thread)", "Release(R, Thread)", "Untouch(R, Thread)"]
G12 = ["Touch(L, Thread)", "Grasp(L, Thread)"] #, "Release(L, Thread)", "Untouch(L, Thread)", "Release(R, Thread)", "Untouch(R, Thread)", "Touch(R, Thread)"]
G13 = ["Release(R, Thread)", "Untouch(R, Thread)", "Pull(L, Thread)"]
G14 = ["Touch(R, Thread)", "Grasp(R, Thread)"] #, "Release(L, Thread)", "Untouch(L, Thread)", "Release(R, Thread)", "Untouch(R, Thread)"]
G15 = ["Pull(L, Thread)", "Pull(R, Thread)"] #, "Release(L, Thread)", "Untouch(L, Thread)", "Release(R, Thread)", "Untouch(R, Thread)"]

# Dictionary of gestures
gestures = {"G1":G1, "G11":G11, "G12":G12, "G13":G13, "G14":G14, "G15":G15}
#print(gestures["G1"])


# Previous and next gestures
bG1 = [];                         aG1 = ["G12"];
bG11 = ["G15"];                   aG11 = [];
bG12 = ["G14", "G15", "G1"];      aG12 = ["G13"];
bG13 = ["G12"];                   aG13 = ["G14"];
bG14 = ["G13", "G15"];            aG14 = ["G12", "G15"];
bG15 = ["G14"];                   aG15 = ["G12", "G14", "G11"];


# Dictionaries of before and after gestures
befores = {"G1":bG1, "G11":bG11, "G12":bG12, "G13":bG13, "G14":bG14, "G15":bG15};
afters = {"G1":aG1, "G11":aG11, "G12":aG12, "G13":aG13, "G14":aG14, "G15":aG15};

# Directories
baseDir = os.getcwd()
# Transcript directories
taskDir = os.path.join(baseDir, "Datasets", "dV", "Knot_Tying")
mpDir = os.path.join(taskDir, "motion_primitives")
gestureDir = os.path.join(taskDir, "gestures")
JIGSAWSDir = "/home/kay/Documents/Research/JIGSAWS_Dataset/Knot_Tying/transcriptions"



# For a given trial, translate MPs to gestures
# e.g. trial = "Suturing_S01_T01"
# returns list of gestures and saves translation to file
def translate(trial):
    # List of gestures
    gestureList = []

    # Read context transcript into cT
    mpTranscriptFilePath = os.path.join(mpDir, trial+".txt")
    with open(mpTranscriptFilePath) as mpT:
        lines = mpT.readlines()

    # Initialize gesture to either G1 or G5
    firstLine = lines[1]
    fL = firstLine.split()[2:]
    fG = ' '.join(fL)
    #print(fG)
    if fG in gestures["G1"]:
        prevG = "G1"
    elif fG in gestures["G12"]:
        prevG = "G12"
    else:
        prevG = "G1"
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

            if len(currMPs) == 0:
                print("Nothing: " + currMPs)
                continue

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

        # Assume last gesture is G11 and fill in end of gesture transcript
        if int(prevF) < int(frame):
            #print("\t".join([str(prevF), str(frame), "G11"]))
            out.write("\t".join([str(prevF), str(frame), "G11"]) + "\n")
            gestureList.append("G11")
        #print("\n")
        # Close files
        out.close()
        mpT.close()

    return gestureList


# Read in ground truth gesture labels from JIGSAWS Dataset
def getGroundTruth(trial):
    # List of gestures
    goldenList = []

    # Convert trial name into JIGSAWS file name
    trialsplit = trial.split("_")
    task = "_".join(trialsplit[:-2])
    subject = trialsplit[-2][1:]
    trialNum = trialsplit[-1][1:]

    # Convert subject number to letter
    subjectletter = chr(ord('@')+int(subject))

    # Format trial number
    trialString = "0"+trialNum

    # concatenate into file path
    JfileName = task + "_" + subjectletter + trialString + ".txt"
    JfilePath = os.path.join(JIGSAWSDir, JfileName)

    # Read in ground truth gesture transcript
    with open(JfilePath) as JT:
        lines = JT.readlines()

    # Pull out only gesture sequence and put into goldenList
    for line in lines:
        l = line.split()
        goldenList.append(l[2])

    # Close file
    JT.close()

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
for trial in sorted(trials): #[4:5]:  #[1:]:
    trialName = trial.rstrip(".txt")
    #print(trialName)

    # Translate context to gestures for this trial
    gestureList = translate(trialName)

    # Read ground truth labels from JIGSAWS
    goldenList = getGroundTruth(trialName)

    print(trialName)
    print(gestureList)
    print(goldenList)

    # Get Levenshtein distance between lists
    d = getDistance(gestureList, goldenList)

    # Calculate Edit distance
    e = (1 - d/max(len(gestureList), len(goldenList)))*100
    print(e)
    print("\n")
    editDistances.append(e)

print(np.average(editDistances))
print(np.std(editDistances))



# EOF
