# Kay Hutchinson 4/14/2021
#
# Combine Touch with Grasp into Grasp and Untouch and Release into Release
#
# To do: 5/9/22
# Add lines to process cases where multiple MPs are on one transcript line



# Imports
import os
import sys
import glob
import csv
from csv import reader, writer
import numpy as np
import pandas as pd



# Function to combine [Touch,Grasp]->Grasp and [Release,Untouch]->Release
def combineMPs(mpDir):
    # Counter
    i = 0

    # For each transcript
    files = glob.glob(mpDir+"/*.txt")
    for file in files:
        # get file name of transcript
        trial = file.split("/")[-1]
        print(trial)

        newlines = []

        # Read in transcript
        with open(file, 'r') as t:
            # skip first line which is a header
            lines = t.readlines()[1:]

            for i in range(len(lines)-1):
                line = lines[i].split()
                #print(line)
                # get MP from line
                mp = line[2].split("(")[0]
                hand = line[2].split("(")[1]
                obj = line[3].split(")")[0]

                # get next MP from next line
                linen = lines[i+1].split()
                mpn = linen[2].split("(")[0]
                handn = linen[2].split("(")[1]
                objn = linen[3].split(")")[0]

                # join touch and grasp into grasp; check that hand and object are the same
                if (mp == "Touch") and (mpn == "Grasp") and (hand == handn) and (obj == objn):
                    start = line[0]
                    end = linen[1]
                    label = " ".join(linen[2:])
                    i = i+1
                    #print("Joining touch and grasp")
                    #print(label)

                # join untouch and release into release; check that hand and object are the same
                elif (mp == "Release") and (mpn == "Untouch") and (hand == handn) and (obj == objn):
                    start = line[0]
                    end = linen[1]
                    label = " ".join(line[2:])
                    i = i+1
                    #print("Joining release and untouch")
                    #print(label)

                else:
                    start = line[0]
                    end = line[1]
                    label = " ".join(line[2:])


                # Handle cases where multiple MPs on one line
                if len(line) > 4:
                    mp2 = line[4].split("(")[0]
                    hand2 = line[4].split("(")[1]
                    obj2 = line[5].split(")")[0]

                    # join touch and grasp into grasp; check that hand and object are the same
                    if (mp == "Touch") and (mp2 == "Grasp") and (hand == hand2) and (obj == obj2):
                        start = line[0]
                        end = line[1]
                        label = " ".join(line[4:])
                        i = i+1
                        #print("Joining touch and grasp")
                        #print(label)

                    # join untouch and release into release; check that hand and object are the same
                    elif (mp == "Release") and (mp2 == "Untouch") and (hand == hand2) and (obj == obj2):
                        start = line[0]
                        end = line[1]
                        label = " ".join(line[2:4])
                        i = i+1
                        #print("Joining release and untouch")
                        #print(label)


                MPs = [str(start), str(end)]
                MPs.append(label)

                # check if new line should be appended to newlines (i.e. skip MPs that got joined in the previous loop)
                if (len(newlines) > 0) and (int(MPs[0]) < int(newlines[-1][1])):
                    #print("skipping")
                    continue

                newlines.append(MPs)
                #print("\n")
                #print(MPs)
            #for n in newlines:
                #print(n)
            #print(i)
            #print(len(lines))
            # get last MP line
            if i == len(lines)-2:
                line = lines[-1].split()
                # get MP from line
                mp = line[2].split("(")[0]
                hand = line[2].split("(")[1]
                obj = line[3].split(")")[0]
                start = line[0]
                end = line[1]
                label = " ".join(line[2:])
                MPs = [str(start), str(end)]
                MPs.append(label)
                newlines.append(MPs)


            # Write to out file
            outMP = os.path.join(mpDirnew, trial)
            with open(outMP, 'w') as o:
                header = ["Start", "Stop", "Motion Primitive"]
                o.write(' '.join(header) + '\n')
                for n in newlines:
                    o.write(' '.join(n))
                    o.write('\n')

    print("Changed: " + str(i))




# Function to combine Grasp and Release into Exchange
def exchangeMPs(mpDir):
    # Counter
    i = 0

    # For each transcript
    files = glob.glob(mpDir+"/*.txt")
    for file in files:
        # get file name of transcript
        trial = file.split("/")[-1]
        print(trial)

        newlines = []

        # Read in transcript
        with open(file, 'r') as t:
            # skip first line which is a header
            lines = t.readlines()[1:]

            for i in range(len(lines)-1):
                line = lines[i].split()
                #print(line)
                # get MP from line
                mp = line[2].split("(")[0]
                hand = line[2].split("(")[1]
                obj = line[3].split(")")[0]

                # get next MP from next line
                linen = lines[i+1].split()
                mpn = linen[2].split("(")[0]
                handn = linen[2].split("(")[1]
                objn = linen[3].split(")")[0]

                # join touch and grasp into grasp; check that hand and object are the same
                if (mp == "Grasp") and (mpn == "Release") and (hand == "L,") and (handn == "R,") and (obj == objn):
                    start = line[0]
                    end = linen[1]
                    label = "Exchange(LR, " + obj + ")"  #" ".join(linen[2:])
                    i = i+1
                    #print("X_LR")
                    #print(label)

                # join untouch and release into release; check that hand and object are the same
                elif (mp == "Grasp") and (mpn == "Release") and (hand == "R,") and (handn == "L,") and (obj == objn):
                    start = line[0]
                    end = linen[1]
                    label = "Exchange(RL," + obj+ ")"#" ".join(line[2:])
                    i = i+1
                    #print("X_RL")
                    #print(label)

                else:
                    start = line[0]
                    end = line[1]
                    label = " ".join(line[2:])


                # Handle cases where multiple MPs on one line
                if len(line) > 4:
                    mp2 = line[4].split("(")[0]
                    hand2 = line[4].split("(")[1]
                    obj2 = line[5].split(")")[0]

                    # join touch and grasp into grasp; check that hand and object are the same
                    if (mp == "Grasp") and (mp2 == "Release") and (hand == "R,") and (hand2 == "L,") and (obj == obj2):
                        start = line[0]
                        end = line[1]
                        label = "Exchange(RL," + obj+ ")"
                        i = i+1
                        #print("X_RL")
                        #print(label)

                    # join untouch and release into release; check that hand and object are the same
                    elif (mp == "Grasp") and (mp2 == "Release") and (hand == "L,") and (hand2 == "R,") and (obj == obj2):
                        start = line[0]
                        end = line[1]
                        label = "Exchange(LR," + obj+ ")"
                        i = i+1
                        #print("X_LR")
                        #print(label)



                MPs = [str(start), str(end)]
                MPs.append(label)

                # check if new line should be appended to newlines (i.e. skip MPs that got joined in the previous loop)
                if (len(newlines) > 0) and (int(MPs[0]) < int(newlines[-1][1])):
                    #print("skipping")
                    continue

                newlines.append(MPs)
                #print(MPs)
            #for n in newlines:
                #print(n)
            #print(i)
            #print(len(lines))
            # get last MP line
            if i == len(lines)-2:
                line = lines[-1].split()
                # get MP from line
                mp = line[2].split("(")[0]
                hand = line[2].split("(")[1]
                obj = line[3].split(")")[0]
                start = line[0]
                end = line[1]
                label = " ".join(line[2:])
                MPs = [str(start), str(end)]
                MPs.append(label)
                newlines.append(MPs)


            # Write to out file
            outMP = os.path.join(mpDirnew, trial)
            with open(outMP, 'w') as o:
                header = ["Start", "Stop", "Motion Primitive"]
                o.write(' '.join(header) + '\n')
                for n in newlines:
                    #print(' '.join(n))
                    o.write(' '.join(n))
                    o.write('\n')


    print("Changed: " + str(i))











# MAIN -------------------------------------------------------------------------
# Get task from command line
try:
    task=sys.argv[1]
    #print(task)
except:
    print("Error: invalid task\nUsage: python adjustMPs.py <task>\nTasks: Suturing, Needle_Passing, Knot_Tying, Pea_on_a_Peg, Post_and_Sleeve, Peg_Transfer")
    sys.exit()

# Directories
baseDir = os.path.dirname(os.getcwd())
# Transcript and video directories
taskDir = os.path.join(baseDir, "Datasets", "dV", task)


# For combining MPs
mpDir = os.path.join(taskDir, "motion_primitives_baseline")
mpDirnew = os.path.join(taskDir, "motion_primitives_combined")
combineMPs(mpDir)


# For creating Exchange (after combined)
mpDir = os.path.join(taskDir, "motion_primitives_combined")
mpDirnew = os.path.join(taskDir, "motion_primitives_exchange")
exchangeMPs(mpDir)











#EOF
