# Kay Hutchinson 4/29/22
#
# Contains functions to convert transcripts to/from lists of labels


import pandas as pd
import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt





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
        fill = [t[2]]*(int(t[1])-int(t[0])+1)
        list[int(t[0]):int(t[1])+1] = fill

    return list



# Convert transcript to sequence (one way conversion)
def transcriptToSequence(transcript):
    sequence = []
    for i in transcript:
        sequence.append(i[2])
    return sequence



# Sample usage for reading and working with an MP transcript
if __name__ == "__main__":

    # Set up paths and directories
    dir = os.getcwd()
    lDir = os.path.join(dir, "motion_primitives_L")
    kDir = os.path.join(dir, "motion_primitives_LK")

    # For each transcript
    for f in os.listdir(kDir):
        # Paths to the transcripts to compare
        transcriptkPath = os.path.join(kDir, f)
        transcriptlPath = os.path.join(lDir, f)

        print("Comparing: " + f)

        # Read in transcripts
        linesk = readMPTranscript(transcriptkPath)
        linesl = readMPTranscript(transcriptlPath)
        # Convert to lists
        listk = transcriptToList(linesk)
        listl = transcriptToList(linesl)
