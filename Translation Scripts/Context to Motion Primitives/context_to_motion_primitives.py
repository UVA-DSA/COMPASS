# Kay Hutchinson 12/1/2021
#
# Convert context transcripts to motion primitives
#
# 2/8/2022 Updates:
# Added pea states, peg states for PaS and PT; update to KT knot state
# Added Wrap() and Unwrap() MPs to KT, specific to that circular motion and
# possible errors from the wrap coming out


# Imports
import os
import sys
import glob
import csv
from csv import reader, writer
import numpy as np
import pandas as pd


# Context and states from gesture_segmentation_labeling_context.py
# General:
#      LHold, LContact, RHold, RContact
GContext = ["LHold", "LContact", "RHold", "RContact"]
# Suturing/Needle Passing:
#      ... Needle, Thread
SNPContext = ["Needle"] #, "Thread"]   # thread state removed 12/1/21
# Knot Tying:
#      ... Knot
KTContext = ["Knot"]   # c-loop state merged into knot status and changed to "thread wrapped" 2/8/22
# Peg Transfer/Post and Sleeve
#      ... Peg
PegContext = ["Peg"]
# Pea on a Peg:
#      ... Pea
PoaPContext = ["Pea"]


# List of objects and states
objects = ["Nothing", "Ball/Block/Sleeve", "Needle", "Thread", "Fabric/Tissue", "Ring", "Other"]
needleStates = ["Out of", "Touching", "In"]
#threadStates = ["Loose", "Taut"]   # not used
#cLoopStates = ["Not formed", "Formed"]  # not used
knotStates = ["N/A", "Thread Wrapped", "Loose", "Tight"]
pegStates = ["On", "Off"]
peaStates = ["Not held", "In cup", "Stuck together", "Not stuck together", "On peg"]




# Function to condense all context variables into one column for easier comparison
# Takes df of raw context transcript and concatenates state variables into one
# string for easier comparison, returned df structure is <frame, context>
def condenseContext(df):
    # number of columns in DataFrame
    numCols = len(df.columns)
    numContext = numCols - 1    # first column is frame number

    # new dataframe to hold frame and context
    dfContext = pd.DataFrame(columns = ['Frame'])
    dfContext['Frame'] = df.iloc[:,0]
    #print(dfContext)


    # concatenate states into one string
    # initialize with first context variable
    dfContext['Context'] = df.iloc[:,1]
    for i in range(2, numContext+1):
        #print(i)
        dfContext['Context'] = dfContext['Context'].map(str) + df.iloc[:,i].map(str)

    # If task is Knot Tying, convert 00003 into 00000 because the transition 00003 -> 00000 doesn't mean anything
    if task == "Knot_Tying":
        dfContext['Context'] = dfContext['Context'].replace(["00003"],"00000")

    # If task is Pea on a Peg, convert 00004 into 00000 because the transition 00004 -> 00000 doesn't mean anything
    # and 00004 is an invalid state because the pea state is only used if a pea is held
    if task == "Pea_on_a_Peg":
        dfContext['Context'] = dfContext['Context'].replace(["00004"],"00000")

    return dfContext



# Function to group all rows with same context label and write into a new df as <start, end, context>
def group(dfContext):
    # find start and end indices of each group of rows with the same context label
    dfContext['subgroup'] = (dfContext['Context'] != dfContext['Context'].shift(1)).cumsum()
    # print('dfContext')
    # print(dfContext)
    # print('dfContext[\'subgroup\']')
    # print(dfContext['subgroup'])
    # print('dfContext.groupby(\'subgroup\',as_index=False)')
    # print(dfContext.groupby('subgroup',as_index=False))

    # df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    # df = df.apply(lambda x: [1, 2], axis=1, result_type='broadcast')
    # print(df)

    # ORIGINAL START
    # # create 'subgroup' column indicating groups of consecutive rows with same 'Context' label
    # dfGrouped = dfContext['subgroup'].apply(lambda x: [x['Frame'].iloc[0], x['Frame'].iloc[-1], x['Context'].iloc[0]], axis=1, result_type='broadcast') 
    # print(dfGrouped)

    # # cast to df and return
    # myDF = pd.DataFrame(dfGrouped.tolist(), columns=["Start", "End", "Context"])
    # return myDF
    # ORIGINAL END

    # dfGrouped = dfContext.groupby('subgroup',as_index=False).apply(lambda x: (x['Frame'].iloc[0], x['Frame'].iloc[-1], x['Context'].iloc[0]))  #head(1)))
    # dfGrouped = dfContext.groupby('subgroup',as_index=False).apply(lambda x: {print('here is x', x)})
    # dfGrouped = dfContext.groupby('subgroup',as_index=False).apply(lambda x: [x['Frame'].iloc[0], x['Frame'].iloc[-1], x['Context'].iloc[0]], axis=1, result_type='broadcast') 
    
    # myDF = pd.DataFrame(dfContext['subgroup'].tolist())
    # print('myDF')
    # print(myDF)

    # for key, item in dfContext.groupby('subgroup',as_index=False):
    #     print(dfContext.groupby('subgroup',as_index=False).get_group(key), "\n\n")

    # df.groupby('group').agg(
    #          a_sum=('a', 'sum'),
    #          a_mean=('a', 'mean'),
    #          b_mean=('b', 'mean'),
    #          c_sum=('c', 'sum'),
    #          d_range=('d', lambda x: x.max() - x.min())

    # dfGrouped = dfContext.groupby('subgroup',as_index=False).apply(lambda x: [x['Frame'].iloc[0], x['Frame'].iloc[-1], x['Context'].iloc[0]]) 
    dfGrouped = dfContext.groupby('subgroup', as_index=False).agg(Start=('Frame', lambda x: x.iloc[0]), End=('Frame', lambda x: x.iloc[-1]), Context=('Context', lambda x: x.iloc[0]))
    myDF = dfGrouped[['Start', 'End', 'Context']]
    # print(dfGrouped)
    return myDF

    # cast to df and return
    # myDF = pd.DataFrame(dfGrouped.tolist(), columns=["Start", "End", "Context"]) # dfGrouped is a Series
    # return myDF





# Look at current and next state and determine the MP, tool, and object involved
def labelMPs(npGrouped):
    #print(npGrouped)
    # MP transcript
    MPList = []
    # for each context group
    for g in range(len(npGrouped)-1):
        # list of MPs for this segment which will be written to the MP transcript .txt file
        MPs = [str(npGrouped[g,0]), str(npGrouped[g, 1]+timeStep-1)]  # start with start and end frames

        # get current and next states
        currState = npGrouped[g, 2]
        nextState = npGrouped[g+1, 2]

        # Get list of states that change between two consecutive groups
        # diff = [s for s in xrange(len(currState)) if currState[s] != nextState[s]]
        # print(currState)
        # print(type(currState))
        diff = [s for s in range(len(currState)) if currState[s] != nextState[s]]
        # check if multiple states change and it's not a L/R grasper transitioning from a contact to a hold
        c = diff

        #print(currState + " -> " + nextState)
        #print(c)

        # figure out which MPs occurred
        # The following sections of code implement the motion primitives defined below
        # which is as of 12/1/21. The code also allows for short cuts that skip
        # the touch or untouch parts of grasp and release and is thus organized
        # so that grasp and release supercede changes in contact states.
        '''
        General context:
        Touch(L, a): X0XX -> XaXX
        Touch(R, a): XXX0 -> XXXa
        Grasp(L, a): 0aXX -> aXXX
        Grasp(R, a): XX0a -> XXaX
        Release(L, a): aXXX -> 0aXX
        Release(R, a): XXaX -> XX0a
        Untouch(L, a): XaXX -> X0XX
        Untouch(R, a): XXXa -> XXX0

        Untouch(L, a) and Touch(L, b): XaXX -> XbXX
        Untouch(R, a) and Touch(R, b): XXXa -> XXXb

        Task-specific:
        Suturing and Needle Passing:
        Touch(2, 4): 2XXX0 -> 2XXX1
        Touch(2, 4): XX2X0 -> XX2X1
        Push(2, 4): 2XXX1 -> 2XXX2
        Push(2, 4): XX2X1 -> XX2X2
        Pull(L, 2): 2XXX1 -> 2XXX0
        Pull(R, 2): XX2X1 -> XX2X0

        Knot Tying:
        Pull(L, 3): 3XXX0 -> 3XXX1   # make wrap around R
        Pull(R, 3): XX3X0 -> XX3X1   # make wrap  'str'>
around L
        Pull(L, 3): 3XXX1 -> 3XXX0  # wrap around R comes undone
        Pull(R, 3): XX3X1 -> XX3X1  # wrap around L comes undone
        Pull(L, 3) and Pull(R, 3): 3X3X1 -> 3X3X2  # pull tail through wrap
        Pull(L, 3) and Pull(R, 3): 3X3X2 -> 3X3X3  # tighten knot

        Note: 00003 -> 00000 lumped into next MP

        Pea on a Peg:
        Grasp(L, 1): 0XXX0 -> 1XXX1   # Grasp pea in cup with L
        Grasp(R, 1): XX0X0 -> XX1X1   # Grasp pea in cup with R
        Pull(L, 1): 1XXX1 -> 1XXX2    # Lift pea from cup with L
        Pull(R, 1): XX1X1 -> XX1X2    # Lift pea from cup with R
        Pull(L, 1): 1XXX1 -> 1XXX3    # Lift pea from cup with L
        Pull(R, 1): XX1X1 -> XX1X3    # Lift pea from cup with R
        Untouch(1, 1): XXXX2 -> XXXX3 # Isolate held pea
        Touch(1, Peg): XXXX3 -> XXXX4 # Touch pea to peg
        Untouch(1, Peg): XXXX4 -> XXXX3 # Untouch pea from peg
        Release(L, 1): 1XXXa -> 0XXX0 # Release pea with L
        Release(R, 1): XX1Xa -> XX0X0 # Release pea with R

        Push(L, 1): 1XXX2 -> 1XXX1    # Push peas back in cup with L
        Push(R, 1): XX1X2 -> XX1X1    # Push peas back in cup with R
        Touch(1, 1): XXXX3 -> XXXX2   # Touch pea to another pea and they stick together

        Peg Transfer and Post and Sleeve:
        Touch(1, Post): XXXX0 -> XXXX1
        Untouch(1, Post): XXXX1 -> XXXX0

        '''

        # reset tool, verb, and obj
        tool = " "
        verb = " "
        obj = " "

        # Untouch(L, a): XaXX -> X0XX
        if (int(currState[1]) > 0) and (int(nextState[1]) == 0):
            tool = "L"
            verb = "Untouch"
            obj = objects[int(currState[1])]
            #print(verb + "(" + tool + ", " + obj +")")
        # Grasp(L, a): 0XXX -> aXXX   (shortcut)
        if (int(currState[0]) == 0) and (int(nextState[0]) > 0):
            tool = "L"
            verb = "Grasp"
            obj = objects[int(nextState[0])]
            #print(verb + "(" + tool + ", " + obj +")")
        # Grasp(L, a): 0aXX -> aXXX
        if (int(currState[0]) == 0) and (int(nextState[0]) == int(currState[1]) > 0):
            tool = "L"
            verb = "Grasp"
            obj = objects[int(nextState[0])]
        #print(verb + "(" + tool + ", " + obj +")")

        if (tool != " "):
            MPs.append(verb + "(" + tool + ", " + obj + ")")
        # Untouch(L, a) and Touch(L, b): XaXX -> XbXX
        elif (int(currState[1]) > 0) and (int(nextState[1]) > 0) and (int(currState[1]) != int(nextState[1])):
            tool = "L"
            verb = "Untouch"
            obj = objects[int(currState[1])]
            MPs.append(verb + "(" + tool + ", " + obj + ")")
            tool = "L"
            verb = "Touch"
            obj = objects[int(nextState[1])]
            MPs.append(verb + "(" + tool + ", " + obj + ")")

        # reset tool, verb, and obj
        tool = " "
        verb = " "
        obj = " "

        # Untouch(R, a): XXXa -> XXX0
        if (int(currState[3]) > 0) and (int(nextState[3]) == 0):
            tool = "R"
            verb = "Untouch"
            obj = objects[int(currState[3])]
            #print(verb + "(" + tool + ", " + obj +")")
        # Grasp(R, a): XX0X -> XXaX     (shortcut)
        if (int(currState[2]) == 0) and (int(nextState[2]) > 0):
            tool = "R"
            verb = "Grasp"
            obj = objects[int(nextState[2])]
            #print(verb + "(" + tool + ", " + obj +")")
        # Grasp(R, a): XX0a -> XXaX
        if (int(currState[2]) == 0) and (int(nextState[2]) == int(currState[3]) > 0):
            tool = "R"
            verb = "Grasp"
            obj = objects[int(nextState[2])]
        #print(verb + "(" + tool + ", " + obj +")")

        if (tool != " "):
            MPs.append(verb + "(" + tool + ", " + obj + ")")
        # Untouch(R, a) and Touch(R, b): XXXa -> XXXb
        elif (int(currState[3]) > 0) and (int(nextState[3]) > 0) and (int(currState[3]) != int(nextState[3])):
            tool = "R"
            verb = "Untouch"
            obj = objects[int(currState[3])]
            MPs.append(verb + "(" + tool + ", " + obj + ")")
            tool = "R"
            verb = "Touch"
            obj = objects[int(nextState[3])]
            MPs.append(verb + "(" + tool + ", " + obj + ")")

        # reset tool, verb, and obj
        tool = " "
        verb = " "
        obj = " "


        # Touch(L, a): X0XX -> XaXX
        if (int(currState[1]) == 0) and (int(nextState[1]) > 0):
            tool = "L"
            verb = "Touch"
            obj = objects[int(nextState[1])]
            #print(verb + "(" + tool + ", " + obj +")")
        # Release(L, a): aXXX -> 0XXX     (shortcut)
        if (int(currState[0]) > 0) and (int(nextState[0]) == 0):
            tool = "L"
            verb = "Release"
            obj = objects[int(currState[0])]
            #print(verb + "(" + tool + ", " + obj +")")
        # Release(L, a): aXXX -> 0aXX
        if (int(nextState[0]) == 0) and (int(nextState[1]) == int(currState[0]) > 0):
            tool = "L"
            verb = "Release"
            obj = objects[int(currState[0])]
        #print(verb + "(" + tool + ", " + obj +")")

        if (tool != " "):
            MPs.append(verb + "(" + tool + ", " + obj + ")")

        # reset tool, verb, and obj
        tool = " "
        verb = " "
        obj = " "

        # Touch(R, a): XXX0 -> XXXa
        if (int(currState[3]) == 0) and (int(nextState[3]) > 0):
            tool = "R"
            verb = "Touch"
            obj = objects[int(nextState[3])]
            #print(verb + "(" + tool + ", " + obj +")")
        # Release(R, a): XXaX -> XX0X     (shortcut)
        if (int(currState[2]) > 0) and (int(nextState[2]) == 0):
            tool = "R"
            verb = "Release"
            obj = objects[int(currState[2])]
            #print(verb + "(" + tool + ", " + obj +")")
        # Release(R, a): XXaX -> XX0a
        if (int(nextState[2]) == 0) and (int(nextState[3]) == int(currState[2]) > 0):
            tool = "R"
            verb = "Release"
            obj = objects[int(currState[2])]
        #print(verb + "(" + tool + ", " + obj +")")
        if (tool != " "):
            MPs.append(verb + "(" + tool + ", " + obj + ")")

        # reset tool, verb, and obj
        tool = " "
        verb = " "
        obj = " "



        # task-specific changes
        # Suturing and Needle_Passing
        if (task == "Suturing") or (task == "Needle_Passing"):
            # Touch(2, 4): XXXX0 -> XXXX1
            if (int(currState[4]) == 0) and (int(nextState[4]) == 1):
                tool = objects[2]
                verb = "Touch"
                obj = objects[4]  # fabric
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "
            # Untouch(2, 4): XXXX1 -> XXXX0
            elif (int(currState[4]) == 1) and (int(nextState[4]) == 0):
                tool = objects[2]
                verb = "Untouch"
                obj = objects[4]  # fabric
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "
            # Push(2, 4): XXXX1 -> XXXX2
            if (int(currState[4]) == 1) and (int(nextState[4]) == 2):
                tool = objects[2]
                verb = "Push"
                obj = objects[4]  # fabric
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "
            # Pull(2, 4): XXXX2 -> XXXX1
            elif (int(currState[4]) == 2) and (int(nextState[4]) == 1):
                tool = objects[2]
                verb = "Pull"
                obj = objects[4]  # fabric
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "
            # Push(2, 4): XXXX0 -> XXXX2  (shortcut)
            if (int(currState[4]) == 0) and (int(nextState[4]) == 2):
                tool = objects[2]
                verb = "Push"
                obj = objects[4]  # fabric
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "
            # Pull(L, 2): 2XXX2 -> 2XXX0     OR    Pull(R, 2): XX2X2 -> XX2X0
            if (int(currState[4]) == 2) and (int(nextState[4]) == 0):
                if (int(currState[0]) == 2):
                    tool = "L"
                if (int(currState[2]) == 2):
                    tool = "R"
                verb = "Pull"
                obj = objects[2]  # needle
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "


        # Knot_Tying
        if (task == "Knot_Tying"):
            # Pull(L, 3): 3XXX0 -> 3XXX1   # make wrap around R
            if (int(currState[0]) == 3) and (int(currState[4]) == 0) and (int(nextState[4]) == 1):
                tool = "L"
                verb = "Pull"
                obj = objects[3]
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "
            # Pull(R, 3): XX3X0 -> XX3X1   # make wrap around L
            if (int(currState[2]) == 3) and (int(currState[4]) == 0) and (int(nextState[4]) == 1):
                tool = "R"
                verb = "Pull"
                obj = objects[3]
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "
            # Pull(L, 3): 3XXX1 -> 3XXX0   # wrap around R comes undone
            if (int(currState[0]) == 3) and (int(currState[4]) == 1) and (int(nextState[4]) == 0):
                tool = "L"
                verb = "Pull"
                obj = objects[3]
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "
            # Pull(R, 3): XX3X1 -> XX3X0   # wrap around L comes undone
            if (int(currState[2]) == 3) and (int(currState[4]) == 1) and (int(nextState[4]) == 0):
                tool = "R"
                verb = "Pull"
                obj = objects[3]
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "
            # Pull(L, 3) and Pull(R, 3): 3X3X1 -> 3X3X2  # pull tail through wrap
            if (int(currState[0]) == 3) and (int(currState[2]) == 3) and (int(currState[4]) == 1) and (int(nextState[4]) == 2):
                tool = "L"
                verb = "Pull"
                obj = objects[3]
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")
                tool = "R"
                verb = "Pull"
                obj = objects[3]
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "
            # Pull(L, 3) and Pull(R, 3): 3X3X2 -> 3X3X3  # tighten knot
            if (int(currState[0]) == 3) and (int(currState[2]) == 3) and (int(currState[4]) == 2) and (int(nextState[4]) == 3):
                tool = "L"
                verb = "Pull"
                obj = objects[3]
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")
                tool = "R"
                verb = "Pull"
                obj = objects[3]
                #print(verb + "(" + tool + ", " + obj +")")
                if (tool != " "):
                    MPs.append(verb + "(" + tool + ", " + obj + ")")

                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "



        # Pea_on_a_Peg
        if (task == "Pea_on_a_Peg"):
            #Grasp(L, 1): 0XXX0 -> 1XXX1   # Grasp pea in cup with L
            #Grasp(R, 1): XX0X0 -> XX1X1   # Grasp pea in cup with R
                # these should be covered by general context grasp

            #Pull(L, 1): 1XXX1 -> 1XXX2    # Lift pea from cup with L
            if (int(currState[0]) == 1) and (int(currState[4]) == 1) and (int(nextState[4]) == 2):
                tool = "L"
                verb = "Pull"
                obj = objects[1]

            #Pull(R, 1): XX1X1 -> XX1X2    # Lift pea from cup with R
            if (int(currState[2]) == 1) and (int(currState[4]) == 1) and (int(nextState[4]) == 2):
                tool = "R"
                verb = "Pull"
                obj = objects[1]

            if (tool != " "):
                MPs.append(verb + "(" + tool + ", " + obj + ")")

            # reset tool, verb, and obj
            tool = " "
            verb = " "
            obj = " "

            #Pull(L, 1): 1XXX1 -> 1XXX3    # Lift pea from cup with L
            if (int(currState[0]) == 1) and (int(currState[4]) == 1) and (int(nextState[4]) == 3):
                tool = "L"
                verb = "Pull"
                obj = objects[1]

            #Pull(R, 1): XX1X1 -> XX1X3    # Lift pea from cup with R
            if (int(currState[2]) == 1) and (int(currState[4]) == 1) and (int(nextState[4]) == 3):
                tool = "R"
                verb = "Pull"
                obj = objects[1]

            if (tool != " "):
                MPs.append(verb + "(" + tool + ", " + obj + ")")

            # reset tool, verb, and obj
            tool = " "
            verb = " "
            obj = " "

            #Untouch(1, 1): XXXX2 -> XXXX3 # Isolate held pea
            if (int(currState[4]) == 2) and (int(nextState[4]) == 3):
                verb = "Untouch"
                obj = objects[1]
                obj2 = objects[1]
                MPs.append(verb + "(" + obj + ", " + obj2 + ")")
                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "

            #Touch(1, Peg): XXXX3 -> XXXX4 # Touch pea to peg
            if (int(currState[4]) == 3) and (int(nextState[4]) == 4):
                verb = "Touch"
                obj = objects[1]
                obj2 = "Peg"
                MPs.append(verb + "(" + obj + ", " + obj2 + ")")
                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "

            #Untouch(1, Peg): XXXX4 -> XXXX3 # Untouch pea from peg
            if (int(currState[4]) == 4) and (int(nextState[4]) == 3):
                verb = "Untouch"
                obj = objects[1]
                obj2 = "Peg"
                MPs.append(verb + "(" + obj + ", " + obj2 + ")")
                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "

            #Release(L, 1): 1XXXa -> 0XXX0 # Release pea with L
            # Should be covered by general context release
            '''
            if (int(currState[0]) == 1) and (int(currState[4]) != 0) and (int(nextState[0]) == 0) and (int(nextState[4]) == 0):
                tool = "L"
                verb = "Release"
                obj = objects[int(currState[0])]
            '''

            #Release(R, 1): XX1Xa -> XX0X0 # Release pea with R
            # should be covered by general context release
            '''
            if (int(currState[2]) == 1) and (int(currState[4]) != 0) and (int(nextState[2]) == 0) and (int(nextState[4]) == 0):
                tool = "R"
                verb = "Release"
                obj = objects[int(currState[0])]

            #print(verb + "(" + tool + ", " + obj +")")
            if (tool != " "):
                MPs.append(verb + "(" + tool + ", " + obj + ")")

            # reset tool, verb, and obj
            tool = " "
            verb = " "
            obj = " "
            '''

            #Push(L, 1): 1XXX2 -> 1XXX1    # Push peas back in cup with L
            if (int(currState[0]) == 1) and (int(currState[4]) == 2) and (int(nextState[4]) == 1):
                tool = "L"
                verb = "Push"
                obj = objects[1]

            #Push(R, 1): XX1X2 -> XX1X1    # Push peas back in cup with R
            if (int(currState[2]) == 1) and (int(currState[4]) == 2) and (int(nextState[4]) == 1):
                tool = "R"
                verb = "Push"
                obj = objects[1]

            if (tool != " "):
                MPs.append(verb + "(" + tool + ", " + obj + ")")

            # reset tool, verb, and obj
            tool = " "
            verb = " "
            obj = " "

            #Touch(1, 1): XXXX3 -> XXXX2   # Touch pea to another pea and they stick together
            if (int(currState[4]) == 3) and (int(nextState[4]) == 2):
                verb = "Touch"
                obj = objects[1]
                obj2 = objects[1]
                MPs.append(verb + "(" + obj + ", " + obj2 + ")")
                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "


        # Peg_Transfer and Post_and_Sleeve
        if (task == "Peg_Transfer") or (task == "Post_and_Sleeve"):
            # Untouch(1, Pole/Post): XXXX0 -> XXXX1
            if (int(currState[4]) == 0) and (int(nextState[4]) == 1):
                verb = "Untouch"
                obj = objects[1]
                if (task == "Peg_Transfer"):
                    obj2 = "Pole"
                else:
                    obj2 = "Post"
                MPs.append(verb + "(" + obj + ", " + obj2 + ")")
                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "

            # Touch(1, Pole/Post): XXXX1 -> XXXX0
            if (int(currState[4]) == 1) and (int(nextState[4]) == 0):
                verb = "Touch"
                obj = objects[1]
                if (task == "Peg_Transfer"):
                    obj2 = "Pole"
                else:
                    obj2 = "Post"
                MPs.append(verb + "(" + obj + ", " + obj2 + ")")
                # reset tool, verb, and obj
                tool = " "
                verb = " "
                obj = " "


        #print(MPs)
        # If no MP was found, print an error message with the un-translated frames
        if len(MPs) < 3:
            print("Help! No MP found.")
            print(MPs)
            print("".join(currState) + " " + "".join(nextState))
        #print("\n")
        MPList.append(MPs)
    #print(MPList)
    return MPList





# MAIN -------------------------------------------------------------------------
# Get task from command line
try:
    task=sys.argv[1]
    #print(task)
except:
    print("Error: invalid task\nUsage: python context_to_motion_primitives.py <task>\nTasks: Suturing, Needle_Passing, Knot_Tying, Pea_on_a_Peg, Post_and_Sleeve, Wire_Chaser_I, Peg_Transfer")
    sys.exit()

# Directories
baseDir = os.path.dirname(os.path.dirname(os.getcwd()))
# Transcript and video directories
# taskDir = os.path.join(baseDir, "Datasets", "dV", task)
taskDir = os.path.join(baseDir, "Datasets", "DCS", task)
transcriptDir = os.path.join(taskDir,"transcriptions")
#gestureDir = os.path.join(taskDir,"gestures")
mpDir = os.path.join(taskDir, "motion_primitives_baseline")


# Based on task, create the context state list
if (task == "Suturing") or (task == "Needle_Passing"):
    context = GContext + SNPContext
elif (task == "Knot_Tying"):
    context = GContext + KTContext
elif (task == "Peg_Transfer") or (task == "Post_and_Sleeve"):
    context = GContext + PegContext
elif (task == "Pea_on_a_Peg"):
    context = GContext + PoaPContext
else:
    context = GContext
#print(context)


# For each transcript
transcripts = glob.glob(transcriptDir+"/*.txt")
for transcript in transcripts:  #[0:5]:   #[0:1]:
    # get file name of transcript
    trial = transcript.split("/")[-1]
    print(trial)

    # Read in transcript
    with open(transcript, 'r') as t:
        df = pd.read_csv(transcript, delimiter = " ", header=None)

        # Time step between context labels
        timeStep = df.iloc[1, 0] - df.iloc[0, 0]
        #print(timeStep)

        # condense all context var columns into a string in one column
        dfContext = condenseContext(df)

        # group rows with same context label
        dfGrouped = group(dfContext)
        print(dfGrouped)

        # convert to a numpy array for the next step of processing
        npGrouped = dfGrouped.to_numpy()
        # label motion primitives
        npMPs = labelMPs(npGrouped)

        outMP = os.path.join(mpDir, trial)

        if not os.path.exists(mpDir):
            os.makedirs(mpDir)

        with open(outMP, 'w') as o:
            header = ["Start", "Stop", "Motion Primitive"]
            o.write(' '.join(header) + '\n')
            for n in npMPs:
                o.write(' '.join(n))
                o.write('\n')
        #print(npMPs)







# EOF




























# Old code because I hate just deleting stuff...

'''
for s in xrange(len(currState)):
    if currState[s] < nextState[s]:
        if s == 0:
            tool = "L"
            verb = "Grasp"
            obj = objects[int(nextState[s])]
        elif s == 1:
            tool = "L"
            verb = "Touch"
            obj = objects[int(nextState[s])]
        elif s == 2:
            tool = "R"
            verb = "Grasp"
            obj = objects[int(nextState[s])]
        elif s == 3:
            tool = "R"
            verb = "Touch"
            obj = objects[int(nextState[s])]
    if currState[s] > nextState[s]:
'''

'''
if (len(diff)>1) and (diff!=[0,1]) and (diff!=[2,3]):
    if (task == "Suturing") and (diff!=[4,5]):
        print("Help! More than one state changed!")
'''

#print(currState, nextState, c)

'''
# Determine gesture based on which state changed and populate with tool and object
# The above line should ensure that only one of the following will run

# Process general context states
# if contact to hold transition
if (c==[0,1]) or (c==[2,3]):
    tool = context[c[0]][0]
    if currState[c[0]] < nextState[c[0]]:
        verb = "Grasp"
        obj = objects[int(nextState[c[0]])]
    elif currState[c[0]] > nextState[c[0]]:
        verb = "Release"
        obj = objects[int(currState[c[0]])]
    else:
        print("Help!")
    #print(str(npGrouped[g,0]) + " " + str(npGrouped[g,1]) + " " + verb + "(" + tool + ", " + obj + ")")


# if hold state changes
if (c==[0]) or (c==[2]):
    tool = context[c[0]][0]   # letter representing L/R tool
    # determine if grasp or release based on if currState[c] increases or decreases to nextState[c]
    if currState[c[0]] < nextState[c[0]]:
        verb = "Grasp"
        obj = objects[int(nextState[c[0]])]   # get the name of the object
    elif currState[c[0]] > nextState[c[0]]:
        verb = "Release"
        obj = objects[int(currState[c[0]])]   # get the name of the object
    else:
        print("Help!")
    #print(str(npGrouped[g,0]) + " " + str(npGrouped[g,1]) + " " + verb + "(" + tool + ", " + obj + ")")


# if contact state changes
if (c==[1]) or (c==[3]):
        tool = context[c[0]][0]   # letter representing L/R tool
        # determine if touch or untouch
        if currState[c[0]] < nextState[c[0]]:
            verb = "Touch"
            obj = objects[int(nextState[c[0]])]
        elif currState[c[0]] > nextState[c[0]]:
            verb = "Untouch"
            obj = objects[int(currState[c[0]])]
        else:
            print("Help!")
        #print(str(npGrouped[g,0]) + " " + str(npGrouped[g,1]) + " " + verb + "(" + tool + ", " + obj + ")")



# Process context-dependent parts of
if (task == "Suturing") or (task == "Needle_Passing"):
    # if both needle and thread states change, pulling suture after throwing a stitch
    if c == [4, 5]:
        # tool is which ever tool(s) has the needle
        if currState[0] == "2":
            tool = context[0][0]   # left tool
        elif currState[2] == "2":
            tool = context[2][0]   # right tool
        else:
            print("Help! Nothing holding needle")

        # if needle goes from in fabric to out of fabric and thread goes from loose to taut
        if (currState[c[0]] > nextState[c[0]]) and (currState[c[1]] < nextState[c[1]]):
            verb = "Pull"
            obj = objects[int("2")]
        else:
            print("Help! Something weird")




    # if needle state changes
    if c == [4]:
        # tool is which ever tool(s) has the needle
        if currState[0] == "2":
            tool = context[0][0]   # left tool
        elif currState[2] == "2":
            tool = context[2][0]   # right tool
        else:
            print("Help! Nothing holding needle")

        # figure out if pushing needle in or pulling needle out of fabric
        if currState[c[0]] < nextState[c[0]]:
            verb = "Push"
            obj = objects[int("2")]   # needle
        elif currState[c[0]] > nextState[c[0]]:
            verb = "Pull"
            obj = objects[int("2")]
        else:
            print("Help!")
'''

    # if thread state changes
    #
    # But the thread could be being pulled by the tool holding the
    # needle pulling the thread or a tool holding a thread, or even
    # just a tool touching the thread, so which is it?
    #
'''
    if c == [5]:
        # tool is which ever tool(s) has the thread
        if currState[0] == "3":
            tool = context[0][0]   # left tool
        elif currState[2] == "3":
            tool = context[2][0]   # right tool
        else:
            print("Help! Nothing holding thread")

        # figure out if tightening or loosening
        if currState[c[0]] < nextState[c[0]]:
            verb = "Pull"
            obj = objects[int("3")]
        elif currState[c[0]] > nextState[c[0]]:
            verb = "ReleaseThread"
            obj = objects[int("3")]
'''
'''
    if c == [5]:
        # tool could be the one pulling tne needle pulling the thread
        # or the tool pulling the thread, or the tool touching the thread
        # if the left tool is holding the thread or needle and the right
        # tool is not holding anything

        if (currState[0:4] == "3000") or (currState[0:4] == "2000"):
            tool = context[0][0]   # left tool
        # if the right tool is holding the thread or needle and left is free
        elif (currState[0:4] == "0030") or (currState[0:4] == "0020"):
            tool = context[2][0]   # right tool
        # else if error?
        elif (currState[0:4] == "0000"):
            print("Help! Nothing holding thread")

        # else both hands pulling thread
        else:
            tool = "LR"   # both tools pulling thread

        # figure out if tightening or loosening
        if currState[c[0]] < nextState[c[0]]:
            verb = "Pull"
            obj = objects[int("3")]
        elif currState[c[0]] > nextState[c[0]]:
            verb = "ReleaseThread"
            obj = objects[int("3")]
'''


'''
if (task == "Knot_Tying"):
    # if cloop state changes
    if c == [4]:
        pass


    # if knot state changes
    if c == [5]:
        pass




if (task == "Pea_on_a_Peg"):
    # if peas go from stuck together to not stuck together
    if c == [4]:

        # if right hand holding peas and left hand touches to separate
        if (currState[0:4] == "0110"):
            tool = context[0][0]   # left tool
        # if left hand holding peas and right hand touches to separate
        elif (currState[0:4] == "1001"):
            tool = context[2][0]   # right tool
        # else if error?
        else:
            print("Help! Peas")

        # deduce peas state and if they are separated or not
        if currState[c[0]] < nextState[c[0]]:
            verb = "Separate"
            obj = objects[int("1")]
        else:
            print("Help! Peas")






if (tool == " ") or (verb == " ") or (obj == " "):
    print("Error, unfilled gesture!")

#print(str(npGrouped[g,0]) + " " + str(npGrouped[g, 1]+9) + " " + verb + "(" + tool + ", " + obj + ")")
#gesture = [str(npGrouped[g,0]), str(npGrouped[g, 1]+9), verb + "(" + tool + ", " + obj + ")"]
#print(gesture)
#gestureList.append(gesture)
#print("\n")
'''
