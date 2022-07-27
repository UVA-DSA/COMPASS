# Kay Hutchinson 4/14/2021
#
# Combine Touch with Grasp into Grasp and Untouch and Release into Release
#
# 5/9/22 Added lines to process cases where multiple MPs are on one transcript line
#
# 5/10/22 Extend Push/Pull to include only the next Idle for the L/R split sets
#
# 6/15/2022 Generate transition probability graph based on data and visualize


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
                try:
                    obj = line[3].split(")")[0]
                except:
                    obj = "A"

                # get next MP from next line
                linen = lines[i+1].split()
                mpn = linen[2].split("(")[0]
                handn = linen[2].split("(")[1]
                try:
                    objn = linen[3].split(")")[0]
                except:
                    objn = "A"


                # # check for Untouch and Pull
                # if (mp == "Untouch") and (mpn == "Pull"):
                #     print(line)
                #     print(linen)



                # join touch and grasp into grasp; check that hand and object are the same
                if (mp == "Idle"):
                    start = line[0]
                    end = line[1]
                    label = " ".join(line[2:])

                elif (mp == "Touch") and (mpn == "Grasp") and (hand == handn) and (obj == objn):
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
                #print(line)
                # get MP from line
                mp = line[2].split("(")[0]
                hand = line[2].split("(")[1]
                #obj = line[3].split(")")[0]
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


# Function to combine [Touch,Push]->Push
def combineMPs2(mpDir):
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
                try:
                    obj = line[3].split(")")[0]
                except:
                    obj = "A"

                # get next MP from next line
                linen = lines[i+1].split()
                mpn = linen[2].split("(")[0]
                handn = linen[2].split("(")[1]
                try:
                    objn = linen[3].split(")")[0]
                except:
                    objn = "A"

                # join touch and grasp into grasp; check that hand and object are the same
                if (mp == "Idle"):
                    start = line[0]
                    end = line[1]
                    label = " ".join(line[2:])

                elif (mp == "Touch") and (mpn == "Push") and (hand == handn) and (obj == objn):
                    start = line[0]
                    end = linen[1]
                    label = " ".join(linen[2:])
                    i = i+1
                    #print("Joining touch and grasp")
                    #print(label)

                elif (mp == "Untouch") and (mpn == "Pull") and (hand == handn) and (obj == objn):
                    print(line)

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
                    if (mp == "Touch") and (mp2 == "Push") and (hand == hand2) and (obj == obj2):
                        start = line[0]
                        end = line[1]
                        label = " ".join(line[4:])
                        i = i+1
                        #print("Joining touch and grasp")
                        #print(label)
                    elif (mp == "Untouch") and (mpn == "Pull") and (hand == handn) and (obj == objn):
                        print(line)



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
                #print(line)
                # get MP from line
                mp = line[2].split("(")[0]
                hand = line[2].split("(")[1]
                #obj = line[3].split(")")[0]
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



# Function to extend [Push,Idle]->Push and [Pull,Idle]->Pull for the L/R split sets
def extendMPs(mpDir):
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
                #obj = line[3].split(")")[0]

                # get next MP from next line
                linen = lines[i+1].split()
                mpn = linen[2].split("(")[0]
                handn = linen[2].split("(")[1]
                #objn = linen[3].split(")")[0]

                # join touch and grasp into grasp; check that hand and object are the same
                if (mp == "Push") and (mpn == "Idle"): # and (hand == handn):
                    start = line[0]
                    end = linen[1]
                    label = " ".join(line[2:])
                    i = i+1
                    #print("Joining touch and grasp")
                    #print(label)

                # join untouch and release into release; check that hand and object are the same
                elif (mp == "Pull") and (mpn == "Idle"): # and (hand == handn):
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
                    print("Help, multiple MPs")
                    print(line)
                    '''
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
                    '''


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
                #obj = line[3].split(")")[0]
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




# Given all the MP transcripts in a task, sum up the transitions in a matrix
#   To: Grasp Release Touch Untouch Push Pull   From:
# Q = [   .       .     .       .    .    .,    Grasp
#         .       .     .       .    .    .,    Release
#         .       .     .       .    .    .,    Touch
#         .       .     .       .    .    .,    Untouch
#         .       .     .       .    .    .,    Push
#         .       .     .       .    .    ., ]  Pull
def getTransitionMatrix(mpDir):
    # based on http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
    import networkx as nx
    import matplotlib.pyplot as plt
    from pprint import pprint
    import graphviz as gv

    # MPs and encodings
    MPs = ["Grasp", "Release", "Touch", "Untouch", "Push", "Pull", "Idle"]
    encMPs = {"Grasp": 0, "Release": 1, "Touch": 2, "Untouch": 3, "Push": 4, "Pull": 5, "Idle": 6}

    # Define threshold for edge probabilities to determine which edges are shown in the graph
    edgeThreshold = 0.01

    # Initialize transition matrix
    Q = np.zeros((len(MPs), len(MPs)))

    # Loop through transcripts and create Q
    files = glob.glob(mpDir+"/*.txt")
    for file in files:
        # get file name of transcript
        trial = file.split("/")[-1]
        #print(trial)

        # Read in transcript
        with open(file, 'r') as t:
            # skip first line which is a header
            lines = t.readlines()[1:]
            #print(lines)
            for i in range(len(lines)-1):
                line = lines[i].split()
                #print(line)
                # get MP from line
                mp = line[2].split("(")[0]
                #print(encMPs[mp])
                #hand = line[2].split("(")[1]
                #obj = line[3].split(")")[0]

                # get next MP from next line
                linen = lines[i+1].split()
                mpn = linen[2].split("(")[0]
                #print(encMPs[mpn])
                #handn = linen[2].split("(")[1]
                #objn = linen[3].split(")")[0]

                # Increment transition in Q matrix
                #print(Q[encMPs[mp], encMPs[mpn]])
                Q[encMPs[mp], encMPs[mpn]] = Q[encMPs[mp], encMPs[mpn]]+1
                #print(Q[encMPs[mp], encMPs[mpn]])
    #print(Q)

    # Normalize Q by row
    rowSums = Q.sum(axis=1)
    Q = Q/rowSums[:,np.newaxis]
    Q = np.nan_to_num(Q)
    #print(Q)


    # Convert np array to df
    dfQ = pd.DataFrame(Q, columns = MPs, index = MPs)
    print(dfQ)
    #print(dfQ.sum(axis=1))

    # Get edge weights
    edges_wts = _get_markov_edges(dfQ)
    #pprint(edges_wts)


    # Visualize and save figure with graphviz
    g = gv.Graph(format='png')
    dot = gv.Digraph('test')
    for k, v in edges_wts.items():
        if v > edgeThreshold:     # draw edges with a probability greater than the defined threshold
            tmp_origin, tmp_destination = k[0], k[1]
            dot.edge(tmp_origin, tmp_destination, label='%.2f'%v)
    dot.format='svg'
    dot.render()


    # # Visualize with networkx
    # # create graph object
    # G = nx.MultiDiGraph()
    #
    # # nodes correspond to states
    # G.add_nodes_from(MPs)
    # #print(G.nodes())
    #
    # # edges represent transition probabilities
    # for k, v in edges_wts.items():
    #     tmp_origin, tmp_destination = k[0], k[1]
    #     G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    # #print('Edges:')
    # #pprint(G.edges(data=True))
    #
    # pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    # #pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    #
    # nx.draw_networkx(G, pos)
    # # create edge labels for jupyter plot but is not necessary
    # edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
    # nx.drawing.nx_pydot.write_dot(G, 'test_markov.dot')
    # plt.show()






# Helper for getTransitionMatrix()
# create a function that maps transition probability dataframe
# to markov edges and weights
def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges




# Given all the gesture transcripts in a task, sum up the transitions in a matrix
#   To:  G1      G2    G3      G4   G5   ...   From:
# Q = [   .       .     .       .    .    .,    G1
#         .       .     .       .    .    .,    G2
#         .       .     .       .    .    .,    G3
#         .       .     .       .    .    .,    G4
#         .       .     .       .    .    .,    G5
#         .       .     .       .    .    ., ]  ...
def getTransitionMatrixGestures(mpDir):
    # based on http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
    import networkx as nx
    import matplotlib.pyplot as plt
    from pprint import pprint
    import graphviz as gv

    # MPs and encodings
    Gs = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15"]
    encGs = {"G1":0, "G2":1, "G3":2, "G4":3, "G5":4, "G6":5, "G7":6, "G8":7, "G9":8, "G10":9, "G11":10, "G12":11, "G13":12, "G14":13, "G15":14}

    # Define threshold for edge probabilities to determine which edges are shown in the graph
    edgeThreshold = 0.1

    # Initialize transition matrix
    Q = np.zeros((len(Gs), len(Gs)))

    # Loop through transcripts and create Q
    files = glob.glob(mpDir+"/*.txt")
    for file in files:
        # get file name of transcript
        trial = file.split("/")[-1]
        #print(trial)

        # Read in transcript
        with open(file, 'r') as t:
            lines = t.readlines()
            #print(lines)
            for i in range(len(lines)-1):
                print(i)
                line = lines[i].split()
                # get gesture from line
                g = line[2]
                # get next gesture from next line
                linen = lines[i+1].split()
                gn = linen[2]
                # Increment transition in Q matrix
                #print(Q[encMPs[mp], encMPs[mpn]])
                print(encGs[g], encGs[gn])
                Q[encGs[g], encGs[gn]] = Q[encGs[g], encGs[gn]]+1
                #print(Q[encMPs[mp], encMPs[mpn]])
    #print(Q)

    # Normalize Q by row
    rowSums = Q.sum(axis=1)
    Q = Q/rowSums[:,np.newaxis]
    Q = np.nan_to_num(Q)
    #print(Q)


    # Convert np array to df
    dfQ = pd.DataFrame(Q, columns = Gs, index = Gs)
    print(dfQ)
    #print(dfQ.sum(axis=1))

    # Get edge weights
    edges_wts = _get_markov_edges(dfQ)
    #pprint(edges_wts)


    # Visualize and save figure with graphviz
    g = gv.Graph(format='png')
    dot = gv.Digraph('test2')
    for k, v in edges_wts.items():
        if v > edgeThreshold:     # draw edges with a probability greater than the defined threshold
            tmp_origin, tmp_destination = k[0], k[1]
            dot.edge(tmp_origin, tmp_destination, label='%.2f'%v)
    dot.format='svg'
    dot.render()







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

'''
# Create gesture transition matrix
mpDir = os.path.join(taskDir, "gestures")
getTransitionMatrixGestures(mpDir)
'''

'''
# Create MP transition matrix
mpDir = os.path.join(taskDir, "motion_primitives_baseline")
getTransitionMatrix(mpDir)
'''


'''
# For combining MPs in baseline
mpDir = os.path.join(taskDir, "motion_primitives_baseline")
mpDirnew = os.path.join(taskDir, "motion_primitives_combined")
combineMPs(mpDir)

'''
# For combining MPs
mpDir = os.path.join(taskDir, "motion_primitives_L")
mpDirnew = os.path.join(taskDir, "motion_primitives_L")
combineMPs(mpDir)
#extendMPs(mpDir)
mpDir = os.path.join(taskDir, "motion_primitives_R")
mpDirnew = os.path.join(taskDir, "motion_primitives_R")
combineMPs(mpDir)
#extendMPs(mpDir)


'''
# For combining MPs
mpDir = os.path.join(taskDir, "motion_primitives_LK")
mpDirnew = os.path.join(taskDir, "motion_primitives_LE")
combineMPs(mpDir)
#extendMPs(mpDir)
'''
'''
# For creating Exchange (after combined)
mpDir = os.path.join(taskDir, "motion_primitives_combined")
mpDirnew = os.path.join(taskDir, "motion_primitives_exchange")
exchangeMPs(mpDir)
'''

'''
# For creating L/R extended Push/Pull in LE and RE
mpDir = os.path.join(taskDir, "motion_primitives_L")
mpDirnew = os.path.join(taskDir, "motion_primitives_LE")
extendMPs(mpDir)
mpDir = os.path.join(taskDir, "motion_primitives_R")
mpDirnew = os.path.join(taskDir, "motion_primitives_RE")
extendMPs(mpDir)
'''










#EOF
