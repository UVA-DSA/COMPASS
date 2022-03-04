# Takes averages of accuracy, edit score, loss, and f scores from logs.

import pandas as pd
import numpy as np
import os
import sys
import glob


# Path to logs, but is changed in the loop
#path = '/home/student/Documents/Research/MICCAI_2022/TCN/DESK/tcn/test_1/log/train_test_result.csv'
#print(path)
#tb=pd.read_csv(path)
#print(tb)

#tb.columns
#print(tb.columns)


# Calculate average metrics in each fold's train_test_results.csv given a path to a model's results folder,
def analyze(modelpath):
    print("Starting analysis")

    modelpath = os.path.join(modelpath, "tcn")
    tests = os.listdir(modelpath)

    #print('v_accuracy, \t v_edit_score, \t  v_loss, \t v_f_scores_10, \t v_f_scores_25, \t v_f_scores_50, \t v_f_scores_75')
    result = []

    for test in tests:
        # Path to log files
        path =  os.path.join(modelpath, test, "log/train_test_result.csv")
        #print(path)

        # Read CSV
        tb=pd.read_csv(path)
        #print(tb)

        # Average results
        vals = [tb['v_accuracy'].rolling(window=5).mean().iloc[-1], \
        tb['v_edit_score'].rolling(window=5).mean().iloc[-1], \
        tb['v_loss'].rolling(window=5).mean().iloc[-1], \
        tb['v_f_scores_10'].rolling(window=5).mean().iloc[-1], \
        tb['v_f_scores_25'].rolling(window=5).mean().iloc[-1], \
        tb['v_f_scores_50'].rolling(window=5).mean().iloc[-1], \
        tb['v_f_scores_75'].rolling(window=5).mean().iloc[-1]]
        result.append(vals)
        #print(vals)


    result = np.array(result)
    #print(result)

    # Print results
    # print("v_accuracy: " + str(np.mean(result,axis = 0)[0]))
    # print("v_edit_score: " + str(np.mean(result,axis = 0)[1]))
    # print("v_loss: " + str(np.mean(result,axis = 0)[2]))
    # print("v_f_scores_10: " + str(np.mean(result,axis = 0)[3]))
    # print("v_f_scores_25: " + str(np.mean(result,axis = 0)[4]))
    # print("v_f_scores_50: " + str(np.mean(result,axis = 0)[5]))
    # print("v_f_scores_75: " + str(np.mean(result,axis = 0)[6]))
    #
    # print("v_accuracy: " + str(np.std(result,axis = 0)[0]))
    # print("v_edit_score: " + str(np.std(result,axis = 0)[1]))
    # print("v_loss: " + str(np.std(result,axis = 0)[2]))

    v_acc = np.mean(result,axis = 0)[0]
    v_edit_score = np.mean(result,axis = 0)[1]
    v_loss = np.mean(result,axis = 0)[2]
    v_f_scores_10 = np.mean(result,axis = 0)[3]
    v_f_scores_25 = np.mean(result,axis = 0)[4]
    v_f_scores_50 = np.mean(result,axis = 0)[5]
    v_f_scores_75 = np.mean(result,axis = 0)[6]
    avgResults = [v_acc, v_edit_score, v_loss, v_f_scores_10, v_f_scores_25, v_f_scores_50, v_f_scores_75]

    #v_accuracy: " + str(np.std(result,axis = 0)[0]
    #v_edit_score: " + str(np.std(result,axis = 0)[1]
    #v_loss: " + str(np.std(result,axis = 0)[2]

    return avgResults

'''
tcn : v_accuracy: {}, v_edit_score: {}, v_loss: {}, v_f_scores_10: {}, v_f_scores_25: {}, v_f_scores_50: {}, v_f_scores_75: {}'.format(np.mean(result,axis = 0)[0],\\
   np.mean(result,axis = 0)[1],np.mean(result,axis = 0)[2],np.mean(result,axis = 0)[3],np.mean(result,axis = 0)[4],np.mean(result,axis = 0)[5],np.mean(result,axis = 0)[6])


tcn : v_accuracy: {} ({}), v_edit_score: {} ({}), v_loss: {} ({}), v_f_scores_10: {} ({}), v_f_scores_25: {} ({}), v_f_scores_50: {} ({}), v_f_scores_75: {} ({})'.format(round(np.mean(result,axis = 0)[0]),\\
   round(np.std(result,axis = 0)[0],2)
   round(np.mean(result,axis = 0)[1],2), round(np.std(result,axis = 0)[1],2),round(np.mean(result,axis = 0)[2],2),round(np.std(result,axis = 0)[2],2),round(np.mean(result,axis = 0)[3],2)\\
      ,round(np.std(result,axis = 0)[3],2),round(np.mean(result,axis = 0)[4],2),\\
      round(np.std(result,axis = 0)[4],2),round(np.mean(result,axis = 0)[5],2),round(np.std(result,axis = 0)[5],2),\\
         round(np.mean(result,axis = 0)[6],2),round(np.std(result,axis = 0)[6],2))
'''


if __name__ == "__main__":

    #path = "/home/student/Documents/Research/MICCAI_2022/COMPASS-main/TCN/Results/results_03_02_22_5ab_noROSMA/All-5a_velocity_MP_LOSO_03_02_2022_1420"

    # Set up paths and directories
    dir = os.getcwd()
    resultsDir = os.path.join(dir, "Results", "results_03_03_22_5ab_ROSMA")
    resultsList = os.listdir(resultsDir)

    # Find folders containing "tcn" folder from training
    for r in resultsList:
        resultpath = os.path.join(resultsDir, r)
        # If tcn folder found, run analysis
        if "tcn" in os.listdir(resultpath):
            print(r)
            results = analyze(resultpath)
            print(results[0:2])
    #sys.exit()

    #analyze(path)


'''

result = []

for i in range(1,6):
    path = './DESK/lstm/test_{}/log/train_test_result.csv'.format(i)
    tb=pd.read_csv(path)

    vals = [tb['v_accuracy'].rolling(window=5).mean().iloc[-1]
    tb['v_edit_score'].rolling(window=5).mean().iloc[-1]
    tb['v_loss'].rolling(window=5).mean().iloc[-1]
    tb['v_f_scores_10'].rolling(window=5).mean().iloc[-1]
    tb['v_f_scores_25'].rolling(window=5).mean().iloc[-1]
    tb['v_f_scores_50'].rolling(window=5).mean().iloc[-1]
    tb['v_f_scores_75'].rolling(window=5).mean().iloc[-1]
    result.append(vals)



lstm : v_accuracy: {} ({}), v_edit_score: {} ({}), v_loss: {} ({}), v_f_scores_10: {} ({}), v_f_scores_25: {} ({}), v_f_scores_50: {} ({}), v_f_scores_75: {} ({})'.format(round(np.mean(result,axis = 0)[0]),\\
   round(np.std(result,axis = 0)[0],2)
   round(np.mean(result,axis = 0)[1],2), round(np.std(result,axis = 0)[1],2),round(np.mean(result,axis = 0)[2],2),round(np.std(result,axis = 0)[2],2),round(np.mean(result,axis = 0)[3],2)\\
      ,round(np.std(result,axis = 0)[3],2),round(np.mean(result,axis = 0)[4],2),\\
      round(np.std(result,axis = 0)[4],2),round(np.mean(result,axis = 0)[5],2),round(np.std(result,axis = 0)[5],2),\\
         round(np.mean(result,axis = 0)[6],2),round(np.std(result,axis = 0)[6],2))




#lstm : v_accuracy: {}, v_edit_score: {}, v_loss: {}, v_f_scores_10: {}, v_f_scores_25: {}, v_f_scores_50: {}, v_f_scores_75: {}'.format(,\\
#,,,,,)
'''
