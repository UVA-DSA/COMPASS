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
def analyze(modelpathin):
    #print("Starting analysis")

    modelpath = os.path.join(modelpathin, "tcn")
    # list all cross val folds (tests)
    tests = os.listdir(modelpath)

    #print('v_accuracy, \t v_edit_score, \t  v_loss, \t v_f_scores_10, \t v_f_scores_25, \t v_f_scores_50, \t v_f_scores_75')
    result = []
    modelresultsdF = pd.DataFrame(columns = ["Fold", "Accuracy", "Edit Score"])

    for test in tests:
        # Path to log files
        path =  os.path.join(modelpath, test, "log/train_test_result.csv")
        print(path)

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
        print(vals)
        newresult = {"Fold": test, "Accuracy": vals[0], "Edit Score": vals[1]}
        modelresultsdF = modelresultsdF.append(newresult, ignore_index=True)


    result = np.array(result)
    #print(result)
    # Save results of each fold to a csv in the model's folder
    modelresultsdF = modelresultsdF.sort_values("Fold")
    resultCSV = os.path.join(modelpathin, "tests.csv")
    modelresultsdF.to_csv(resultCSV, sep="\t", index=False)

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
    resultsDir = os.path.join(dir, "Results") #, "results_07_26_2022_fixedMPhypparams") #, "results_07_25_2022_fixedgesturehypparams")  #, "results_03_04_22")
    resultsList = os.listdir(resultsDir)


    resultsdF = pd.DataFrame(columns = ["Data", "Variables", "Labels", "Cross Val", "Accuracy", "Edit Score"])
    #print(resultsdF)

    # Find folders containing "tcn" folder from training
    for r in resultsList:
        try:
            resultpath = os.path.join(resultsDir, r)
            # If tcn folder found, run analysis
            if "tcn" in os.listdir(resultpath):
                print(r)   # print model name (a.k.a. folder name)
                results = analyze(resultpath)
                print(results[0:2])   # print accuracy and edit score for this model
                name = r.split("_")
                data = name[0]
                vars = name[1]
                labels = name[2]
                crossval = name[3]
                acc = results[0]
                es = results[1]
                newresult = {"Data": data, "Variables": vars, "Labels": labels, "Cross Val": crossval, "Accuracy": acc, "Edit Score": es}
                resultsdF = resultsdF.append(newresult, ignore_index=True)
                print("\n")
        except:
            print("Failed: " + r + "\n")
    print(resultsdF)
    resultCSV = os.path.join(resultsDir, "summary.csv")
    resultsdF.to_csv(resultCSV, sep="\t", index=False)
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
