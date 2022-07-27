# Kay Hutchinson 3/14/22
# Create confusion matrix from saved predicted files and calculate class accuracies.

# Takes averages of accuracy, edit score, loss, and f scores from logs.

import pandas as pd
import numpy as np
import os
import sys
import glob
import seaborn as sns
import matplotlib.pyplot as plt




# Get class TP, TN, FP, FN from a .npy file by creating a confusion matrix
def getCM(modelpath):
    modelpath = os.path.join(modelpath, "tcn")
    tests = os.listdir(modelpath)

    #print('v_accuracy, \t v_edit_score, \t  v_loss, \t v_f_scores_10, \t v_f_scores_25, \t v_f_scores_50, \t v_f_scores_75')
    result = []

    # confusion matrix for entire model
    folddF = pd.DataFrame()


    # For each fold
    for test in tests: #[3:4]:
        print(test)
        # Path to log files
        logpath =  os.path.join(modelpath, test, "log")
        epochs = [f for f in os.listdir(logpath) if os.path.isdir(os.path.join(logpath, f))] #os.listdir(logpath)


        # Dataframe to hold class accuracy results
        classResults = pd.DataFrame()
        # Dataframe to hold class confusion matrix (just sum, no averaging or anything)
        classdF = pd.DataFrame()

        # For each epoch
        for epoch in epochs:
            #print(epoch.split("_")[-1])
            epochpath =  os.path.join(logpath, epoch)

            # dF to hold epoch results for confusion matrices
            epochdF = pd.DataFrame()

            # For each test_*_pred_gt.npy file:
            vals = os.listdir(epochpath)
            for v in vals:
                vpath = os.path.join(epochpath, v)
                #print(vpath)
                # Read the .npy file
                [data,pred,gt]=np.load(vpath ,allow_pickle=True)
                cf = pd.crosstab(gt, pred, rownames=['Actual'], colnames=['Predicted'])

                # Add confusion matrices together
                epochdF = uadd(epochdF, cf)

            # Add epochdF to classdF
            classdF = uadd(classdF, epochdF)
            # Analyze the confusion matrix and calculate accuracies for each class
            classAccs = analyzecF(epochdF)
            # Rename index number as epoch number
            classAccs = classAccs.rename(index={0:int(epoch.split("_")[-1])})
            # Append epoch results to fold results
            classResults = classResults.append(classAccs) #classResults.append(classAccs, index=epoch)#[epoch,:] = classAccs
        # Add classdF to folddF
        folddF = uadd(folddF, classdF)
        # sort by epoch index number and save to csv
        classResults = classResults.sort_index()
        #print(classResults)
        # save classResults as a csv in the fold's folder
        savePath = os.path.join(logpath, "class_results.csv")
        classResults.to_csv(savePath, sep="\t", index=True)
    print(folddF)
    # Save model confusion matrix to csv
    cfsavepath = os.path.join(os.path.dirname(modelpath), "class_cf.csv")
    #print(cfsavepath)
    folddF.to_csv(cfsavepath, sep=",", index=True)




# Add two confusion matrices and take union of rows and columns
def uadd(df1, df2):
    # Sum the two dataframes with the union of the rows and columns
    rows = df1.index.union(df2.index)
    cols = df1.columns.union(df2.columns)

    df1_new = df1.reindex(index=rows, columns=cols).fillna(0).astype(int)
    df2_new = df2.reindex(index=rows, columns=cols).fillna(0).astype(int)
    return df2_new.add(df1_new)




# Given a confusion matrix dataframe, calculate TP, TN, FP, FN and then
# precision, recall, F1 score and accuracy for each class and dataframe of classes and accuracies
def analyzecF(cF):
        #print(cF)
        # Actual classes are on the rows, so get row names of classes
        classes = list(cF.index)

        # Fill in confusion matrix with zeros for classes that are missing
        actuals = list(cF.columns)
        for cl in classes:
            if cl not in actuals:
                cF[cl] = 0
        #print(cF)
        #print(classes)

        # df to hold results
        metricsdF = pd.DataFrame(columns=classes)

        # For each class, calculate TP, TN, FP, FN
        for cl in classes:
            #print(cl)

            # Print class row
            row = cF.loc[cl,:]
            # Print class column
            col = cF.loc[:,cl]
            # Get the rest of the confusion matrix without the class row and column
            notcl = cF.drop([cl], axis=1)
            notcl = notcl.drop([cl], axis=0)
            #print(notcl)

            # Calculate TP, TN, FP, FN
            TP = row.loc[cl]
            #print(TP)
            TN = notcl.sum().sum()
            #print(TN)
            FP = col.sum()-TP
            #print(FP)
            FN = row.sum()-TP
            #print(FN)

            # Calculate precision and recall
            precision = TP/(TP+FP)
            #print(precision)
            recall = TP/(TP+FN)
            #print(recall)

            # Calculate F1 score
            F1 = 2*(precision*recall)/(precision + recall)
            #print(F1)

            # Calculate accuracy
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            #print(accuracy)
            #print("\n")

            # Append metric to results df
            metricsdF[cl] = [accuracy]
            #print(metricsdF)


        #print(metricsdF)
        return metricsdF





# Calculate average metrics in each fold's class_results.csv given a path to a model's results folder,
def analyze(modelpath):
    #print("Starting analysis")

    modelpath = os.path.join(modelpath, "tcn")
    tests = os.listdir(modelpath)

    #print('v_accuracy, \t v_edit_score, \t  v_loss, \t v_f_scores_10, \t v_f_scores_25, \t v_f_scores_50, \t v_f_scores_75')
    result = pd.DataFrame()

    for test in tests:
        # Path to log files
        path =  os.path.join(modelpath, test, "log/class_results.csv")
        #print(path)

        # Read CSV
        tb=pd.read_csv(path, sep="\t")
        #print(tb)

        vals = tb.rolling(5).mean().iloc[-1, 1:]
        result = result.append(vals)
        #print(vals)

    avgResults = pd.DataFrame(result.mean(0)).transpose()
    #print(avgResults.transpose())
    return avgResults


# Visualize a confusion matrix given a path to the csv
def plotcf(cfpath):

    # Read in a confusion matrix
    cf = pd.read_csv(cfpath, index_col=0)
    fig = plt.figure()
    # Raw values
    #ax = sns.heatmap(cf, annot=True, cmap='Blues')
    # Percentages
    #ax = sns.heatmap(cf/np.sum(np.sum(cf)), annot=True, fmt='.1%', cmap='Blues')

    cm =  (cf.astype('float').T / cf.sum(axis=1)).T  # normalize along row
    #cm =  (cf.astype('float').T / cf.sum().sum()).T  # normalize all
    ax = sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues')
    title = cfpath.split("/")[-2]
    xlabels = cf.columns
    ylabels = cf.index

    ax.set_title(title);
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(xlabels)
    ax.yaxis.set_ticklabels(ylabels)

    ## Display the visualization of the Confusion Matrix.
    plt.show()

    # Save
    #resultsDir = os.path.dirname(os.path.dirname(os.path.dirname(cfpath)))
    resultsDir = os.path.dirname(os.path.dirname(cfpath))
    savePath = os.path.join(resultsDir, "Figures", title + ".jpg")
    print(savePath)
    fig.savefig(savePath)
    #sys.exit()











if __name__ == "__main__":

    # Set up paths and directories
    dir = os.getcwd()
    resultsDir = os.path.join(dir, "Results") #, "results_06_17_2022_gesture_test") #, "results_06_16_2022_nofabric") #, "results_05_11_2022_MPvars") #, "results_03_04_22")
    #resultsList = os.listdir(resultsDir)
    resultsList = [f for f in os.listdir(resultsDir) if os.path.isdir(os.path.join(resultsDir, f))]



    '''
    # This section takes the longest to process and run, so comment it out if it's run before
    # Find folders containing "tcn" folder from training
    # Only need to run the following section once to generate class_results.csv
    # can take a while to process, too
    for r in resultsList:
        resultpath = os.path.join(resultsDir, r)
        # If tcn folder found, run analysis
        if "tcn" in os.listdir(resultpath):
            print(r)
            getCM(resultpath)
    '''



    # Analyze class_results.csv files; modified from calculate_mean_cv.py
    #resultsdF = pd.DataFrame(columns = ["Data", "Variables", "Labels", "Cross Val", "Accuracy"])
    resultsdF = pd.DataFrame(columns = ["Data", "Variables", "Labels", "Cross Val"])
    # Find folders containing "tcn" folder from training
    for r in resultsList:
        try:
            resultpath = os.path.join(resultsDir, r)
            # If tcn folder found, run analysis
            if "tcn" in os.listdir(resultpath):
                print(r)

                results = analyze(resultpath)
                name = r.split("_")
                data = name[0]
                vars = name[1]
                labels = name[2]
                crossval = name[3]
                #acc = results[0]
                newresult = {"Data": [data], "Variables": [vars], "Labels": [labels], "Cross Val": [crossval]}
                newdf = pd.DataFrame(newresult)

                # Create df for this model's results
                df1 = newdf
                df2 = results
                newdf = pd.concat([df1, df2], 1).fillna(0)
                #print(newdf)

                # Append to resultsdF
                cols = resultsdF.columns.union(newdf.columns)
                #print(cols)

                df1_new = newdf.reindex(columns=cols)
                df2_new = resultsdF.reindex(columns=cols)
                resultsdF = pd.concat([df2_new, df1_new])  #df2_new.concat(df1_new)
                #print(resultsdF)
        except:
            print("Failed: " + r + "\n")


    #resultsdF = resultsdF[["Data", "Variables", "Labels", "Cross Val", "G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "Grasp", "Release", "Touch", "Untouch", "Pull", "Push", "Move"]]
    #resultsdF = resultsdF[["Data", "Variables", "Labels", "Cross Val", "Grasp", "Release", "Touch", "Untouch", "Push"]]
    #resultsdF = resultsdF[["Data", "Variables", "Labels", "Cross Val", "G1", "G2", "G3", "G4", "G5", "G6", "G8"]]

    print(resultsdF)
    resultCSV = os.path.join(resultsDir, "summary_class.csv")
    resultsdF.to_csv(resultCSV, sep="\t", index=False)


    # Plot and visualize confusion matrices
    for r in resultsList:
        resultpath = os.path.join(resultsDir, r)
        # If class_cf.csv found, run analysis
        if "class_cf.csv" in os.listdir(resultpath):
            cfpath = os.path.join(resultpath, "class_cf.csv")
            plotcf(cfpath)

















# EOF
