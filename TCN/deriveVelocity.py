# Sara Liu, 5/9/2022
import csv
import os

import pandas as pd

baseDir = "F:/Downloads/UVA/Fall 2022/CS 4980 Capstone Research"


def deriveVelocity(from_dir, to_dir):
    isExist = os.path.exists(to_dir)
    if not isExist:
        os.makedirs(to_dir)
    # print("os.listdir(from_dir)", os.listdir(from_dir))
    for file in os.listdir(from_dir):
        print("file", file)
        fileName = os.path.join(from_dir, file)
        df = pd.read_csv(fileName)
        # print("df", df)
        if "Relative frame #" in df.columns:
            df = df.rename(columns={"Relative frame #": "Frame"})
        if "Frame timestamp" in df.columns:
            df = df.rename(columns={"Frame timestamp": "Timestamp"})
        cols = [i for i in df.columns if "position" in i]
        # print("cols", cols)
        timediff = df["Timestamp"].diff()
        for col in cols:
            newCol = col.split("_")[0] + "_velocity_" + col.split("_")[2]
            # print("newCol", newCol)
            newColVal = (df[col].diff()/timediff).rolling(3).mean()
            df[newCol] = newColVal
            df = df.dropna()

        numStart = 0
        numEnd = 0
        if "Raven" in from_dir:
            numStart = file.rindex("_") + 1
            numEnd = file.rindex(".csv")
        elif "trakStar" in from_dir:
            numStart = file.rindex("_T") + 2
            numEnd = file.index("_trakStar_final.csv")
        num = file[numStart:numEnd].zfill(2)
        # print("num", num)
        newCSVFile = "Peg_Transfer_S01_T" + num + ".csv"
        # print("newCSVFile", newCSVFile)
        newCSVFileName = os.path.join(to_dir, newCSVFile)
        df.to_csv(newCSVFileName, index=False, header=True)
        # if num == "02":
        #     print(timediff)


if __name__ == "__main__":
    fromDir = baseDir + "/Filtered Raven Kinematic Data"
    toDir = baseDir + "/Final Raven Data"
    deriveVelocity(fromDir, toDir)
    fromDir = baseDir + "/Synchronized Parsed trakStar Kinematic Data"
    toDir = baseDir + "/Final DCS Data"
    deriveVelocity(fromDir, toDir)
