# Sara Liu, 5/9/2022
import csv
import os

import pandas as pd

baseDir = "F:/Downloads/UVA/Fall 2022/CS 4980 Capstone Research"


def filterRavenData():
    fromDir = baseDir + "/Synchronized and Resampled Raven Kinematic Data"
    toDir = baseDir + "/Filtered Raven Kinematic Data"
    isExist = os.path.exists(toDir)
    if not isExist:
        os.makedirs(toDir)
    # print("os.listdir(fromDir)", os.listdir(fromDir))
    for file in os.listdir(fromDir):
        print("file", file)
        fileName = os.path.join(fromDir, file)
        # with open(fileName, 'r') as fileRead:
        #     r = csv.reader(fileRead)
        #     numStart = file.rindex("_") + 1
        #     numEnd = file.rindex(".csv")
        #     num = int(file[numStart:numEnd])
        #     print("num", num)
        #     newCSVFile = "filtered_raven_kinematic_" + str(num) + ".csv"
        #     print("newCSVFile", newCSVFile)
        #     newCSVFileName = os.path.join(toDir, newCSVFile)
        #
        #     with open(newCSVFileName, "w") as fileWrite:
        #         w = csv.writer(fileWrite)
        #         header = ["Relative frame #", "PSML_position_x", "PSML_position_y", "PSML_position_z", "PSMR_position_x", "PSMR_position_y", "PSMR_position_z", "PSML_gripper_angle", "PSMR_gripper_angle"]
        #         w.writerow(header)
        #
        #         for row in r:
        #             print("row", row)
        df = pd.read_csv(fileName)
        # print("df", df)
        newdf = df[["Unnamed: 0", "field.pos0", "field.pos1", "field.pos2", "field.pos3", "field.pos4", "field.pos5",
                    "field.grasp_d0", "field.grasp_d1", "epoch_ms"]]
        # print("newdf", newdf)
        newdf = newdf.rename(columns={"Unnamed: 0": "Frame", "field.pos0": "PSML_position_z",
                                      "field.pos1": "PSML_position_x", "field.pos2": "PSML_position_y",
                                      "field.pos3": "PSMR_position_x", "field.pos4": "PSMR_position_z",
                                      "field.pos5": "PSMR_position_y", "field.grasp_d0": "PSML_gripper_angle",
                                      "field.grasp_d1": "PSMR_gripper_angle", "epoch_ms": "Timestamp"})
        # print("newdf", newdf)
        cols = newdf.columns.tolist()
        # print("cols", cols)
        cols = [cols[0]] + [cols[9]] + [cols[2]] + [cols[3]] + [cols[1]] + [cols[4]] + [cols[6]] + [cols[5]] + [cols[7]] + [cols[8]]
        # print("cols", cols)
        newdf = newdf[cols]
        # print("newdf", newdf)
        newdf["PSML_position_y"] = newdf["PSML_position_y"].apply(lambda x: x * -1)
        newdf["PSML_position_z"] = newdf["PSML_position_z"].apply(lambda x: x * -1)
        newdf["PSMR_position_x"] = newdf["PSMR_position_x"].apply(lambda x: x * -1)
        newdf["PSMR_position_z"] = newdf["PSMR_position_z"].apply(lambda x: x * -1)
        # print("newdf", newdf)

        numStart = file.rindex("_") + 1
        numEnd = file.rindex(".csv")
        num = int(file[numStart:numEnd])
        # print("num", num)
        newCSVFile = "filtered_raven_kinematic_" + str(num) + ".csv"
        # print("newCSVFile", newCSVFile)
        newCSVFileName = os.path.join(toDir, newCSVFile)
        newdf.to_csv(newCSVFileName, index=False, header=True)


if __name__ == "__main__":
    filterRavenData()
