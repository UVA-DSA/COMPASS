To do: Add code and instructions for using TCN model.

Go into config.json and change "raw_feature_dir" to the correct paths to each task's 'preprocessed' folder.
In the terminal run
```
python preprocess.py <set> <var> <labeltype>
```
Options for set are: DESK, JIGSAWS
Options for var are: velocity, orientation, all 
Option for labeltype are: MP, gesture

Then, to tune or train, run:
```
python tune.py <set> <var> <labeltype>
```
or 
```
python train_test_val.py <set> <var> <labeltype>
```
Note: You must run preprocess.py before tune.py or train_test_val.py in order to preprocess the data and set up the correct parameters for the model in the config.json folder.
The set, var, and labeltype arguments must match between the commands as well.

To analyze and print the results (Caution: hard coded path in calculate_mean_cv):
```
python calculate_mean_cv.py
```
