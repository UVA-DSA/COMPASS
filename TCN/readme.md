Useage: \

Go into config.json and change "raw_feature_dir" to the correct paths to each task's 'preprocessed' folder. \
In the terminal run
```
python preprocess.py <set> <var> <labeltype> <valtype>
```
Options for set are: S, NP, KT, SNP, JIGSAWS, PT, PaS, PoaP, ROSMA, PTPaS, All  \
Options for var are: velocity, orientation, all \
Options for labeltype are: MPbaseline, MPleft, MPright, gesture \
Options for valtype are: LOSO, LOUO, LOTO 

Then, to tune or train, run:
```
python tune.py <set> <var> <labeltype> <valtype>
```
or 
```
python train_test_val.py <set> <var> <labeltype> <valtype>
```
Note: You must run preprocess.py before tune.py or train_test_val.py in order to preprocess the data and set up the correct parameters for the model in the config.json folder. 
The set, var, labeltype, and valtype arguments must match between the commands as well.
This process is also automated with run.py in which combinations of options can be selected and trained.

To analyze and print the results (Caution: hard coded path in calculate_mean_cv):
```
python calculate_mean_cv.py
```
