# MICCAI2022_model

## directories
```
model
 ┣ JIGSAWS
 ┃ ┣ tcn  (This folder save the cross validation result test_1 means trainig on the rest while test on trial 1)
 ┃ ┃ ┣ test_1
 ┃ ┃ ┃ ┣ log
 ┃ ┃ ┃ ┃ ┗ train_test_result.csv
 ┃ ┃ ┃ ┣ checkpoint_0.pth
 ┃ ┃ ┣ test_2...
 ┣ Knot_Tying
 ┃ ┣ kin_ges
 ┃ ┃ ┣ Knot_Tying_B001.txt ... [contains the kinematics columns +'Y' (categorical values of the gestures)]
 ┣ Needle_Passing
 ┃ ┣ kin_ges
 ┃ ┃ ┣ Needle_Passing_B001.txt...
 ┣ Suturing
 ┃ ┣ kin_ges
 ┃ ┃ ┣ Suturing_B001.txt...
 ┣ JIGSAWS-TRANSFORM.pkl
 ┣ calculate_mean_cv.ipynb
 ┣ config.json
 ┣ config.py
 ┣ data_loading.py
 ┣ dataprep_code.py
 ┣ get_festure.py
 ┣ logger.py
 ┣ parameter_tuning.py (use this code to perform parameter tuning)
 ┣ requirements.txt
 ┣ tcn_model.py
 ┣ train_test_cross.py (use this code to perform cross validation)
 ┗ utils.py
 ```
## setting up the environment 
* install anaconda 
* conda create --name myenv
* conda activate myenv
* run  conda install --file requirements.txt
## data preparation
* use/ modifiy the code in the dataprep_code to create the datasets under kin_ges. This code assigns each time step a corresponding gesture from the transcription dataset.
* use / modifiy the code in the dataprep_code to create the y transformation model 'G1'-> 1 categorical to ordinal
* change the column index in data_loading.py LOCS_JIGSAWS=
* change the y columns transformation model directory (with open('.....JIGSAWS-TRANSFORM.pkl','rb') as f:)  in data_loading.py
* change directories in the json file
```
        "input_size": 14,---feature size
        "gesture_class_num":14,---unique classes
        "validation_trial":1, -- for parameter tuning train on [2,3,4,5], validation on trial 1
        "test_trial":[1,2,3,4,5],---for cross validation test set
        "train_trial":[[2,3,4,5],[1,3,4,5],[1,2,4,5],[1,2,3,5],[1,2,3,4]], ---for cross validation train set
        "raw_feature_dir":["/home/aurora/Documents/MICCAI2022_baseline/LSTM_model/Needle_Passing/kin_ges","/home/aurora/Documents/MICCAI2022_baseline/LSTM_model/Suturing/kin_ges","/home/aurora/Documents/MICCAI2022_baseline/LSTM_model/Knot_Tying/kin_ges"],
 "tcn_params":{
            "model_params":{
                "class_num":14, ---need to be changed to be same as gesture_class_num
        
```
* check the config.py to see everything is correct

## parameter tuning
to find the learning rate & weight decay with the parameter_tuning.py
```
config = {"learning_rate":tune.loguniform(1e-5,1e-3), "batch_size":1, "weight_decay": tune.loguniform(1e-4,1e-2)}

```
 to find the appropriate epochs
```
main_tcn(num_samples=1, max_num_epochs=60) with fixed config eg. config = {'learning_rate': 0.0003042861945575232, 'batch_size': 1, 'weight_decay': 0.00012035748692105724} 
```
can use the tensorboard to visualize (the tuning result is saved by default at the home directory called ray-tune.Then use the tensorboard command to visual the trainig curve) tensorboard --logdir='/your/path/here'

## cross validation
run the train_test_cross.py code and run the calculate_mean_cv to get the mean values
