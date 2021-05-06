# EXstream Reimplementation

## GET SETUP LOCALLY
1. Clone this repo locally
2. Create a virtual environment called `venv` using python 3.7
```
# FIRST, make sure python3 is python 3.7
✿ 12:09♡ python3
Python 3.7.4 (default, Sep  7 2019, 18:27:02)
[Clang 10.0.1 (clang-1001.0.46.4)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> quit()

# IF, virtualenv module is not already install, use pip3 to install
✿ 12:09♡ pip3 install virtualenv

# THEN, create virtualenv
✿ 12:09♡ virtualenv venv --python $(which python3)

# NEXT, activate virtualenv - notice that (venv) appears before your prompt now
✿ 12:10♡ source venv/bin/activate
(venv) ✿ 12:10♡ 

# FINALLY, install all requirements from requirements.txt
(venv) ✿ 12:11♡ pip install -r requirements.txt
```
## Directory description
```
├── data
│   ├── aggregated - contains the aggregated segmentation entropy result
│   ├── clean - contains data after cleaning
│   ├── MLpreds - contains prediction results from machine learning model on original data
│   ├── raw - contains raw data
│   ├── segment - contains the segmented segmentation entropy result
│   ├── stability - contains stability calculation
│   ├── test - contains test interval data
│   └── truth - contains ground truth label intervals for training data
├── prediction_result
│   ├── exstream - contains prediction results for test intervals using exstream model
│   └── extension - contains prediction results for test intervals using machine learning extension model
├── src - contains all python scripts we wrote
├── timeplot - contains plots of all features against time
├── .gitignore
├── README.md
├── feature_result.xlsx
├── fig15.png - Figure 15
└── requirements.txt 
```
## Data Preparation
1. The `data` directory is already created and is already storing all the data we use in this project. It is ready to be used by just cloning this repo. If user does not wish to clean data from scratch, this section can be skipped. 
2. However, if users want to run from scratch, create the data directory with all the sub-folders as listed from the tree above and follow the steps below.
3. To clean the raw data, run
```
python3 src/clean.py
```
3. To save training time, we calculated distance and saved it to a csv file for each batch 
```
python3 src/aggregated_data_creation.py
```

## RUNNING STUFF
1. Since each training batch has a separate prediction batch and using different threshold of features, we use 3 different scripts to train and predict each batch:
   1.1. Train batch 17, predict batch 13
```
python3 src/prediction_batch_1713.py
```
   1.2. Train batch 19, predict batch 18
```
python3 src/prediction_batch_1918.py
```
   1.3. Train batch 20, predict batch 15
```
python3 src/prediction_batch_2015.py
```

2. Stability - the results can be found in data/stability folder 
```
python3 src/stability.py
```
3. Extension (Trang)


