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
├── extensions_plot - contains plots generated from our extension: feature importance and ROC curve
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
4. To save training time, we calculated distance and saved it to a csv file for each batch 
```
python3 src/aggregated_data_creation.py
```

## Main implementation
**1. Training the logical model**

Since each training batch has a separate prediction batch and using different threshold of features, we use 3 different scripts to train and predict each batch:

- Train the model on batch 17, and use it to predict batch 13
```
python3 src/prediction_batch_1713.py
```
- Train the model on batch 19, and use it to predict batch 18
```
python3 src/prediction_batch_1918.py
```
- Train the model on batch 20, and use it to predict batch 15
```
python3 src/prediction_batch_2015.py
```

**2. Stability table**

The results can be found in `data/stability` folder 

```
python3 src/stability.py
```

**3. Extensions**

All three extension models can be found in the `src/extensions.py` script.

```
python3 src/extensions.py
```

To generate predictions on the given test set using XGBoost (batch 13, 18, and 15), use
```
python3 src/generate_ML_preds.py
```


## Visualization
1. Reproduction of Figure 15 (Conciseness) from the paper

```
python3 src/plot_fig15.py
```
2. Features against time plots

```
python3 src/plot_time.py
```
3. Feature Importance figure or the ROC curve in the Extensions

```
python3 src/feature_importance_XGBoost.py
```
