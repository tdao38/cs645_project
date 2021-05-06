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
## Data Preparation
1. The `data` directory is already created and is already storing all the data we use in this project. It is ready to be used by just cloning this repo. However, if users want to run from scratch, then create following directory. 
    1. data/aggregated
    2. data/clean
    3. data/MLpreds
    4. data/raw
    5. data/segment
    6. data/stability
    7. data/test 
    8. data/truth

1. clean up (thu)
2. To save training time, we calculated distance and saved it to a csv file for each batch 
```
python3 src/aggregated_data_creation.py
```

## RUNNING STUFF
1. prediction (Thu)
2. Stability - the results can be found in data/stability folder 
```
python3 src/stability.py
```
3. Extension (Trang)


