# Author: Trang Tran
# Email: ttrang@umass.edu

## Extensions - Generate ML predictions using XGBoost
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from extensions_utils import *
ROOT_DIR = get_project_root()

if __name__ == '__main__':
    path_MLpreds = 'data/MLpreds'
    print('==============================================================================')
    print('Training batch 17')
    scaler = StandardScaler()
    df = pd.read_csv(os.path.join(ROOT_DIR, 'data/clean/batch146_17_clean.csv'))
    X = df.iloc[:, 1:-1]
    y_train = df.label.values.ravel()
    X_train = scaler.fit_transform(X)
    df_test = pd.read_csv(os.path.join(ROOT_DIR, 'data/clean/batch146_13_clean.csv'))
    X_test_raw = df_test[df_test.columns.drop('t')]
    X_test = scaler.fit_transform(X_test_raw)
    xgb = XGBClassifier(use_label_encoder=False,
                        random_state=9821,
                        learning_rate=0.07,
                        max_depth=6)
    xgb.fit(X_train, y_train)
    print('Predict batch 13 and write result')
    y_pred = xgb.predict(X_train)
    df_test['label'] = xgb.predict(X_test)
    df_test.to_csv(os.path.join(ROOT_DIR, path_MLpreds, 'batch146_13.csv'), index=False)

    print('==============================================================================')
    print('Training batch 19')
    scaler = StandardScaler()
    df = pd.read_csv(os.path.join(ROOT_DIR, 'data/clean/batch146_19_clean.csv'))
    X = df.iloc[:, 1:-1]
    y_train = df.label.values.ravel()
    X_train = scaler.fit_transform(X)
    df_test = pd.read_csv(os.path.join(ROOT_DIR, 'data/clean/batch146_18_clean.csv'))
    X_test_raw = df_test[df_test.columns.drop('t')]
    X_test = scaler.fit_transform(X_test_raw)
    xgb = XGBClassifier(use_label_encoder=False,
                        random_state=9821,
                        learning_rate=0.07,
                        max_depth=6)
    xgb.fit(X_train, y_train)
    print('Predict batch 18 and write result')
    y_pred = xgb.predict(X_train)
    df_test['label'] = xgb.predict(X_test)
    df_test.to_csv(os.path.join(ROOT_DIR, path_MLpreds, 'batch146_18.csv'), index=False)

    print('==============================================================================')
    print('Training batch 20')
    path_MLpreds = 'data/MLpreds'
    scaler = StandardScaler()
    df = pd.read_csv(os.path.join(ROOT_DIR, 'data/clean/batch146_20_clean.csv'))
    X = df.iloc[:, 1:-1]
    y_train = df.label.values.ravel()
    X_train = scaler.fit_transform(X)
    df_test = pd.read_csv(os.path.join(ROOT_DIR, 'data/clean/batch146_15_clean.csv'))
    X_test_raw = df_test[df_test.columns.drop('t')]
    X_test = scaler.fit_transform(X_test_raw)
    xgb = XGBClassifier(use_label_encoder=False,
                        random_state=9821,
                        learning_rate=0.07,
                        max_depth=6)
    xgb.fit(X_train, y_train)
    print('Predict batch 15 and write result')
    y_pred = xgb.predict(X_train)
    df_test['label'] = xgb.predict(X_test)
    df_test.to_csv(os.path.join(ROOT_DIR, path_MLpreds, 'batch146_15.csv'), index=False)