# Author: Trang Tran
# Email: ttrang@umass.edu

## Extensions - Anomaly Detection Algorithm
## Train/Test split and Model training will be done on batch 17 only
## Generate ROC curve graph, and Feature Importance graph

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
from xgboost import plot_importance
from matplotlib import pyplot as plt
from extensions_utils import *
ROOT_DIR = get_project_root()

if __name__ == '__main__':
    print('==============================================================================')
    print('Reading Batch 17')
    df = pd.read_csv(os.path.join(ROOT_DIR, 'data/clean/batch146_17_clean.csv'))
    print('Train/Test split and standardizing')
    scaler = StandardScaler()
    X = df.iloc[:, 1:-1]
    y = df.label.values.ravel()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=9821)
    print('Shape of X_test:', X_test.shape, ';', 'Abnormal percentage:',
          round(sum(y_test) / len(y_test), 3))
    print('Shape of X_train:', X_train.shape, ';', 'Abnormal percentage:',
          round(sum(y_train) / len(y_train), 3))

    print('==============================================================================')
    print('Building XGBoost')
    xgb = XGBClassifier(use_label_encoder=False, random_state=9821, learning_rate=0.01)
    xgb.fit(X_train, y_train)

    print('==============================================================================')
    print('Generating graphs')
    # plot ROC curve
    metrics.plot_roc_curve(xgb, X_test, y_test)
    plt.savefig(os.path.join(ROOT_DIR, 'extensions_plot/xgb_roc_curve.png'))
    plt.show()

    # plot feature importance
    plot_importance(xgb)
    plt.savefig(os.path.join(ROOT_DIR, 'extensions_plot/xgb_feature_importance.png'))
    plt.show()
