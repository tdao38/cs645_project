# Author: Trang Tran
# Email: ttrang@umass.edu

## Extensions - Anomaly Detection Algorithm
## Train/Test split and Model training will be done on batch 17 only
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, \
    classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
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
    y_pred = xgb.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('\n clasification report:\n', classification_report(y_test, y_pred))
    print('\n confussion matrix:\n', confusion_matrix(y_test, y_pred))

    print('==============================================================================')
    print('Building Random Forest')
    rf = RandomForestClassifier(random_state=9821)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('\n clasification report:\n', classification_report(y_test, y_pred))
    print('\n confussion matrix:\n', confusion_matrix(y_test, y_pred))

    print('==============================================================================')
    print('Building Neural Networks')
    tf.random.set_seed(9821)
    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[-1],)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(9e-3),
        loss="binary_crossentropy",
        metrics=[f1]
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=512,
        epochs=100,
        verbose=2,
        validation_split=0.2)
    y_pred = (model.predict(X_test) > 0.4).astype('uint8')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('\n clasification report:\n', classification_report(y_test, y_pred))
    print('\n confussion matrix:\n', confusion_matrix(y_test, y_pred))

