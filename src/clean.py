import pandas as pd
import os

def load_data(path):
    df = pd.read_csv(path)

    return df

def clean_data(df_raw, df_truth):
    # Convert column t to datetime format
    timestamp = pd.to_datetime(df_raw['t'])

    # Convert to timestamp format
    timestamp = timestamp.astype('int64') // 10**9

    # Append timestamp to existing df
    df_clean = df_raw
    df_clean['timestamp'] = timestamp

    # Initiate column label
    df_clean['label'] = 0

    for i in range(len(df_truth)):
        start = df_truth.loc[i, 'start']
        end = df_truth.loc[i, 'end']
        df_clean.loc[(df_clean['timestamp'] >= start) & (df_clean['timestamp'] <= end), 'label'] = 1

    return df_clean

if __name__ == '__main__':
    # Set up path
    path_raw = 'data/raw'
    path_clean = 'data/clean'
    path_truth = 'data/truth'

    # Raw data
    file_raw_list = ['batch146_17_raw.csv', 'batch146_19_raw.csv', 'batch146_20_raw.csv']

    for file_raw in file_raw_list:
        # Log
        print('Cleaning file: ', file_raw)

        # Set up file paths
        file_clean = file_raw.replace('raw', 'clean')
        file_truth = file_raw.replace('raw', 'truth')

        # Load data
        df_raw = load_data(os.path.join(path_raw, file_raw))
        df_truth = load_data(os.path.join(path_truth, file_truth))

        # Clean data
        df_clean = clean_data(df_raw, df_truth)

        # Save file
        df_clean.to_csv(os.path.join(path_clean, file_clean))

        print('Saved file: ', file_raw)