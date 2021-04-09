import pandas as pd
import os

def load_data(path):
    df = pd.read_csv(path)

    return df

def add_timestamp(df_raw):
    # Convert column t to datetime format
    timestamp = pd.to_datetime(df_raw['t'])

    # Convert to timestamp format
    timestamp = timestamp.astype('int64') // 10**9

    # Append timestamp to existing df
    df = df_raw
    df['timestamp'] = timestamp

    return df

def add_label(df_with_timestamp, df_truth):
    # Initiate column label
    df_with_timestamp['label'] = 0

    for i in range(len(df_truth)):
        start = df_truth.loc[i, 'start']
        end = df_truth.loc[i, 'end']
        df_with_timestamp.loc[(df_with_timestamp['timestamp'] >= start) & (df_with_timestamp['timestamp'] <= end), 'label'] = 1

    return df_with_timestamp

if __name__ == '__main__':
    # Set up path
    path_raw = 'data/raw'
    path_clean = 'data/clean'
    path_truth = 'data/truth'

    # Raw data
    file_raw_list = ['batch146_13_raw.csv',
                     'batch146_15_raw.csv',
                     'batch146_17_raw.csv',
                     'batch146_18_raw.csv',
                     'batch146_19_raw.csv',
                     'batch146_20_raw.csv']

    files_with_label = ['batch146_17_raw.csv',
                        'batch146_19_raw.csv',
                        'batch146_20_raw.csv']

    for file_raw in file_raw_list:
        # Log
        print('Cleaning file: ', file_raw)

        # Set up file paths
        file_clean = file_raw.replace('raw', 'clean')
        file_truth = file_raw.replace('raw', 'truth')

        # Load data
        df_raw = load_data(os.path.join(path_raw, file_raw))

        # Add timestamp
        df_with_timestamp = add_timestamp(df_raw)

        # Add label to files with ground truth
        if file_raw in files_with_label:
            df_truth = load_data(os.path.join(path_truth, file_truth))
            df_clean = add_label(df_with_timestamp, df_truth)
        else:
            df_clean = df_with_timestamp

        # Save file
        df_clean.to_csv(os.path.join(path_clean, file_clean), index=False)

        print('Saved file: ', file_clean)