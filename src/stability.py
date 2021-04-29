import pandas as pd
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_reward, combine_data, drop_features, remove_monotonic_feature
from clustering import remove_correlated_features
import os
import numpy as np


#### Stability: Amy
# 6 anomalies: resample 10 times for each anomaly, sample size: 20% from A, 20% from N, without replacement
# Repeat exstream for each sample
# 10 lists of features for each anomaly (60 lists in total)
# list 1: A B C
# list 2: B C D
# => A B B C C D
# => A: 1/6, B: 2/6, C: 2/6, D: 1/6
# H:  - (1/6 * log_2(1/6) + 2/6 * log_2(2/6) + .....)
# 1 stablity for each anomaly => 6 stability for each batch, calculate for all 3 training batches
# Create a table for each batch (3 tables) with 2 columns: average feature size + average stability
# each table has average feature size and stability for each anomaly (7 rows) - average at the end
# average stability: go through each 10 list to count how many times each feature appears
# run this outside of main.py

def random_data(filtered_data):
    new_filtered_data = []
    for i in range(len(filtered_data)):
        normal= filtered_data[i][filtered_data[i].label == 0]
        abnomral = filtered_data[i][filtered_data[i].label == 1]
        remove_normal = int(len(normal) * 0.2)
        remove_abnormal = int(len(abnomral) * 0.2)
        drop_indices_normal = np.random.choice(normal.index, remove_normal, replace=False)
        drop_indices_abnormal = np.random.choice(abnomral.index, remove_abnormal, replace=False)
        normal_random = normal.drop(drop_indices_normal)
        abnormal_random = abnomral.drop(drop_indices_abnormal)
        new_filtered_data.append(pd.concat([normal_random , abnormal_random]))

    return new_filtered_data

def stability(filtered_data, features_list, iteration):
    feature_list_result = []
    for i in range(iteration):
        new_data = random_data(filtered_data)
        index_data = calculate_class_entropy(new_data, "stability")
        new_data = select_segment(data, index_data)
        data_segment_entropy = calculate_segment_entropy(new_data)
        distance = calculate_D(data_segment_entropy, index_data['h_class'])
        for j in range(len(distance)):
            correlated_feature_index = remove_monotonic_feature(filtered_data[j], features_list)
            Exstream_feature, Exstream_data = drop_features(distance[j,:], filtered_data[j], features_list, correlated_feature_index)
            if len(Exstream_feature)==1:
                feature_list_result.append(Exstream_data.columns[:-1].values)
            else:
                Exstream_cluster = remove_correlated_features(Exstream_data, Exstream_feature, features_list, distance[j,:])
                feature_list_result.append(Exstream_cluster.columns[:-1].values)

    stability_matrix = np.zeros((2, len(distance)))
    list = np.array(feature_list_result)
    for i in range(len(distance)):
        index = np.array(range(i, len(list), len(distance)))
        temp = list[index]
        avg_size, stability = stats(temp)
        stability_matrix[:, i] = avg_size, stability

    return stability_matrix, feature_list_result

def stats(temp):
    total=np.zeros(len(temp))
    for i in range(len(temp)):
        total[i]=len(temp[i])
    aggregate = np.concatenate(temp)
    stability = calculate_stability(aggregate)

    return total.mean(), stability

def calculate_stability(aggregate_list):
    unique, counts = np.unique(aggregate_list, return_counts=True)
    counts = counts/len(aggregate_list)
    stability =  -(counts * np.log2(counts)).sum()

    return stability


if __name__ == '__main__':

    path_clean = 'data/clean'
    path_truth = 'data/truth'
    path_segment = 'data/segment'
    path_output = 'data/stability'

    file_clean_list = ['batch146_17_clean.csv',
                       'batch146_19_clean.csv',
                       'batch146_20_clean.csv']

    iteration = 20
    for file_clean in file_clean_list:
        print('=============================================================')
        print('Calculating file ', file_clean)

        # set up export files:
        file_truth = file_clean.replace('clean', 'truth')
        file_segment = file_clean.replace('clean', 'segment')

        ## read cleaned data
        data = pd.read_csv(os.path.join(path_clean, file_clean))
        # read index data
        index_data = pd.read_csv(os.path.join(path_truth, file_truth))

        ## map index data and calculate class entropy
        index_data_mapped = mapping(index_data)
        index_data_class_entropy = calculate_class_entropy(index_data_mapped)

        ## calculate segment entropy
        filtered_data = select_segment(data, index_data_class_entropy)
        features_list = filtered_data[0].columns[1:-2].values
        array, list = stability(filtered_data,features_list, iteration)
        ## convert your array into a dataframe
        df = pd.DataFrame(array.T)
        df.columns = ["avg_size", "stability"]
        df["start"] = index_data["normal_start"]
        df["end"] = index_data["end"]
        print(list)

        ## save to xlsx file

        filepath = os.path.join(path_output, file_clean[:11]+'_stability.csv')

        df.to_csv(filepath, index=False)





