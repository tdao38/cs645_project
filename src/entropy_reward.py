import pandas as pd
import numpy as np
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty

def calculate_D(segment_entropy, class_entropy):
# calculate the entropy based distance for each features
    distance = np.empty(segment_entropy.shape)
    for i in range(len(class_entropy)):
        distance[i, :]= (class_entropy[i]/ segment_entropy.loc[i])

    return distance


def aggreate_reward(distance):
# aggregate the entropy for each features
# currently we just simply add them
    aggregated_distance = distance.sum(axis=0)

    return aggregated_distance

def combine_data(filtered_data):
# convert list of dataframes to one single data frame
    return pd.concat(filtered_data)


def drop_features(aggregated_distance, aggregated_data, features_list):
    temp = np.stack((aggregated_distance, features_list))
    sorted = temp[:, temp[0, :].argsort()[::-1]]
    index = np.argmin(np.diff(sorted[0, :]))
    feature_name = sorted[1, 0:index]
    clean_data = aggregated_data[np.append(feature_name, 'label')]

    return feature_name, clean_data

if __name__ == '__main__':
    ## read cleaned data
    data = pd.read_csv('data/clean/batch146_17_clean.csv')
    # read index data
    index_data = pd.read_csv('data/truth/batch146_17_truth.csv')

    ## map index data and calculate class entropy
    index_data_mapped = mapping(index_data)
    index_data_class_entropy = calculate_class_entropy(index_data_mapped)

    ## calculate segment entropy
    filtered_data = select_segment(data, index_data_class_entropy)
    data_segment_entropy = calculate_segment_entropy(filtered_data)

    ## 6x1 class entropy:
    h_class = index_data_class_entropy['h_class']
    ## 6x19 class entropy:
    h_segment = data_segment_entropy
    #list of all the features
    features_list = data_segment_entropy.columns

    # numpy array len(anomalies) x len(features)
    distance = calculate_D(h_segment, h_class)
    # adding reward for different features
    # numpy array len(feature) x 1
    aggregated_distance = aggreate_reward(distance)
    # convert the list of data frames to one data
    aggregated_data = combine_data(filtered_data)
    feature_name, clean_data = drop_features(aggregated_distance, aggregated_data, features_list)


    # feature_name numpy array
    # clean_data - pandas data frame (features + label)
    Exstream_feature = feature_name
    Exstream_data = clean_data

    # after removing correlated features we will have Exstream_cluster










