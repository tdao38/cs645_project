import pandas as pd
import numpy as np
# from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty

def calculate_D(segment_entropy, class_entropy):
    """
    This function calculates distance for every features given class and segmentation entropy
    :param segment_entropy: segmentation entropy for all feature across anomalies
    :param  class_entropy: class_entropy for all anomalies
    :return:
    2D array contain distance for all the features across anomalies
    """
    distance = np.empty(segment_entropy.shape)
    for i in range(len(class_entropy)):
        distance[i, :] = (class_entropy[i] / segment_entropy.loc[i])

    return distance


def aggreate_distance(distance):
    """
    This function aggregate distance across anomalies
    :param distance: 2D array
    :return:
    1D array contain distance for all the features
    """
    aggregated_distance = distance.sum(axis=0)

    return aggregated_distance


def remove_monotonic_feature(aggregated_data, features_list):
    """
    This function return features that are correlated with time either monotonically increasing or decreasing
    The index of that feature will be recorded and return a list of feature index that need to be removed
    :param aggregated_data: aggregated data across anomalies
    :param features_list: list of features
    :return:
    1D list contain feature index that are correlated with time
    """
    montonic_index = []
    for i in range(len(features_list)):
        if aggregated_data[features_list[i]].is_monotonic:
            montonic_index.append(i)
        elif aggregated_data[features_list[i]].is_monotonic_decreasing:
            montonic_index.append(i)

    return montonic_index


def combine_data(filtered_data):
    """
    This function aggregate list of dataframes to a single dataframe
    :param filtered_data: list of panda dataframe
    :return:
    1 dataframe
    """
    return pd.concat(filtered_data)


def drop_features(aggregated_distance, aggregated_data, features_list, correlated_feature_index):
    """
    This function remove features that has low distance by measure the sharpest drop and remove
    feature that has distance ranked after that drop
    also remove features that are correlated with time
    :param aggregated_distance: 1D array contain distance for all the features
    :param aggregated_data: 1 dataframe
    :param features_list: list of features
    :param correlated_feature_index: list of feature index that are correlated with time
    :return:
    feature_name - a list of feature
    clean_data - dataframe with only kept feature and label

    """
    temp = np.stack((aggregated_distance, features_list))
    sorted = temp[:, temp[0, :].argsort()[::-1]]
    index = np.argmin(np.diff(sorted[0, :]))
    index_keep = list(set(range(0, index + 1)) - set(correlated_feature_index))
    feature_name = sorted[1, index_keep]
    clean_data = aggregated_data[np.append(feature_name, 'label')]

    return feature_name, clean_data
