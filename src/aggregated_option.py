import pandas as pd
import os
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_distance, combine_data, drop_features, remove_monotonic_feature
from clustering import remove_correlated_features

if __name__ == '__main__':
    data = pd.read_csv('./data/clean/batch146_17_clean.csv')
    # read index data
    index_data = pd.read_csv('./data/truth/batch146_17_truth.csv')
    index_data_mapped = mapping(index_data)
    index_data_class_entropy = calculate_class_entropy(index_data_mapped)
    filtered_data = select_segment(data, index_data_class_entropy)
    aggregated_data = combine_data(filtered_data)
    index_data = calculate_class_entropy(aggregated_data, "aggregate")
    data_segment_entropy = pd.read_csv('./data/aggregated/batch146_17_aggregated.csv')
    #data_segment_entropy = calculate_segment_entropy(aggregated_data, "aggregate")
    distance = calculate_D(data_segment_entropy, index_data['h_class'])
    features_list = data_segment_entropy.columns
    correlated_feature_index = remove_monotonic_feature(aggregated_data, features_list)
    Exstream_feature, Exstream_data = drop_features(distance[0], aggregated_data, features_list,
                                                    correlated_feature_index)
    Exstream_cluster = remove_correlated_features(Exstream_data, Exstream_feature, features_list, distance[0])
    print(Exstream_cluster.columns)