import pandas as pd
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_reward, combine_data, drop_features
from clustering import remove_correlated_features

if __name__ == '__main__':
    ## read cleaned data
    data = pd.read_csv('~/cs645_project/data/clean/batch146_17_clean.csv')
    # read index data
    index_data = pd.read_csv('~/cs645_project/data/truth/batch146_17_truth.csv')

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

    # numpy array len(anomalies) x len(features)
    distance = calculate_D(h_segment, h_class)
    # adding reward for different features
    # numpy array len(feature) x 1
    aggregated_distance = aggreate_reward(distance)
    # convert the list of data frames to one data
    aggregated_data = combine_data(filtered_data)
    #list of all the features
    features_list = data_segment_entropy.columns
    Exstream_feature, Exstream_data = drop_features(aggregated_distance, aggregated_data, features_list)

    # after removing correlated features we will have Exstream_cluster
    clean_exstream = remove_correlated_features(Exstream_data, Exstream_feature, features_list, aggregated_distance)