import pandas as pd
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_reward, combine_data, drop_features, remove_monotonic_feature
from clustering import remove_correlated_features

if __name__ == '__main__':
    ## read cleaned data
    data = pd.read_csv('./data/clean/batch146_17_clean.csv')
    # read index data
    index_data = pd.read_csv('./data/truth/batch146_17_truth.csv')

    ## map index data and calculate class entropy
    index_data_mapped = mapping(index_data)
    index_data_class_entropy = calculate_class_entropy(index_data_mapped)

    ## calculate segment entropy
    filtered_data = select_segment(data, index_data_class_entropy)
    data_segment_entropy = pd.read_csv('./data/segment/batch146_17_segment.csv')

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
    correlated_feature_index= remove_monotonic_feature(aggregated_data, features_list)
    Exstream_feature, Exstream_data = drop_features(aggregated_distance, aggregated_data, features_list, correlated_feature_index)

    # after removing correlated features (via clustering) we will have Exstream_cluster
    Exstream_cluster = remove_correlated_features(Exstream_data, Exstream_feature, features_list, aggregated_distance)
    print(Exstream_cluster.columns)

    # data = Exstream_cluster[[Exstream_cluster.columns[0], 'label']]
    # data = data.sort_values(by=Exstream_cluster.columns[0])

    # data = aggregated_data[['1_diff_node5_CPU_ALL_Idle%', 'label']]
    # data = data.sort_values(by='1_diff_node5_CPU_ALL_Idle%').reset_index()


    #### Prediction: Thu
    # Utility function:
    # input a column => output the range of abnormal as a list of tuple (start, end)
    # do the range of > right before &  < after period, not the exact value

    # Prediction function:
    # input the test data, call the utility function => do the filter
    # 1 1 1 0 0 => 1 => pick the majority => do this way
    # 1 1 0 0 => 1 because we rather misclassify as 1 than 0

    # return the test df predicted label of the test data

    # Don't need to plot the figure but export the csv


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


    #### Extension: Trang
    # train test split - just use one batch for now
    # down sample
    # Tree - baseline
    # Kmean - rescale - can plot if we have 2 variables
    # Neural net: