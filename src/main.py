import pandas as pd
import numpy as np
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_reward, combine_data, drop_features
from clustering import remove_correlated_features
from prediction import get_prediction_range

pd.options.mode.chained_assignment = None

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
    Exstream_feature, Exstream_data = drop_features(aggregated_distance, aggregated_data, features_list)

    # after removing correlated features (via clustering) we will have Exstream_cluster
    Exstream_cluster = remove_correlated_features(Exstream_data, Exstream_feature, features_list, aggregated_distance)

    prediction_range_dict = get_prediction_range(Exstream_cluster)

    test_data = aggregated_data

    test_data = pd.read_csv('./data/clean/batch146_13_clean.csv')

    # test_data = test_data.reset_index()

    ### Predict:
    k = 0
    label_cols = []
    for feature in prediction_range_dict.keys():
        print('Predicting using feature: ', feature)
        label_col = 'label' + str(k)
        label_cols.append(label_col)
        test_data[label_col] = 0
        prediction_range = prediction_range_dict[feature]
        for i in range(len(test_data)):
            feature_val = test_data[feature][i]
            if any((prediction_range.start < feature_val) & (feature_val < prediction_range.end)):
                test_data[label_col][i] = 1
        k += 1

    test_data['label_count'] = test_data[label_cols].sum(axis=1)
    test_data['label_predict'] = np.where(test_data['label_count'] >= 4, 1, 0)
    test_data.label_predict.sum()

    test_df = test_data[['timestamp', 'label_predict']]
    test_interval = test_df[(test_df.timestamp >= 1528985132) & (test_df.timestamp <= 1528985732)]
    test_interval = test_df[(test_df.timestamp >= 1528985732) & (test_df.timestamp <= 1528986332)]
    test_interval = test_df[(test_df.timestamp >= 1528983932) & (test_df.timestamp <= 1528984532)]
    test_interval = test_df[(test_df.timestamp >= 1528984532) & (test_df.timestamp <= 1528985132)]
    test_interval = test_df[(test_df.timestamp >= 1528983332) & (test_df.timestamp <= 1528983932)]
    test_interval = test_interval
    len(test_interval)
    sum(test_interval.label_predict)

    test_data.label.sum()

    from sklearn.metrics import precision_score, recall_score, confusion_matrix, \
        classification_report, accuracy_score, f1_score

    print('Accuracy:', accuracy_score(test_data.label, test_data.label_predict))
    print('F1 score:', f1_score(test_data.label, test_data.label_predict))
    print('Recall:', recall_score(test_data.label, test_data.label_predict))
    print('Precision:', precision_score(test_data.label, test_data.label_predict))
    print('\n clasification report:\n', classification_report(test_data.label, test_data.label_predict))
    print('\n confussion matrix:\n', confusion_matrix(test_data.label, test_data.label_predict))


    #### Prediction: Thu
    # Utility function: get_prediction_range
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