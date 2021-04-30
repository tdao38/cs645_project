import pandas as pd
import numpy as np
import os
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_reward, combine_data, drop_features, remove_monotonic_feature
from clustering import remove_correlated_features
from prediction import get_prediction_range

pd.options.mode.chained_assignment = None

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

    prediction_range_dict = get_prediction_range(Exstream_cluster)

    # test_data = aggregated_data
    # test_data = test_data.reset_index()

    test_data = pd.read_csv('./data/clean/batch146_13_clean.csv')

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
    test_data['label_predict'] = np.where(test_data['label_count'] >=4, 1, 0)
    test_data.label_predict.sum()

    test_df = test_data[['timestamp', 'label_predict']]
    test_interval = test_df[(test_df.timestamp >= 1528985132) & (test_df.timestamp <= 1528985732)]
    test_interval = test_df[(test_df.timestamp >= 1528985732) & (test_df.timestamp <= 1528986332)]
    test_interval = test_df[(test_df.timestamp >= 1528983932) & (test_df.timestamp <= 1528984532)]
    test_interval = test_df[(test_df.timestamp >= 1528984532) & (test_df.timestamp <= 1528985132)]
    test_interval = test_df[(test_df.timestamp >= 1528983332) & (test_df.timestamp <= 1528983932)]
    test_interval = test_df[(test_df.timestamp >= 1528982732) & (test_df.timestamp <= 1528983332)]
    len(test_interval)
    sum(test_interval.label_predict)

    from sklearn.metrics import precision_score, recall_score, confusion_matrix, \
        classification_report, accuracy_score, f1_score

    print('Accuracy:', accuracy_score(test_data.label, test_data.label_predict))
    print('F1 score:', f1_score(test_data.label, test_data.label_predict))
    print('Recall:', recall_score(test_data.label, test_data.label_predict))
    print('Precision:', precision_score(test_data.label, test_data.label_predict))
    print('\n clasification report:\n', classification_report(test_data.label, test_data.label_predict))
    print('\n confussion matrix:\n', confusion_matrix(test_data.label, test_data.label_predict))