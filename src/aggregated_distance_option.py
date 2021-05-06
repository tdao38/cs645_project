import pandas as pd
import numpy as np
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_distance, combine_data, drop_features, remove_monotonic_feature
from clustering import remove_correlated_features
from prediction import get_prediction_range
from prediction import get_prediction_range, predict, predict_interval
from sklearn.metrics import precision_score, recall_score, confusion_matrix, \
    classification_report, accuracy_score, f1_score

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    ## read cleaned data
    data = pd.read_csv('./data/clean/batch146_20_clean.csv')
    # read index data
    index_data = pd.read_csv('./data/truth/batch146_20_truth.csv')

    ## map index data and calculate class entropy
    index_data_mapped = mapping(index_data)
    index_data_class_entropy = calculate_class_entropy(index_data_mapped)

    ## calculate segment entropy
    filtered_data = select_segment(data, index_data_class_entropy)
    data_segment_entropy = pd.read_csv('./data/segment/batch146_20_segment.csv')

    ## 6x1 class entropy:
    h_class = index_data_class_entropy['h_class']
    ## 6x19 class entropy:
    h_segment = data_segment_entropy

    # numpy array len(anomalies) x len(features)
    distance = calculate_D(h_segment, h_class)
    # adding reward for different features
    # numpy array len(feature) x 1
    aggregated_distance = aggreate_distance(distance)
    # convert the list of data frames to one data
    aggregated_data = combine_data(filtered_data)
    #list of all the features
    features_list = data_segment_entropy.columns
    correlated_feature_index= remove_monotonic_feature(aggregated_data, features_list)
    Exstream_feature, Exstream_data = drop_features(aggregated_distance, aggregated_data, features_list, correlated_feature_index)

    # after removing correlated features (via clustering) we will have Exstream_cluster
    Exstream_cluster = remove_correlated_features(Exstream_data, Exstream_feature, features_list, aggregated_distance)
    print(Exstream_cluster.columns)

    ### Prediction
    ### Get a dictionary of prediction ranges for each feature
    prediction_range_dict = get_prediction_range(Exstream_cluster)

    ### For training performance:
    test_data = aggregated_data
    test_data = test_data.reset_index()

    ### Get predicted data with "label_predict" column
    predicted_data = predict(test_data, prediction_range_dict, 4)

    ### Only for training data:
    print('Accuracy:', accuracy_score(test_data.label, test_data.label_predict))
    print('F1 score:', f1_score(test_data.label, test_data.label_predict))
    print('Recall:', recall_score(test_data.label, test_data.label_predict))
    print('Precision:', precision_score(test_data.label, test_data.label_predict))
    print('\n clasification report:\n', classification_report(test_data.label, test_data.label_predict))
    print('\n confussion matrix:\n', confusion_matrix(test_data.label, test_data.label_predict))

    ### Repeat for testing:
    ### For testing
    test_data = pd.read_csv('./data/clean/batch146_13_clean.csv')
    test_interval = pd.read_csv('./data/test/batch146_13_test.csv')
    predicted_data_ml = pd.read_csv('./data/MLpreds/batch146_13.csv')
    predicted_data_ml = predicted_data_ml.rename(columns={"label": "label_predict"})

    ### Get predicted data with "label_predict" column
    predicted_data = predict(test_data, prediction_range_dict, 4)

    # predicted_data.label_predict.sum()

    ### Predict only the test interavals, compare with result from ML model
    predicted_interval = predict_interval(predicted_data, test_interval)
    print(predicted_interval)
    predicted_interval_ml = predict_interval(predicted_data_ml, test_interval)
    print(predicted_interval_ml)
