import pandas as pd
import numpy as np
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_reward, combine_data, drop_features, remove_monotonic_feature
from clustering import remove_correlated_features
from prediction import get_prediction_range, predict, predict_interval
from sklearn.metrics import precision_score, recall_score, confusion_matrix, \
    classification_report, accuracy_score, f1_score

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    print('==============================================================================')
    print('Training Batch 19')
    data = pd.read_csv('./data/clean/batch146_19_clean.csv')

    # read index data
    index_data = pd.read_csv('./data/truth/batch146_19_truth.csv')
    index_data_mapped = mapping(index_data)
    index_data_class_entropy = calculate_class_entropy(index_data_mapped)
    filtered_data = select_segment(data, index_data_class_entropy)
    aggregated_data = combine_data(filtered_data)
    index_data = calculate_class_entropy(aggregated_data, "aggregate")
    data_segment_entropy = pd.read_csv('./data/aggregated/batch146_19_aggregated.csv')
    #data_segment_entropy = calculate_segment_entropy(aggregated_data, "aggregate")
    distance = calculate_D(data_segment_entropy, index_data['h_class'])
    features_list = data_segment_entropy.columns
    correlated_feature_index = remove_monotonic_feature(aggregated_data, features_list)
    Exstream_feature, Exstream_data = drop_features(distance[0], aggregated_data, features_list,
                                                    correlated_feature_index)
    Exstream_cluster = remove_correlated_features(Exstream_data, Exstream_feature, features_list, distance[0])
    print(Exstream_cluster.columns)

    ### Prediction
    ### Get a dictionary of prediction ranges for each feature
    prediction_range_dict = get_prediction_range(Exstream_cluster)

    ## For training performance:
    test_data = aggregated_data
    test_data = test_data.reset_index()

    ### Get predicted data with "label_predict" column
    predicted_data = predict(test_data, prediction_range_dict, 4)

    ### Only for training data:
    print('==============================================================================')
    print('Training Result')
    print('Accuracy:', accuracy_score(test_data.label, test_data.label_predict))
    print('F1 score:', f1_score(test_data.label, test_data.label_predict))
    print('Recall:', recall_score(test_data.label, test_data.label_predict))
    print('Precision:', precision_score(test_data.label, test_data.label_predict))
    print('\n clasification report:\n', classification_report(test_data.label, test_data.label_predict))
    print('\n confussion matrix:\n', confusion_matrix(test_data.label, test_data.label_predict))

    ### Repeat for testing:
    ### For testing
    print('==============================================================================')
    print('Testing Batch 18')
    test_data = pd.read_csv('./data/clean/batch146_18_clean.csv')
    test_interval = pd.read_csv('./data/test/batch146_18_test.csv')
    predicted_data_ml = pd.read_csv('./data/MLpreds/batch146_18.csv')
    predicted_data_ml = predicted_data_ml.rename(columns={"label": "label_predict"})

    ### Get predicted data with "label_predict" column
    predicted_data = predict(test_data, prediction_range_dict, 5)

    print('==============================================================================')
    print('Testing and Prediction Result')
    print('Exstream model')
    ### Predict only the test interavals, compare with result from ML model
    predicted_interval = predict_interval(predicted_data, test_interval)
    predicted_interval['label'] = np.where(predicted_interval['ratio'] >= 0.07, 1, 0)
    predicted_interval[['start', 'end', 'label']].to_csv('prediction_result/exstream/batch146_18_predicted_exstream.csv', index=False)
    print(predicted_interval)

    print('Machine learning model')
    predicted_interval_ml = predict_interval(predicted_data_ml, test_interval)
    predicted_interval_ml['label'] = np.where(predicted_interval_ml['ratio'] >= 0.07, 1, 0)
    predicted_interval_ml[['start', 'end', 'label']].to_csv('prediction_result/extension/batch146_18_predicted_ml.csv', index=False)
    print(predicted_interval_ml)
