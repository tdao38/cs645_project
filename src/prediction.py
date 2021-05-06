import pandas as pd
import numpy as np

# Thu Dao tdao@umass.edu
def get_prediction_range(Exstream_cluster):

    """
    Pass in Exstream_cluster data to get a dictionary of prediction range values with start, end interval of a
    abnormal period
    :param Exstream_cluster: Exstream_cluster data
    :return: a dictionary of prediction range, key of dictionary = features chosen by Exstream
    """
    features = Exstream_cluster.columns.tolist()
    features.remove('label')
    prediction_range_dict = {}
    CPU_feature_list = ['1_diff_node5_CPU_ALL_Idle%','1_diff_node6_CPU_ALL_Idle%', '1_diff_node7_CPU_ALL_Idle%',
                       '1_diff_node8_CPU_ALL_Idle%']

    for j in range(len(features)):
        feature = features[j]
        predict_data = Exstream_cluster[[feature, 'label']].iloc[np.lexsort((Exstream_cluster.index, Exstream_cluster[feature]))]
        predict_data['count_0'] = np.where(predict_data['label'] == 0, 1, 0)
        predict_data['count_1'] = np.where(predict_data['label'] == 1, 1, 0)
        count_data = predict_data[[feature, 'count_0', 'count_1']].groupby(feature, as_index=False).sum()
        count_data['final_label'] = np.where(count_data['count_1'] > count_data['count_0'] , 1, 0)

        current_state = 'normal'
        start = []
        end = []
        for i in range(len(count_data)):
            if count_data['final_label'][i] == 0 and current_state == 'normal':
                continue
            elif count_data['final_label'][i] == 1 and current_state == 'normal':
                if i == 0:
                    start.append(count_data[feature][i])
                elif i == len(count_data) - 1:
                    start.append(count_data[feature][i - 1])
                    end.append(count_data[feature][i])
                else:
                    start.append(count_data[feature][i - 1])
                current_state = 'abnormal'
                continue
            elif count_data['final_label'][i] == 1 and current_state == 'abnormal':
                if i == len(count_data) - 1:
                    end.append(count_data[feature][i])
                else:
                    continue
            elif count_data['final_label'][i] == 0 and current_state == 'abnormal':
                end.append(count_data[feature][i])
                current_state = 'normal'
                continue

        prediction_range = pd.DataFrame()
        prediction_range['start'] = start
        prediction_range['end'] = end
        if feature in CPU_feature_list:
            for feature_one in CPU_feature_list:
                prediction_range_dict[feature_one] = prediction_range
        else:
            prediction_range_dict[feature] = prediction_range


    return prediction_range_dict

def predict(test_data, prediction_range_dict, no_of_feature=None):
    """
    Predict the label for the test data, saved under a new label_predict column
    :param test_data:
    :param prediction_range_dict: result from get_prediction_range function
    :param no_of_feature: number of features to use for majority voting. For example, if no_of_feature = 3,
    if at least 3 features agree that a data point is abnormal, then the data point is classified as
    abnormal. If nothing is passed, the default value is the number of Exstream features - 1
    :return:
    """
    if no_of_feature is None:
        no_of_feature = len(prediction_range_dict.keys()) - 1

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
    test_data['label_predict'] = np.where(test_data['label_count'] >=no_of_feature, 1, 0)

    return test_data

def predict_interval(predicted_data, test_interval):
    """
    Return the predicted number of abnormal points and ratio of abnormal period for each test interval. No
    final label for each interval YET.
    :param predicted_data: predicted data from predict function
    :param test_interval: interval data
    :return: the length and predicted number of abnormal points for each test interval
    """
    test_df = predicted_data[['timestamp', 'label_predict']]
    test_interval['length'] = test_interval['end'] - test_interval['start'] + 1
    test_interval['number_of_abnormal'] = 0
    for i in range(len(test_interval)):
        test_interval_i = test_df[(test_df.timestamp >= test_interval.start[i]) & (test_df.timestamp <= test_interval['end'][i])]
        test_interval['number_of_abnormal'][i] = sum(test_interval_i.label_predict)

    test_interval['ratio'] = test_interval['number_of_abnormal']/test_interval['length']

    predicted_interval = test_interval

    return predicted_interval