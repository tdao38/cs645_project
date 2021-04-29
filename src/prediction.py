import pandas as pd
import numpy as np

def get_prediction_range(Exstream_cluster):
    features = Exstream_cluster.columns.tolist()
    features.remove('label')
    prediction_range_dict = {}
    for j in range(len(features)):
        feature = features[j]
        predict_data = Exstream_cluster[[feature, 'label']].sort_values(by=feature)
        predict_data['count_0'] = np.where(predict_data['label'] == 0, 1, 0)
        predict_data['count_1'] = np.where(predict_data['label'] == 1, 1, 0)
        count_data = predict_data[[feature, 'count_0', 'count_1']].groupby(feature, as_index=False).sum()
        count_data['final_label'] = np.where(count_data['count_1'] > 0, 1, 0)

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

        prediction_range_dict[feature] = prediction_range

    return prediction_range_dict