import pandas as pd
import numpy as np

def mapping(index_data):
    # find the refernced index begining
    index_data['tsa'] = index_data['end'] - index_data['start'] + 1
    index_data['tsr'] = index_data['tsa']*2
    index_data['normal_start'] = index_data['start'] - index_data['tsr']
    index_data['normal_end'] = index_data['start'] - 1

    return index_data

def calculate_class_entropy(index_data):
    # calculate the class entropy & add a new column called class_entropy
    pa = index_data['tsa']/(index_data['tsa']+index_data['tsr'])
    pr = index_data['tsr']/(index_data['tsa']+index_data['tsr'])

    index_data['h_class'] = pa * np.log(1/pa) + pr * np.log(1/pr)

    return index_data

def select_segment(data, index_data):
    # create a list of numpy array that contain filtered segments
    filtered_data = []
    for i in range(len(index_data)):
        start = index_data['normal_start'][i]
        end = index_data['end'][i]
        df = data[(data['timestamp'] >= start) & (data['timestamp'] <= end)]
        filtered_data.append(df)

    return filtered_data

def calculate_segment_entropy(filtered_data):

    results_matrix = np.zeros((len(filtered_data), len(filtered_data[0].columns)-3))
    results_df = pd.DataFrame(data=results_matrix, columns=filtered_data[0].columns[1:-2])

    for i in range(len(filtered_data)):
        print("Calculating segment entropy ", i + 1, " of ", len(filtered_data))
        df = filtered_data[i]
        features = df.columns

        # Initiate dictionary to save segment entropy. Key: feature.
        for j in range(1, len(features) - 2):
            # No mix:
            df_feature = df[[features[j], 'label']]
            df_feature = df_feature.sort_values(by=features[j])
            changes = (df_feature.label != df_feature.label.shift()).cumsum()
            df_feature['segment'] = changes
            pi = df_feature['segment'].value_counts(normalize=True)
            h_segment_no_penalty = np.sum(pi * np.log(1/pi))
            h_segment_penalty = calculate_segment_penalty(df_feature)
            h_segment = h_segment_no_penalty + h_segment_penalty

            results_df[features[j]][i] = h_segment
            # h_segment_dict['h_segment_dict_' + str(i)][features[j]]['h_segment_no_penalty'] = h_segment_no_penalty
            # h_segment_dict['h_segment_dict_' + str(i)][features[j]]['h_segment_penalty'] = h_segment_penalty


    return results_df

def calculate_segment_penalty(df_feature):

    unique_vals = pd.unique(df_feature[df_feature.columns[0]])
    h_segment_penalty_all = 0

    for i in range(len(unique_vals)):
        df_filter = df_feature[df_feature[df_feature.columns[0]] == unique_vals[i]]
        if len(df_filter.label.unique()) > 1:
            a = sum(df_filter['label'] == 1)
            n = len(df_filter) - a
            if a == n:
                number_of_segments = 2*a
                h_segment_penalty = np.log(number_of_segments)
                h_segment_penalty_all += h_segment_penalty
            else:
                number_of_segments = min(a, n)*2 + 1
                len_mixed_segment = number_of_segments - 1
                len_pure_segment = len(df_filter) - len_mixed_segment
                h_segment_penalty = len_pure_segment * 1/len(df_filter) * np.log(len(df_filter)) + \
                                    len_mixed_segment * 1/len(df_filter) * np.log(len(df_filter)/len_mixed_segment)
                h_segment_penalty_all += h_segment_penalty

    return h_segment_penalty_all

if __name__ == '__main__':
    ## read cleaned data
    data = pd.read_csv('data/clean/batch146_17_clean.csv')
    # read index data
    index_data = pd.read_csv('data/truth/batch146_17_truth.csv')

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