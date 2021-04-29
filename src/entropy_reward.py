import pandas as pd
import numpy as np
# from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty

def calculate_D(segment_entropy, class_entropy):
# calculate the entropy based distance for each features
    distance = np.empty(segment_entropy.shape)
    for i in range(len(class_entropy)):
        distance[i, :]= (class_entropy[i]/ segment_entropy.loc[i])

    return distance

def calculate_class_entropy_new(new_data):
    """
    STEP 2:
    Calculate the class entropy based on the sampled reference class
    :param index_data: index_data after STEP 1
    :return: index_data: with added h_class column that represents the class entropy
    """
    # calculate the class entropy & add a new column called class_entropy
    results_matrix = np.zeros((len(new_data), 3))
    index_data = pd.DataFrame(data=results_matrix, columns=("normal_start", "end", "h_class"))
    for i in range(len(new_data)):
        pr=len(new_data[i][new_data[i].label==0])/len(new_data[i])
        pa = len(new_data[i][new_data[i].label==1])/len(new_data[i])
        index_data.loc[i] = [new_data[i].timestamp.min(),new_data[i].timestamp.max(),pa * np.log(1 / pa) + pr * np.log(1 / pr)]

    return index_data

def aggreate_reward(distance):
# aggregate the entropy for each features
# currently we just simply add them
    aggregated_distance = distance.sum(axis=0)

    return aggregated_distance

def remove_monotonic_feature(aggregated_data, features_list):
    montonic_index = []
    for i in range(len(features_list)):
        if aggregated_data[features_list[i]].is_monotonic:
            montonic_index.append(i)
        elif aggregated_data[features_list[i]].is_monotonic_decreasing:
            montonic_index.append(i)

    return montonic_index
# TO DO
# test the code -
# perfect scenario when the distance is 1 - Thu
# less perfect scenario when the distance is close to 1

# before drop - Amy  change the drop function
# feature with strongly correlated with time - need to remove them- montonically increasing or decreasing then remove
# after false positive filtering , before removing correlation
# graph

# predictive model issue
## option 1
#
### concate the refernce data for six instances and treat them as 1 pass in to the exstream
## NANNNAAA
# CLASS ENTROPT
# SEMETATION

# 1x 10 vector feature size is 10

# option 2
# reward for each anaomlues
# six models for each anomlies, then we look at the features list for all six and then union them
## remove inconsistency
# less weight on the bad features
# stability score or correlated with time


def combine_data(filtered_data):
# convert list of dataframes to one single data frame
    return pd.concat(filtered_data)


def drop_features(aggregated_distance, aggregated_data, features_list, correlated_feature_index):
    #1, 0.7 , 0.6, 0.5, 0.4, 0.3
    #
    # drop  0.7 , 0.6, 0.5, 0.4, 0.3
    # keep 1

    temp = np.stack((aggregated_distance, features_list))
    sorted = temp[:, temp[0, :].argsort()[::-1]]
    index = np.argmin(np.diff(sorted[0, :]))
    index_keep = list(set(range(0, index+1)) - set(correlated_feature_index))
    feature_name = sorted[1, index_keep]
    clean_data = aggregated_data[np.append(feature_name, 'label')]

    return feature_name, clean_data

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
    #list of all the features
    features_list = data_segment_entropy.columns

    # numpy array len(anomalies) x len(features)
    distance = calculate_D(h_segment, h_class)
    # adding reward for different features
    # numpy array len(feature) x 1
    aggregated_distance = aggreate_reward(distance)
    # convert the list of data frames to one data
    aggregated_data = combine_data(filtered_data)
    feature_name, clean_data = drop_features(aggregated_distance, aggregated_data, features_list)


    # feature_name numpy array
    # clean_data - pandas data frame (features + label)
    Exstream_feature = feature_name
    Exstream_data = clean_data

    # after removing correlated features we will have Exstream_cluster










