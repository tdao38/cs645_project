import pandas as pd
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, \
    calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_distance, combine_data, drop_features, remove_monotonic_feature
from clustering import remove_correlated_features
import os
import numpy as np

# shiqiGao shiqigao@umass.edu
if __name__ == '__main__':

    path_clean = 'data/clean'
    path_truth = 'data/truth'
    path_output = 'data/aggregated'

    file_clean_list = ['batch146_17_clean.csv',
                       'batch146_19_clean.csv',
                       'batch146_20_clean.csv']

    for file_clean in file_clean_list:
        print('=============================================================')
        print('Calculating file ', file_clean)

        # set up export files:
        file_truth = file_clean.replace('clean', 'truth')

        ## read cleaned data
        data = pd.read_csv(os.path.join(path_clean, file_clean))
        # read index data
        index_data = pd.read_csv(os.path.join(path_truth, file_truth))

        index_data_mapped = mapping(index_data)
        index_data_class_entropy = calculate_class_entropy(index_data_mapped)
        filtered_data = select_segment(data, index_data_class_entropy)
        aggregated_data = combine_data(filtered_data)
        index_data = calculate_class_entropy(aggregated_data, "aggregate")
        data_segment_entropy = calculate_segment_entropy(aggregated_data, "aggregate")

        ## save the stability result to cvs file
        filepath = os.path.join(path_output, file_clean[:11] + '_aggregate.csv')
        data_segment_entropy.to_csv(filepath, index=False)
