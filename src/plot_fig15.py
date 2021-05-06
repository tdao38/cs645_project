import pandas as pd
import os
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_reward, combine_data, drop_features, remove_monotonic_feature
from clustering import remove_correlated_features
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

# Thu Dao tdao@umass.edu
if __name__ == '__main__':
    Exstream_list = []
    Exstream_cluster_list = []

    path_clean = 'data/clean'
    path_truth = 'data/truth'
    path_segment = 'data/aggregated'
    file_clean_list = ['batch146_17_clean.csv',
                       'batch146_19_clean.csv',
                       'batch146_20_clean.csv']

    for file_clean in file_clean_list:
        file_truth = file_clean.replace('clean', 'truth')
        file_segment = file_clean.replace('clean', 'aggregated')
        data = pd.read_csv(os.path.join(path_clean, file_clean))

        # read index data
        index_data = pd.read_csv(os.path.join(path_truth, file_truth))
        index_data_mapped = mapping(index_data)
        index_data_class_entropy = calculate_class_entropy(index_data_mapped)
        filtered_data = select_segment(data, index_data_class_entropy)
        aggregated_data = combine_data(filtered_data)
        index_data = calculate_class_entropy(aggregated_data, "aggregate")
        data_segment_entropy = pd.read_csv(os.path.join(path_segment, file_segment))
        #data_segment_entropy = calculate_segment_entropy(aggregated_data, "aggregate")
        distance = calculate_D(data_segment_entropy, index_data['h_class'])
        features_list = data_segment_entropy.columns
        correlated_feature_index = remove_monotonic_feature(aggregated_data, features_list)
        Exstream_feature, Exstream_data = drop_features(distance[0], aggregated_data, features_list,
                                                        correlated_feature_index)
        Exstream_cluster = remove_correlated_features(Exstream_data, Exstream_feature, features_list, distance[0])
        Exstream_list.append(len(Exstream_feature))
        Exstream_cluster_list.append(len(Exstream_cluster.columns) - 1)
        # print(Exstream_cluster.columns)

    # Set up plot data
    plotdata = pd.DataFrame({'Exstream': Exstream_list,
                             'Exstream_cluster': Exstream_cluster_list},
                            index=['batch146_17', 'batch146_19', 'batch146_20'])
    # %%
    # Plot figure 15
    plotdata.reset_index().plot(x='index', y=['Exstream', 'Exstream_cluster'], kind='bar', rot=0)
    plt.title('Figure 15. Conciseness comparison')
    plt.xlabel('Workloads')
    plt.ylabel('Number of features')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('fig15.png', bbox_inches='tight')
    plt.show()
