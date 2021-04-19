from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import pandas as pd

# Clustering using hierarchy linkage from SciPy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
def remove_correlated_features(clean_data, clean_feature, features_lst, aggregated_dist, threshold=0.95):
    """

    :param clean_data: Data after dropping based on sharp drop
    :param clean_feature: Feature names of clean_data (not including 'label')
    :param features_list: Original feature names (before any droppping)
    :param aggregated_dist: the feature rewards of all features in features_list
    :param threshold: TODO
    :return:
    Final clean data for the logical model
    """
    features = clean_data.iloc[:, :-1]
    # Calculate the correlation matrix (absolute value version), which shows how close each feature is (range 0,1)
    corr_matrix = features.corr().abs()

    # Calculate how far each feature is
    dissimilarity = 1 - abs(corr_matrix)

    # hierarchy linkage
    Z = linkage(squareform(dissimilarity), 'complete')
    labels = fcluster(Z, threshold, criterion='distance')

    # cluster label
    labels = pd.DataFrame(labels)
    labels['name'] = clean_feature.tolist()
    labels.columns = ['cluster', 'name']

    # map label of cluster with actual reward value
    agg_distance = pd.DataFrame(aggregated_dist)
    agg_distance['name'] = features_lst.tolist()
    agg_distance.columns = ['reward', 'name']
    reward_and_cluster = labels.merge(agg_distance, how='left')

    # columns to keep (highest reward in a group)
    to_keep = reward_and_cluster.loc[reward_and_cluster.groupby(['cluster'])["reward"].idxmax()]
    names_to_keep = to_keep['name'].tolist()
    names_to_keep.append('label')

    # only best features in each group is kept
    uncorrelated_data = clean_data[names_to_keep]

    return uncorrelated_data