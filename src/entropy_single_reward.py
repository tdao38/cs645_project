# pass in the cleaned data (pandas)

# input the starting index data


# THU
def mapping(index_data):

# find the refernced index begining
# calculate the class entropu

return index_data


def select_segment(data, index_data):
# create a list of numpy array that contain features
# add a new column called class_entropy


return filtered_data

def calculate_segmentation_entropy(filtered_data):
# loop over each matrix om the list
# len(interval- refernced + abnormal) x (num of features + label + class entropy)

    # no mix


    # if mix
    ## add penalized


# create numpy array that contain features
# 10 features ( with penalty)
# 6 intervals
# 6 x 10 entropy_list

# extract class entropy from filtered_data ( last column)
# 6x1 class_entropy



return entropy_list, class_entropy


# amy

def calculate_D(entropy_list, class_entropy):

# return a numpy array 6 x 10

return distance


def aggreate_reward(distance):

# sum distance for each features acordd the rows
# 10x1
# [1,1,1,1,1,1,1,1]

return aggregated_distance


def drop_features(aggregated_distance, filtered_data):


# warining filtered_data might have different feature index


# convert filtered_data to one single data frame

# sort from largest to smallest

# find the biggest drop

# find the remaining features index


#[1,5,2]
#[5,2,1]
#[5]
#return 2

# clean_data ( selected features + label)
# pandas type

return features_indx, name, clean_data
# save this featrues for




# trang - assume direty data is used for the tree
# use feature_name, clean_data
def remove_high_correlationn(features_indx, filtered_data):
# warining filtered_data might have different feature index


# convert filtered_data to one single data frame

# for every two features, calculate the correlation

# if correlation is >  |0.5\ , cluster

# select one from each cluster
## keep the feature that has the (highest reward for each cluster
# keep note of the index

# clean_data ( selected features + label)
# pandas type

return uncorrelayed_features_indx, names, clean_data


def logical_model(clean_data):

# some decision tree

return model


