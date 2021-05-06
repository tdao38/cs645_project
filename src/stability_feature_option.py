import pandas as pd
import numpy as np
import os
from segmentation import mapping, calculate_class_entropy, select_segment, calculate_segment_entropy, \
    calculate_segment_penalty
from entropy_reward import calculate_D, aggreate_distance, combine_data, drop_features, remove_monotonic_feature
from prediction import get_prediction_range, predict, predict_interval

# shiqiGao shiqigao@umass.edu
# Thu Dao tdao@umass.edu
pd.options.mode.chained_assignment = None

# batch 17, "driver_StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value", "1_diff_node7_CPU_ALL_Idle%"
# batch 19, "driver_StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value" ,  "driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value"
# batch 20, "driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value",  "1_diff_node7_CPU_ALL_Idle%"

if __name__ == '__main__':
    data = pd.read_csv('./data/clean/batch146_17_clean.csv')
    # read index data
    index_data = pd.read_csv('./data/truth/batch146_17_truth.csv')

    ## map index data and calculate class entropy
    index_data_mapped = mapping(index_data)
    index_data_class_entropy = calculate_class_entropy(index_data_mapped)

    ## calculate segment entropy
    filtered_data = select_segment(data, index_data_class_entropy)

    aggregated_data = combine_data(filtered_data)

    Exstream_cluster = aggregated_data[
        ["driver_StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value", "1_diff_node7_CPU_ALL_Idle%",
         "label"]]
    ### Prediction
    ### Get a dictionary of prediction ranges for each feature
    prediction_range_dict = get_prediction_range(Exstream_cluster)

    ### Repeat for testing:
    ### For testing
    test_data = pd.read_csv('./data/clean/batch146_13_clean.csv')
    test_interval = pd.read_csv('./data/test/batch146_13_test.csv')
    predicted_data_ml = pd.read_csv('./data/MLpreds/batch146_13.csv')
    predicted_data_ml = predicted_data_ml.rename(columns={"label": "label_predict"})

    ### Get predicted data with "label_predict" column
    predicted_data = predict(test_data, prediction_range_dict, 5)

    # predicted_data.label_predict.sum()

    ### Predict only the test interavals, compare with result from ML model
    predicted_interval = predict_interval(predicted_data, test_interval)
    print(predicted_interval)
    predicted_interval_ml = predict_interval(predicted_data_ml, test_interval)
    print(predicted_interval_ml)
