import numpy as np
import pandas as pd

from segmentation import calculate_segment_entropy, calculate_segment_penalty

# Thu Dao tdao@umass.edu
if __name__ == '__main__':
    # Fake column, won't be used, won't be used
    time = np.repeat(0, 3000)
    timestamp = np.repeat(0, 3000)
    feature = np.arange(3000)
    feature_penalty = np.repeat(np.array([1,2]), [1500,1500])

    # Perfect separation case, balance
    label1 = np.repeat(np.array([0,1]), [1500, 1500])

    # Perfect separation case, imbalance
    label2 = np.repeat(np.array([0,1]), [1000, 2000])

    # 3 segment case, should be low
    label3 = np.repeat(np.array([0,1,0]), [1500, 1499, 1])

    # 3 segment case, but larger segment size
    label4 = np.repeat(np.array([0,1,0]), [1500, 750, 750])

    # Penalty case
    label5 = np.repeat(np.array([0, 1, 0, 1]), [500, 1000, 750, 750])

    # Combine into df
    df1 = pd.DataFrame({'time': time, 'feature': feature, 'timestamp': timestamp, 'label': label1})
    df2 = pd.DataFrame({'time': time, 'feature': feature, 'timestamp': timestamp, 'label': label2})
    df3 = pd.DataFrame({'time': time, 'feature': feature, 'timestamp': timestamp, 'label': label3})
    df4 = pd.DataFrame({'time': time, 'feature': feature, 'timestamp': timestamp, 'label': label4})
    df5 = pd.DataFrame({'time': time, 'feature': feature_penalty, 'timestamp': timestamp, 'label': label5})

    filtered_data = [df1, df2, df3, df4, df5]

    # filtered_data = [df5]

    data_segment_entropy = calculate_segment_entropy(filtered_data)

    print(data_segment_entropy)