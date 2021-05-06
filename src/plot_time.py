import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Thu Dao tdao@umass.edu
if __name__ == '__main__':
    print('==============================================================================')
    print('Plot each feature vs time')
    data = pd.read_csv('./data/clean/batch146_17_clean.csv')
    data['color'] = np.where(data['label'] == 0, 'blue', 'red')
    for i in range(1, 20):
        data.plot(x="timestamp", y=data.columns[i], color=data['color'], lw=0.25)
        plt.savefig('./time_plot/' + data.columns[i] + '.png')
        plt.show()