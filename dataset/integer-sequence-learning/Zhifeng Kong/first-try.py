import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
traindata = pd.read_csv('../input/train.csv',header = 0)
print(np.shape(traindata))
print(traindata)