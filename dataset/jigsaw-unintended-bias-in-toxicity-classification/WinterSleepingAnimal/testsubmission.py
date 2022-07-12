# Because of the connection always broke and reconnect from time to time, I choose to use my own computer
# to train and get the submission files from different checkpoints then upload fpr averaging !

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
submission_wfull = pd.read_csv('../input/submissionfilefromlongtimekl/submission_wf.csv')
submission_12000 = pd.read_csv('../input/predictionfromcheckpoints/submission_12000.csv')
submission_7000 = pd.read_csv('../input/predictionfromcheckpoints/submission_7000.csv')
submission_2000 = pd.read_csv('../input/predictionfromcheckpoints/submission.csv')
submission = pd.concat([submission_12000.iloc[:,0], submission_wfull.iloc[:,1]], axis=1)
# submission.iloc[:, 1] = np.array((0.85 * submission_wfull.iloc[:, 1]) +
#                                  (0.1 * submission_12000.iloc[:, 1]) +
#                                 (0.05 * submission_7000.iloc[:, 1]))

print("done")
submission.columns = ['id','prediction']
submission.to_csv('submission.csv', index=False, header=True)
submission.head()