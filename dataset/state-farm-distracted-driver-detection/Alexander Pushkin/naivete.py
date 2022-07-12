# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
img_list = pd.read_csv('../input/driver_imgs_list.csv')
print(img_list['classname'].value_counts())
probs = [0.110997146,0.10462005,0.103728148,0.103683553,0.103326793,0.103103817,0.101097039,0.094942918,0.089279344,0.085221192]
submission = pd.read_csv('../input/sample_submission.csv')
print(len(submission.values))
vecOnes = np.ones(79726)
header = list(submission)
for i in range(10):
    submission['c'+str(i)] = probs[i]*vecOnes
submission.to_csv('output.csv', columns = header,index=False)


# Any results you write to the current directory are saved as output.