# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from tqdm import tqdm

n_labels = 4716
label_cnt = np.zeros(n_labels, dtype=int)

# Read labels from train&validate set
#for fname in ('../input/train_labels.csv', '../input/validate_labels.csv'):
for fname in ():
    print('Read', fname)
    with open(fname) as f:
        for line in tqdm(f):
            for label in map(int, line.strip().split(',')[1].split(' ')):
                label_cnt[label] += 1

label_sum = label_cnt.sum()



print('There are {} labels in the train&validate set'.format(label_sum))
print(list(label_cnt[:20]))

# Write rates of most polular labels
n_preds = 20
print('Write submission')
with open('../input/sample_submission.csv') as f,\
     open('submission_{}.csv'.format(n_preds), 'w') as wf:
    wf.write(f.readline())
    for line in tqdm(f):
        wf.write('{},{}\n'.format(\
            line.split(',')[0],\
            ' '.join(['{} {:0.4f}'.format(i, label_cnt[i] / label_sum) for i in range(n_preds)]) ))