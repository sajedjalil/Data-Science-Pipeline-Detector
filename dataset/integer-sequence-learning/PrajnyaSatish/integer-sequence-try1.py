# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train_df= pd.read_csv('../input/train.csv', index_col="Id", nrows=100)
test_df = pd.read_csv('../input/test.csv', index_col="Id", nrows=100)

train_df= train_df['Sequence'].to_dict()
test_df= test_df['Sequence'].to_dict()
seqs={0: [1 for x in range(0,400)]}

#for key in train_df:
#    print(str(key) +':')
#    print(train_df[key])
#    print()
#    print()


toplotseq = train_df[3]
print('To plot')
print(toplotseq)
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()










from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.