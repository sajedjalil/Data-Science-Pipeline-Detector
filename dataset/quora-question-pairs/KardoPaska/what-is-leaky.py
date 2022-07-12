# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd
import numpy as np
import time, os, pickle
from string import punctuation
from collections import defaultdict
from tqdm import tqdm

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv(r'../input/train.csv', encoding="Latin-1", nrows=100000)
test_df = pd.read_csv(r'../input/test.csv', encoding="Latin-1", nrows=1000)

ques = pd.concat([train_df[['question1', 'question2']], \
        test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)

for i in tqdm(range(ques.shape[0])):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

def q1_freq(row):
    return(len(q_dict[row['question1']]))
    
def q2_freq(row):
    return(len(q_dict[row['question2']]))
    
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


tqdm.pandas(desc="my bar!")
train_df['q1_q2_intersect'] = train_df.progress_apply(q1_q2_intersect, axis=1, raw=True)
train_df['q1_freq'] = train_df.progress_apply(q1_freq, axis=1, raw=True)
train_df['q2_freq'] = train_df.progress_apply(q2_freq, axis=1, raw=True)

test_df['q1_q2_intersect'] = test_df.progress_apply(q1_q2_intersect, axis=1, raw=True)
test_df['q1_freq'] = test_df.progress_apply(q1_freq, axis=1, raw=True)
test_df['q2_freq'] = test_df.progress_apply(q2_freq, axis=1, raw=True)

#leaks = train_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
#test_leaks = test_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]


qid1_most_freq = train_df.loc[train_df['q1_freq'].idxmax()]['qid1']
print(train_df[train_df['qid1'] == qid1_most_freq])

print('um...')
test_1_int_idx = test_df['q1_q2_intersect'].idxmax()
testq1_q2_intersect_most_freq = test_df.loc[test_1_int_idx]
print(test_1_int_idx)
print(testq1_q2_intersect_most_freq)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@r\n'*4)
print(test_df[test_df['question1'] == test_df['question1'].loc[test_1_int_idx]])

#print(test_df.loc[testq1_q2_intersect_most_freq])
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\r\n'*4)
#q1_freq_count_most_common = train_df['q1_freq'].nlargest(n=3, keep='first').index.values
q1_freq_count_most_common = train_df.nlargest(n=1000, columns='q1_freq', keep="last")['question1'].unique()
print(q1_freq_count_most_common[:10])

for each_mostcom_q1 in q1_freq_count_most_common[:5]:
    #print(train_df[train_df['question1'] == each_mostcom_q1])
    print(train_df[np.logical_or(train_df['question1'] == each_mostcom_q1, train_df['question2'] == each_mostcom_q1)])
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\r\n'*4)
    

#train_df['q1_freq']

#pprint(repr(qid1_count.most_common(10)))

