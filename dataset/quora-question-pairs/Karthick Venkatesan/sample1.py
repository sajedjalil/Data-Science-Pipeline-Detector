# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

'''
from nltk.corpus import stopwords
import pandas as pd


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row[0]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row[1]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    return (0.5*len(shared_words_in_q1)/len(q1words) + 0.5*len(shared_words_in_q2)/len(q2words))

test = pd.DataFrame.from_csv("../input/test.csv")
stops = set(stopwords.words("english"))
test["is_duplicate"] = test.apply(word_match_share, axis=1, raw=True)
test["is_duplicate"].to_csv("count_words_benchmark.csv", header=True)

'''
df_train = pd.read_csv("../input/train.csv")
#print(df_train.head())
p =  df_train['is_duplicate'].mean()
df_test = pd.read_csv("../input/test.csv")
v=0.0
sub1 = pd.DataFrame({'test_id':df_test['test_id'],'is_duplicate':v})
sub1.to_csv('sub1.csv',index=False) 
