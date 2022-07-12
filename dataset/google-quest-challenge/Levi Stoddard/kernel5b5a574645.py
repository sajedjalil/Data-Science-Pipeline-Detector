# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


train_df = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')
train_df = train_df.drop(['question_user_page', 'answer_user_page', 'url'], axis=1)
predict_columns = train_df.columns[8:]

medians = train_df[predict_columns].median()

# try to apease the spearman coefficient.
import random
def addDeviation(x):
    return max(0, min(1, x + random.randrange(-1000, 1000)/1000/100))

test_df = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')
submission_df = pd.DataFrame(data={'qa_id':test_df['qa_id']})
for i, col in enumerate(predict_columns):
   submission_df.loc[:,col] = [addDeviation(x) for x in [medians[i]]*len(test_df)]
print(submission_df.head())

submission_df.to_csv('submission.csv', index=False)