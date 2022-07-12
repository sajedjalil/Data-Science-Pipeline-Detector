# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import surprise

train = pd.read_csv('../input/train.csv')
print("Training Set Loaded")

algo = surprise.SVD()
reader = surprise.Reader(rating_scale=(0,1))
data = surprise.Dataset.load_from_df(train[['msno', 'song_id', 'target']].dropna(), reader)
trainset = data.build_full_trainset()
algo.train(trainset)
print("Done Training")

test = pd.read_csv('../input/test.csv')
submit = []
for index, row in test.iterrows():
    est = algo.predict(row['msno'], row['song_id']).est
    submit.append((row['id'], est))
submit = pd.DataFrame(submit, columns=['id', 'target'])
submit.to_csv('submission.csv', index=False)
print("Created submission.csv")
