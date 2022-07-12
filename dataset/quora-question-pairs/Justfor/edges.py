# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
import pandas as pd
import networkx as nx

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

df = pd.concat([train_df, test_df])


g = nx.Graph()
g.add_nodes_from(df.question1)
g.add_nodes_from(df.question2)
edges = list(df[['question1', 'question2']].to_records(index=False))
g.add_edges_from(edges)


def get_intersection_count(row):
    return(len(set(g.neighbors(row.question1)).intersection(set(g.neighbors(row.question2)))))

train_ic = pd.DataFrame()
test_ic = pd.DataFrame()


train_df['intersection_count'] = train_df.apply(lambda row: get_intersection_count(row), axis=1)
test_df['intersection_count'] = test_df.apply(lambda row: get_intersection_count(row), axis=1)
train_ic['intersection_count'] = train_df['intersection_count']
test_ic['intersection_count'] = test_df['intersection_count']

train_ic.to_csv("train_ic.csv", index=False)
test_ic.to_csv("test_ic.csv", index=False)
