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
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def LeaveOneOut(data1, data2, columnName, useLOO=False):
    grpOutcomes = data1.groupby(columnName).mean().reset_index()
    outcomes = data2['outcome'].values
    x = pd.merge(data2[[columnName, 'outcome']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['outcome']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
    return x.fillna(x.mean())


def main():
    directory = '../input/'
    train = pd.read_csv(directory+'act_train.csv')
    test = pd.read_csv(directory+'act_test.csv')
    people = pd.read_csv(directory+'people.csv')
    train = pd.merge(train, people,
                     how='left',
                     on='people_id',
                     left_index=True)
    train.fillna('-999', inplace=True)
    lootrain = pd.DataFrame()
    for col in train.columns:
        if(col != 'outcome' and col != 'people_id'):
            print(col)
            lootrain[col] = LeaveOneOut(train, train, col, True).values
    
    print('hello')
    from sklearn.linear_model import LogisticRegression
    #lr = LogisticRegression(C=100000.0)
    
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import Pipeline
    from sklearn.grid_search import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest, f_classif
    #rfe = RFE(lr, 10)
    #rfe = rfe.fit(lootrain, train['outcome'])
# summarize the selection of the attributes
    #print(rfe.support_)
    #print(rfe.ranking_)
    
    kbest = SelectKBest(f_classif)
    pipeline = Pipeline([('kbest', kbest), ('lr', LogisticRegression())])
    grid_search = GridSearchCV(pipeline, {'kbest__k': [1,2,3,4], 'lr__C': np.logspace(-10, 10, 5)})
    grid_search.fit(lootrain , train['outcome'])
    
    
    
    
    #skb = SelectKBest()
    #SKB = skb.fit(lootrain, train['outcome'])
    #print(SKB.get_params)
    #print(SKB.get_support)
    

if __name__ == "__main__":
    print('Started')
    main()
    print('Finished')
