# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

RANDOM_STATE=34

def get_x(df):
    df['Cabin'].fillna('Unknown', inplace=True)
    df['Embarked'].fillna('Unknown', inplace=True)
    df['Age'].fillna(-1., inplace=True)
    columns = list(df.columns)
    if 'Survived' in columns:
        columns.remove('Survived')
    columns.remove('PassengerId')
    return df[columns].values
    
def get_xy(df):
    X = get_x(df)
    y = df['Survived']
    return X, y
    

def main():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    
    X_train, y_train = get_xy(train)
    X_test = get_x(test)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                         y_train,
                                                         train_size=0.8, 
                                                         random_state=RANDOM_STATE,
                                                         shuffle=True,
                                                         stratify=y_train
                                                         )
    
    model = CatBoostClassifier(iterations=10000,
                               custom_loss=['Accuracy'], 
                               random_seed=RANDOM_STATE, 
                               logging_level='Verbose',
                               depth=3
                               )
    model.fit(X_train, 
              y_train,
              cat_features=[0,1,2,6,8,9],
              eval_set=(X_valid, y_valid),
              logging_level='Verbose'
    )
    
    sub = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':model.predict(X_test).astype(int)})
    sub.to_csv('cat_sub_1.csv',index=False)
    
if __name__=='__main__':
    main()