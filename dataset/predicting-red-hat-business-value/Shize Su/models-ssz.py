import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score
np.random.seed(8)


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
    print('Started')
    directory = '../input/'
    train = pd.read_csv(directory+'act_train.csv',
                        usecols=['people_id', 'outcome'])
    test = pd.read_csv(directory+'act_test.csv',
                       usecols=['activity_id', 'people_id'])
    people = pd.read_csv(directory+'people.csv',
                         usecols=['people_id',
                                  'group_1',
                                  'char_2',
                                  'char_38'])
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
    model = ExtraTreesRegressor(n_estimators=100,n_jobs=8,random_state=88) #  LogisticRegression(C=100000.0)
    model.fit(lootrain[['group_1', 'char_2', 'char_38']], train['outcome'])
    #preds = lr.predict_proba(lootrain[['group_1', 'char_2', 'char_38']])[:, 1]
    preds = model.predict(lootrain[['group_1', 'char_2', 'char_38']])
    print('roc', roc_auc_score(train.outcome, preds))
    print("roc score")
    test = pd.read_csv(directory+'act_test.csv',
                       usecols=['activity_id', 'people_id'])
    test = pd.merge(test, people,
                    how='left',
                    on='people_id',
                    left_index=True)
    test.fillna('-999', inplace=True)
    activity_id = test.activity_id.values
    test.drop('activity_id', inplace=True, axis=1)
    test['outcome'] = 0
    lootest = pd.DataFrame()
    for col in train.columns:
        if(col != 'outcome' and col != 'people_id'):
            print(col)
            lootest[col] = LeaveOneOut(train, test, col, False).values
    #preds = model.predict_proba(lootest[['group_1', 'char_2', 'char_38']])[:, 1]
    preds = model.predict(lootest[['group_1', 'char_2', 'char_38']])
    submission = pd.DataFrame()
    submission['activity_id'] = activity_id
    submission['outcome'] = preds
    submission.to_csv('Model_etr_v1_ssz.csv', index=False, float_format='%.5f')
    print('Finished')

if __name__ == "__main__":
    print('Started')
    main()
    print('Finished')
