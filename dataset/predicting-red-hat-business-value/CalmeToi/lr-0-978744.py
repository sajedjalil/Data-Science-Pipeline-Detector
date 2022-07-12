import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
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

def tt_split(train,y):
    ppl = train.people_id.as_matrix()
    ppl_unique = train.people_id.unique()
    mask_tt, val = train_test_split(ppl_unique,\
            test_size=0.3, random_state=777)
    mask =  np.in1d(ppl, mask_tt)
    
    return mask, y
def main():
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
    lootrain['outcome'] = train.outcome.as_matrix()
    lootrain['people_id'] = train.people_id.as_matrix()
    lootrain = lootrain.drop_duplicates()
    #lootrain = lootrain.drop()
    print(lootrain.shape)
    
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
    print(lootest.shape)
    mask, y = tt_split(lootrain, lootrain['outcome'].as_matrix())
    lr = LogisticRegression(C=100000.0)
    lr.fit(lootrain[['group_1', 'char_2', 'char_38']][mask], y[mask])
    preds = lr.predict_proba(lootrain[['group_1', 'char_2', 'char_38']][~mask])[:, 1]
    pred_val = pd.DataFrame(columns = ['pred_val', 'y_val'])
    print(roc_auc_score(y[~mask], preds))
    pred_val['pred_val'] = np.array(preds)
    pred_val['y_val'] = y[~mask]
    pred_val.to_csv('pred_val.csv', index=False)
    lr.fit(lootrain[['group_1', 'char_2', 'char_38']][~mask], y[~mask])
    preds = lr.predict_proba(lootrain[['group_1', 'char_2', 'char_38']][mask])[:, 1]
    pred_train = pd.DataFrame(columns = ['pred_train', 'y_trainl'])
    pred_train['pred_train'] = np.array(preds)
    pred_train['y_train'] = y[mask]
    pred_train.to_csv('pred_train.csv', index=False)
    #print('roc', roc_auc_score(train.outcome, preds))
   
    lr.fit(lootrain[['group_1', 'char_2', 'char_38']], lootrain['outcome'])
    preds = lr.predict_proba(lootest[['group_1', 'char_2', 'char_38']])[:, 1]
    submission = pd.DataFrame()
    submission['activity_id'] = activity_id
    submission['outcome'] = preds
    submission.to_csv('LR_pred.csv', index=False)
    
    mask_df = pd.DataFrame(mask, columns = ['mask'])
    mask_df.to_csv('mask.csv', index = 0)
    


if __name__ == "__main__":
    print('Started')
    main()
    print('Finished')
