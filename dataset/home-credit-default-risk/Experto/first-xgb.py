import pandas as pd

import numpy as np



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold



import xgboost as xgb

from xgboost import XGBClassifier



import gc



from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from sklearn.feature_selection import VarianceThreshold





print('Importing data...')

data = pd.read_csv('../input/application_train.csv')

test = pd.read_csv('../input/application_test.csv')

prev = pd.read_csv('../input/previous_application.csv')

buro = pd.read_csv('../input/bureau.csv')

buro_balance = pd.read_csv('../input/bureau_balance.csv')

credit_card  = pd.read_csv('../input/credit_card_balance.csv')

POS_CASH  = pd.read_csv('../input/POS_CASH_balance.csv')

payments = pd.read_csv('../input/installments_payments.csv')

lgbm_submission = pd.read_csv('../input/sample_submission.csv')



#Separate target variable

y = data['TARGET']

del data['TARGET']



#Feature engineering

#data['loan_to_income'] = data.AMT_ANNUITY/data.AMT_INCOME_TOTAL

#test['loan_to_income'] = test.AMT_ANNUITY/test.AMT_INCOME_TOTAL



#One-hot encoding of categorical features in data and test sets

categorical_features = [col for col in data.columns if data[col].dtype == 'object']



one_hot_df = pd.concat([data,test])

one_hot_df = pd.get_dummies(one_hot_df, columns=categorical_features)



data = one_hot_df.iloc[:data.shape[0],:]

test = one_hot_df.iloc[data.shape[0]:,]



#Pre-processing buro_balance

print('Pre-processing buro_balance...')

buro_grouped_size = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()

buro_grouped_max = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()

buro_grouped_min = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()



buro_counts = buro_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)

buro_counts_unstacked = buro_counts.unstack('STATUS')

buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X',]

buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size

buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min

buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max



buro = buro.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')



#Pre-processing previous_application

print('Pre-processing previous_application...')

#One-hot encoding of categorical features in previous application data set

prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']

prev = pd.get_dummies(prev, columns=prev_cat_features)

avg_prev = prev.groupby('SK_ID_CURR').mean()

cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()

avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']

del avg_prev['SK_ID_PREV']



#Pre-processing buro

print('Pre-processing buro...')

#One-hot encoding of categorical features in buro data set

buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']

buro = pd.get_dummies(buro, columns=buro_cat_features)

avg_buro = buro.groupby('SK_ID_CURR').mean()

avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']

del avg_buro['SK_ID_BUREAU']



#Pre-processing POS_CASH

print('Pre-processing POS_CASH...')

le = LabelEncoder()

POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))

nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()

nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()

POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']

POS_CASH['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']

POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)



#Pre-processing credit_card

print('Pre-processing credit_card...')

credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))

nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()

nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()

credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']

credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']

credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)



#Pre-processing payments

print('Pre-processing payments...')

avg_payments = payments.groupby('SK_ID_CURR').mean()

avg_payments2 = payments.groupby('SK_ID_CURR').max()

avg_payments3 = payments.groupby('SK_ID_CURR').min()

del avg_payments['SK_ID_PREV']



#Join data bases

print('Joining databases...')

data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')



data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')



data = data.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')



data = data.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')



data = data.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')



data = data.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')



data = data.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')



#Remove features with many missing values

print('Removing features with more than 80% missing...')

test = test[test.columns[data.isnull().mean() < 0.85]]

data = data[data.columns[data.isnull().mean() < 0.85]]





excluded_feats = ['SK_ID_CURR']

features = [f_ for f_ in data.columns if f_ not in excluded_feats]



folds = KFold(n_splits=4, shuffle=True, random_state=546789)

oof_preds = np.zeros(data.shape[0])

sub_preds = np.zeros(test.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):

    trn_x, trn_y = data[features].iloc[trn_idx], y.iloc[trn_idx]

    val_x, val_y = data[features].iloc[val_idx], y.iloc[val_idx]

    

    clf = XGBClassifier(

        objective = 'binary:logistic', 

        booster = "gbtree",

        eval_metric = 'auc', 

        nthread = 4,

        eta = 0.05,

        gamma = 0,

        max_depth = 6, 

        subsample = 0.7, 

        colsample_bytree = 0.7, 

        colsample_bylevel = 0.675,

        min_child_weight = 22,

        alpha = 0,

        random_state = 42, 

        nrounds = 2000,
        
        n_estimators=2000

    )



    clf.fit(trn_x, trn_y, eval_set= [(trn_x, trn_y), (val_x, val_y)], verbose=10, early_stopping_rounds=30)

    

    oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]

    sub_preds += clf.predict_proba(test[features])[:, 1] / folds.n_splits

    

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))

    del clf, trn_x, trn_y, val_x, val_y

    gc.collect()

    

print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))   



test['TARGET'] = sub_preds



test[['SK_ID_CURR', 'TARGET']].to_csv('xgb_submission_esi.csv', index=False, float_format='%.8f')