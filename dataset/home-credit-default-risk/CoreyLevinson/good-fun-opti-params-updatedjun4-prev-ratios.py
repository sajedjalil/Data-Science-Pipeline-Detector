import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc

gc.enable()
pd.set_option('display.max_columns', None)

buro_bal = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')
print('Buro bal shape : ', buro_bal.shape)

print('transform to dummies')
#Concatenate dummies to df, and rename them as buro_bal_status_A
#Axis=1 refers to doing the operation on the columns
buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS', axis=1)

print('Counting buros')
#SELECT COUNT(SK_ID_BRUEAU) GROUP BY SK_ID_BUREAU FROM DATAFRAME [[SK_ID_BUREAU, MONTHS_BALANCE]]
buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
#Create new column buro_count by merging buro_counts with buro_bal
buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])

print('averaging buro bal')
#Take the mean of everything. So you are taking the means of the Statuses, and also the means of the counts, and also the means of the MONTH_BALANCE.
avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()

#Rename the columns for aesthetic reasons
avg_buro_bal.columns = ['avg_buro_bal_' + f_ for f_ in avg_buro_bal.columns]
del buro_bal
gc.collect()

print('Read Bureau')
buro = pd.read_csv('../input/home-credit-default-risk/bureau.csv')

print('Go to dummies')
#Create dummies for the three character columns
buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')

#Concatenate dummies onto full df
buro_full = pd.concat([buro, buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum], axis=1)
# buro_full.columns = ['buro_' + f_ for f_ in buro_full.columns]

del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
gc.collect()

print('Merge with buro avg')
buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', on='SK_ID_BUREAU', suffixes=('', '_bur_bal'))

print('Counting buro per SK_ID_CURR')
#Get count of unique bureaus for each SK_ID_CURR
nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()

#Rename the number SK_ID_BUREAU
buro_full['Num_Unique_Bureaus'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

print('Averaging bureau')
avg_buro = buro_full.groupby('SK_ID_CURR').mean()
avg_buro.columns = ['avg_buro_' + f_ for f_ in avg_buro.columns]

print('Summing bureau')
#sum_buro = buro_full[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM','ca__Active','ca__Bad debt','ca__Closed','ca__Sold','CREDIT_DAY_OVERDUE']].groupby('SK_ID_CURR').sum()
sum_buro = buro_full.groupby('SK_ID_CURR').sum()
sum_buro.columns = ['sum_buro_' + f_ for f_ in sum_buro.columns]
avg_buro = avg_buro.merge(right=sum_buro.reset_index(), how='left', on='SK_ID_CURR')

print('Std Dev bureau')
std_buro = buro_full.groupby('SK_ID_CURR').std()
std_buro.columns = ['std_buro_' + f_ for f_ in std_buro.columns]
avg_buro = avg_buro.merge(right=std_buro.reset_index(), how='left', on='SK_ID_CURR')

del buro, buro_full, sum_buro, std_buro, nb_bureau_per_curr
gc.collect()

print('Read prev')
prev = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')

prev_cat_features = [
    f_ for f_ in prev.columns if prev[f_].dtype == 'object'
]

print('Go to dummies')
prev_dum = pd.DataFrame()
for f_ in prev_cat_features:
    prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)

prev = pd.concat([prev, prev_dum], axis=1)

del prev_dum
gc.collect()

print('Counting number of Prevs')
nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])

print('Averaging prev')
avg_prev = prev.groupby('SK_ID_CURR').mean()
avg_prev.columns = ['avg_prev_' + f_ for f_ in avg_prev.columns]

print('Summing prev')
sum_prev = prev.groupby('SK_ID_CURR').sum()
sum_prev.columns = ['sum_prev_' + f_ for f_ in sum_prev.columns]
avg_prev = avg_prev.merge(right=sum_prev.reset_index(), how='left', on='SK_ID_CURR')

print('Std prev')
std_prev = prev.groupby('SK_ID_CURR').std()
std_prev.columns = ['std_prev_' + f_ for f_ in std_prev.columns]
avg_prev = avg_prev.merge(right=std_prev.reset_index(), how='left', on='SK_ID_CURR')

del prev, sum_prev, std_prev, nb_prev_per_curr
gc.collect()

print('Reading POS_CASH')
pos = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')

print('Go to dummies')
pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)

print('Counting number of prevs per curr')
nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

print('Averaging POS')
avg_pos = pos.groupby('SK_ID_CURR').mean()
avg_pos.columns = ['avg_pos_' + f_ for f_ in avg_pos.columns]

print('Summing POS')
sum_pos = pos.groupby('SK_ID_CURR').sum()
sum_pos.columns = ['sum_pos_' + f_ for f_ in sum_pos.columns]
avg_pos = avg_pos.merge(right=sum_pos.reset_index(), how='left', on='SK_ID_CURR')

print('Std POS')
std_pos = pos.groupby('SK_ID_CURR').std()
std_pos.columns = ['std_pos_' + f_ for f_ in std_pos.columns]
avg_pos = avg_pos.merge(right=std_pos.reset_index(), how='left', on='SK_ID_CURR')

del pos, nb_prevs, sum_pos, std_pos
gc.collect()

print('Reading CC balance')
cc_bal = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv')

print('Go to dummies')
cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')], axis=1)

nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

print('Averaging CC_Bal')
avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
avg_cc_bal.columns = ['avg_cc_bal_' + f_ for f_ in avg_cc_bal.columns]

print('Summing CC_Bal')
sum_cc_bal = cc_bal.groupby('SK_ID_CURR').sum()
sum_cc_bal.columns = ['sum_cc_bal_' + f_ for f_ in sum_cc_bal.columns]
avg_cc_bal = avg_cc_bal.merge(right=sum_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

print('Std CC_Bal')
std_cc_bal = cc_bal.groupby('SK_ID_CURR').std()
std_cc_bal.columns = ['std_cc_bal_' + f_ for f_ in std_cc_bal.columns]
avg_cc_bal = avg_cc_bal.merge(right=std_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

del cc_bal, nb_prevs, sum_cc_bal, std_cc_bal
gc.collect()

print('Reading Installments')
inst = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')
nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

print('Averaging inst')
avg_inst = inst.groupby('SK_ID_CURR').mean()
avg_inst.columns = ['avg_inst_' + f_ for f_ in avg_inst.columns]

print('Summing inst')
sum_inst = inst.groupby('SK_ID_CURR').sum()
sum_inst.columns = ['sum_inst_' + f_ for f_ in sum_inst.columns]
avg_inst = avg_inst.merge(right=sum_inst.reset_index(), how='left', on='SK_ID_CURR')

print('Std inst')
std_inst = inst.groupby('SK_ID_CURR').std()
std_inst.columns = ['std_inst_' + f_ for f_ in std_inst.columns]
avg_inst = avg_inst.merge(right=std_inst.reset_index(), how='left', on='SK_ID_CURR')

del inst, nb_prevs, sum_inst, std_inst
gc.collect()

print('Read data and test')
data = pd.read_csv('../input/jun4-feats/jun4-tr.csv')
test = pd.read_csv('../input/jun4-feats/jun4-te.csv')
print('Shapes : ', data.shape, test.shape)

y = data['TARGET']
del data['TARGET']

del data['AMT_REQ_CREDIT_BUREAU_total_week']
del data['AMT_REQ_CREDIT_BUREAU_total']
del data['is_na_car_age']
del data['FLAG_CAR_NO_REALTY']
del data['income_div_child']

del test['AMT_REQ_CREDIT_BUREAU_total_week']
del test['AMT_REQ_CREDIT_BUREAU_total']
del test['is_na_car_age']
del test['FLAG_CAR_NO_REALTY']
del test['income_div_child']


categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]
categorical_feats
for f_ in categorical_feats:
    data[f_], indexer = pd.factorize(data[f_])
    test[f_] = indexer.get_indexer(test[f_])
    
data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

del avg_buro, avg_prev, avg_pos, avg_cc_bal, avg_inst
gc.collect()

prevfeats = pd.read_csv('../input/prevfeats/prev-feats.csv')
data = data.merge(right=prevfeats.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=prevfeats.reset_index(), how='left', on='SK_ID_CURR')

#incomediffs = pd.read_csv('../input/incomediffs/income_diffs.csv')
#data = data.merge(right=incomediffs.reset_index(), how='left', on='SK_ID_CURR')
#test = test.merge(right=incomediffs.reset_index(), how='left', on='SK_ID_CURR')

del prevfeats#, incomediffs
gc.collect()

rats = pd.read_csv('../input/ratios/them_ratios.csv')
data = data.merge(right=rats.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=rats.reset_index(), how='left', on='SK_ID_CURR')
del rats
gc.collect()

#print('reading in newstuff')
#Feature Engineered stuff from application dataset
#newstuff = pd.read_csv('../input/featureengv2/newstuffv2.csv')
#data = data.merge(right=newstuff, how='left',on='SK_ID_CURR')
#test = test.merge(right=newstuff, how='left',on='SK_ID_CURR')

#del newstuff
#gc.collect()
#print('reading in isnan')
#nainfo2 = pd.read_csv('../input/isnanloanv2/nainfo2.csv')
#data = data.merge(right=nainfo2, how='left',on='SK_ID_CURR')
#test = test.merge(right=nainfo2, how='left',on='SK_ID_CURR')
#del nainfo2
#gc.collect()

from lightgbm import LGBMClassifier
import gc

gc.enable()

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4321)
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

feature_importance_df = pd.DataFrame()

feats = [f for f in data.columns if f not in ['SK_ID_CURR']]

#upsample=True

print('Reached fold loop')

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data, y)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]
    
    clf = LGBMClassifier(
        # n_estimators=1000,
        # num_leaves=20,
        # colsample_bytree=.8,
        # subsample=.8,
        # max_depth=7,
        # reg_alpha=.1,
        # reg_lambda=.1,
        # min_split_gain=.01
        n_estimators=5000, #5000
        learning_rate=0.02,
        num_leaves=13,
        colsample_bytree=.9234,
        subsample=.855,
        max_depth=-1,
        reg_alpha=2,
        reg_lambda=5,
        min_child_weight=.1,
        silent=-1,
        verbose=-1,
        #zero_as_missing=True No improvement
        min_child_samples=399,
    )
    
    #Upsampling from https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    #if upsample:
    #    pos = pd.Series(trn_y == 1)
    #    # Add positive examples
    #    trn_x = pd.concat([trn_x, trn_x.loc[pos]], axis=0)
    #    trn_y = pd.concat([trn_y, trn_y.loc[pos]], axis=0)
    #    # Shuffle data
    #    idx = np.arange(len(trn_x))
    #    np.random.shuffle(idx)
    #    trn_x = trn_x.iloc[idx]
    #    trn_y = trn_y.iloc[idx]
 
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=300, early_stopping_rounds=300  #30
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    #fold_importance_df["feature"] = feats
    fold_importance_df["feature"] = np.concatenate((feats, ['a','b','c','d','e','f','g','h','i','j','k','l']))
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
    
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 

test['TARGET'] = sub_preds

test[['SK_ID_CURR', 'TARGET']].to_csv('good_fun_with_is_nan.csv', index=False)

# Plot feature importances
cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:100].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(8,10))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

# Plot ROC curves
plt.figure(figsize=(6,6))
scores = [] 
for n_fold, (_, val_idx) in enumerate(folds.split(data, y)):  
    # Plot the roc curve
    fpr, tpr, thresholds = roc_curve(y.iloc[val_idx], oof_preds[val_idx])
    score = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
    scores.append(score)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
fpr, tpr, thresholds = roc_curve(y, oof_preds)
score = roc_auc_score(y, oof_preds)
plt.plot(fpr, tpr, color='b',
         label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LightGBM ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig('roc_curve.png')

# Plot ROC curves
plt.figure(figsize=(6,6))
precision, recall, thresholds = precision_recall_curve(y, oof_preds)
score = roc_auc_score(y, oof_preds)
plt.plot(precision, recall, color='b',
         label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('LightGBM Recall / Precision')
plt.legend(loc="best")
plt.tight_layout()

plt.savefig('recall_precision_curve.png')