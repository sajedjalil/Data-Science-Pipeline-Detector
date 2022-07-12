__author__ = 'lbernardi'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

# Calculate the bins, code from https://github.com/lucjb/pydata/blob/master/HeadTailBreaks.py
# Look to the video of Lucas Bernardi at PyData16: https://www.youtube.com/watch?v=HS7mObQttxU

def plot_htb(values, breaks):
    breaks.append(min(values))
    breaks.append(max(values))
    breaks.sort()
    plt.xlabel('Number of var15 Age')
    plt.ylabel('var15 (age)')
    plt.plot(values)
    f, _ = np.histogram(values, bins=breaks)
    plt.hist(values, bins=breaks, orientation='horizontal')
    plt.plot(values, color='red')
    plt.xticks(f, rotation=45)
    plt.yticks(breaks)
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.grid()
    plt.show()
    plt.savefig("binning_var15.png", format='png')


def htb(data, tails=[], breaks=[]):
    if len(data)==0:
        return tails, breaks
    mean = sum([d[1] for d in data])/len(data)
    head = []
    tail = []
    for x, size in data:
        if size > mean:
            head.append((x, size))
        else:
            tail.append((x, size))

    if len(head)>=len(data):
	    return tails, breaks

    tails.append(tail)
    breaks.append(mean)
    return htb(head, tails, breaks)

def head_tail_breaks(data):
    tails, breaks =  htb(data, [], [])
    return breaks

def head_tail_breaks_encode(X, col):
    values = sorted(X.loc[:,col].values, reverse=True)
    rank_size = list(zip(range(1,len(values)+1), values))
    breaks = head_tail_breaks(rank_size)
    if len(breaks)<1:
	    return X, breaks
    digitized_values = np.digitize(values, bins=breaks)
    digitized_values = digitized_values.reshape((-1,1))
    enc = OneHotEncoder(sparse=False)
    ohe_values =  enc.fit_transform(digitized_values)
    X_new = np.append(X, digitized_values, axis=1)

    return X_new, breaks
    
X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)
    
# Select the features calculated in https://www.kaggle.com/cast42/santander-customer-satisfaction/select-features-rfecv/code
# 
features = \
['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1', 
'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 
'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'ind_var8_0', 'ind_var30', 'num_var4',
'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_ult1', 
'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var37', 
'num_var39_0', 'num_var42', 'saldo_var5', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 
'imp_trans_var37_ult1', 'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 
'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 
'num_meses_var39_vig_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var41_comer_ult1',
'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult3', 
'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3',
'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'var38', 'n0']

X_train, X_test, y_train, y_test = \
  cross_validation.train_test_split(X[features], y, random_state=1301, stratify=y, test_size=0.35)
ratio = float(np.sum(y == 1)) / np.sum(y==0)
clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 7,
                n_estimators=1000,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                scale_pos_weight = ratio,
                seed=1301)
                
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_test, y_test)])
        
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X[features], ntree_limit=clf.best_iteration)[:,1]))

X_bin, breaks = head_tail_breaks_encode(X[features], 'var15')

print ('Break of var 15')
print (breaks)

X_train, X_test, y_train, y_test = \
   cross_validation.train_test_split(X_bin, y, random_state=1301, stratify=y, test_size=0.35)
ratio = float(np.sum(y == 1)) / np.sum(y==0)
clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 7,
                n_estimators=1000,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                scale_pos_weight = ratio,
                seed=1301)
                
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_test, y_test)])
        
print('Overall AUC after binning var15 :', roc_auc_score(y, clf.predict_proba(X_bin, ntree_limit=clf.best_iteration)[:,1]))

#Plots the most interesting feature with breaks and a histogram.
values = sorted(X['var15'].values, reverse=True)
plot_htb(values, breaks)




