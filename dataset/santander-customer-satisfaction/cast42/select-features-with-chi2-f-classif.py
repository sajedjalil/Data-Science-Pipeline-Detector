import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
training = training.replace(-999999,2)

print('There are {} features.'.format(training.shape[1]))

# Harcode result from caret in
# https://www.kaggle.com/sionek/santander-customer-satisfaction/reverse-feature-engineering/code

# constant_features= [
# "ind_var2_0",                    "ind_var2",                     
# "ind_var27_0",                   "ind_var28_0",                 
# "ind_var28",                     "ind_var27",                    
# "ind_var41",                     "ind_var46_0",                  
# "ind_var46",                     "num_var27_0",                 
# "num_var28_0",                   "num_var28",                    
# "num_var27",                     "num_var41",                    
# "num_var46_0",                   "num_var46",                    
# "saldo_var28",                   "saldo_var27",                  
# "saldo_var41",                   "saldo_var46",                  
# "imp_amort_var18_hace3",         "imp_amort_var34_hace3",        
# "imp_reemb_var13_hace3",         "imp_reemb_var33_hace3",        
# "imp_trasp_var17_out_hace3",     "imp_trasp_var33_out_hace3",    
# "num_var2_0_ult1",               "num_var2_ult1",                
# "num_reemb_var13_hace3",         "num_reemb_var33_hace3",        
# "num_trasp_var17_out_hace3",     "num_trasp_var33_out_hace3",    
# "saldo_var2_ult1",               "saldo_medio_var13_medio_hace3"]

# training.drop(constant_features, inplace=True, axis=1)
# test.drop(constant_features, inplace=True, axis=1)
# print('Dropped {} constant features. {} features remaining'.format(len(constant_features), training.shape[1]))

# function_of_other_features = [
# "imp_op_var39_comer_ult1", "imp_op_var39_comer_ult3",
# "imp_op_var39_efect_ult1", "imp_op_var39_efect_ult3",
# "imp_op_var39_ult1",       "ind_var13_0",            
# "ind_var13",               "num_var1_0",             
# "num_var1",                "num_var8_0",             
# "num_var8",                "num_var13_0",            
# "num_var14_0",             "num_var14",              
# "num_var17_0",             "num_var17",              
# "num_var20_0",             "num_var20",              
# "num_op_var41_ult3",       "num_op_var39_hace2",     
# "num_op_var39_ult3",       "num_var30_0",            
# "num_var31_0",             "num_var31",              
# "num_var33_0",             "num_var33",              
# "num_var40_0",             "num_var40",              
# "num_var39",               "num_var42_0",            
# "saldo_var1",              "saldo_var12",            
# "saldo_var13",             "saldo_var30",            
# "saldo_var31",             "saldo_var40",            
# "saldo_var42",             "num_var22_ult3",         
# "num_meses_var17_ult3",    "num_var45_ult3"]

# training.drop(function_of_other_features, inplace=True, axis=1)
# test.drop(function_of_other_features, inplace=True, axis=1)
# print('Dropped {} function of other features. {} features remaining'.format(len(function_of_other_features), training.shape[1]))


# correlated_features = [
# "ind_var29_0",                  "num_var6_0",                   
# "num_var29_0",                   "ind_var29",                    
# "num_var6",                      "num_var29",                    
# "num_var13_corto",               "ind_var13_medio",              
# "num_var13_medio_0",             "num_var13_medio",              
# "num_meses_var13_medio_ult3",    "ind_var18",                    
# "num_var18_0",                   "num_var18",                    
# "delta_imp_amort_var18_1y3",     "num_var24",                    
# "ind_var26"  ,                   "ind_var25",                    
# "ind_var32",                     "ind_var34",                    
# "num_var34_0",                   "num_var34",                    
# "delta_imp_amort_var34_1y3",     "ind_var37",            
# "ind_var39",                     "num_var44",                    
# "num_var26",                     "num_var25",                    
# "num_var32",                     "num_var37",                    
# "saldo_var29",                   "saldo_medio_var13_medio_ult1", 
# "imp_amort_var18_ult1",          "delta_num_aport_var13_1y3",    
# "delta_num_aport_var17_1y3",     "delta_num_aport_var33_1y3",    
# "delta_num_compra_var44_1y3",    "delta_num_reemb_var13_1y3",    
# "num_reemb_var13_ult1",          "delta_num_reemb_var17_1y3",    
# "delta_num_reemb_var33_1y3",     "imp_reemb_var33_ult1",         
# "num_reemb_var33_ult1",          "delta_num_trasp_var17_in_1y3", 
# "num_trasp_var17_in_ult1",       "delta_num_trasp_var17_out_1y3",
# "num_trasp_var17_out_ult1",      "delta_num_trasp_var33_in_1y3", 
# "delta_num_trasp_var33_out_1y3", "imp_trasp_var33_out_ult1",     
# "num_trasp_var33_out_ult1",      "delta_num_venta_var44_1y3",    
# "num_reemb_var17_hace3",         "num_trasp_var17_in_hace3",     
# "num_var7_emit_ult1",            "num_op_var39_efect_ult3"]

# training.drop(correlated_features, inplace=True, axis=1)
# test.drop(correlated_features, inplace=True, axis=1)

# print('Dropped {} highly correlated features. {} features remaining'.format(len(correlated_features), training.shape[1]))

X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

# Results after removing constant, linear combinations and high correlated features
# p = 50 103 features validation_0-auc:0.842372 LB: 0.831820
# p = 60 124 features validation_0-auc:0.842577
# p = 80 176 features validation_0-auc:0.845218
p = 85 # 192 features validation_0-auc:0.845515
# p = 87 194 features validation_0-auc:0.845541
# p = 90 204 features validation_0-auc:0.844807
# p = 95 220 features validation_0-auc:0.844879

# Let only chi2 and f_classif select features
# p = 80 283 features validation_0-auc:0.845573
# p = 84 # 302 features validation_0-auc:0.846003
# p = 85 306 features validation_0-auc:0.846130
p = 86 # 307 features validation_0-auc:0.846242
# p = 87 312 features validation_0-auc:0.845979
# p = 90 329 features validation_0-auc:0.845926


X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)

X_sel = X[features]

X_train, X_test, y_train, y_test = \
   cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.35)
# clf = xgb.XGBClassifier(max_depth   = 7,
#                 learning_rate       = 0.02,
#                 subsample           = 0.9,
#                 colsample_bytree    = 0.85,
#                 n_estimators        = 1000)
clf = xgb.XGBClassifier(max_depth = 5,
                n_estimators=525,
                learning_rate=0.02, 
                nthread=4,
                subsample=0.95,
                colsample_bytree=0.85, 
                seed=4242)
# clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
#         eval_set=[(X_test, y_test)])
        
# print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel)[:,1]))

clf.fit(X_sel, y, eval_metric="auc", eval_set=[(X_sel, y)])

print('Overall AUC on whole train set:', roc_auc_score(y, clf.predict_proba(X_sel)[:,1]))

test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]
y_pred = clf.predict_proba(sel_test)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())
#ts.index = ts.reset_index()['index'].map(mapFeat)
ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)



