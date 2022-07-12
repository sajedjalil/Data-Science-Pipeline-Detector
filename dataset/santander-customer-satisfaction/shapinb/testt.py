import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

ind_var_columns = ["ind_var1", "ind_var2", "ind_var5", "ind_var6", "ind_var8", "ind_var12", "ind_var13",
                   "ind_var14",
                   "ind_var17", "ind_var18", "ind_var19", "ind_var20", "ind_var24", "ind_var26", "ind_var25",
                   "ind_var28", "ind_var27", "ind_var29", "ind_var30", "ind_var31", "ind_var32", "ind_var33",
                   "ind_var34", "ind_var37", "ind_var40", "ind_var41", "ind_var39", "ind_var44", "ind_var46"]

df_train['nonzero_ind_var_count'] = (df_train[ind_var_columns] > 0).sum(1)
df_test['nonzero_ind_var_count'] = (df_test[ind_var_columns] > 0).sum(1)

saldo_var_columns = ["saldo_var1", "saldo_var5", "saldo_var6", "saldo_var8", "saldo_var12", "saldo_var13_medio",
                     "saldo_var13", "saldo_var14", "saldo_var17", "saldo_var18", "saldo_var20", "saldo_var24",
                     "saldo_var26", "saldo_var25", "saldo_var28", "saldo_var27", "saldo_var29", "saldo_var30",
                     "saldo_var31", "saldo_var32", "saldo_var33", "saldo_var34", "saldo_var37", "saldo_var40",
                     "saldo_var41", "saldo_var42", "saldo_var44", "saldo_var46"]

df_train['nonzero_saldo_var_count'] = (df_train[saldo_var_columns] > 0).sum(1)
df_test['nonzero_saldo_var_count'] = (df_test[saldo_var_columns] > 0).sum(1)


delta_imp_columns = ["delta_imp_amort_var18_1y3", "delta_imp_amort_var34_1y3", "delta_imp_aport_var13_1y3",
                     "delta_imp_aport_var17_1y3", "delta_imp_aport_var33_1y3", "delta_imp_compra_var44_1y3",
                     "delta_imp_reemb_var13_1y3", "delta_imp_reemb_var17_1y3", "delta_imp_reemb_var33_1y3",
                     "delta_imp_trasp_var17_in_1y3", "delta_imp_trasp_var17_out_1y3",
                     "delta_imp_trasp_var33_in_1y3", "delta_imp_trasp_var33_out_1y3",
                     "delta_imp_venta_var44_1y3", "delta_num_aport_var13_1y3", "delta_num_aport_var17_1y3",
                     "delta_num_aport_var33_1y3", "delta_num_compra_var44_1y3", "delta_num_reemb_var13_1y3",
                     "delta_num_reemb_var17_1y3", "delta_num_reemb_var33_1y3", "delta_num_trasp_var17_in_1y3",
                     "delta_num_trasp_var17_out_1y3", "delta_num_trasp_var33_in_1y3",
                     "delta_num_trasp_var33_out_1y3", "delta_num_venta_var44_1y3"]

df_train['delta_imp_columns_count'] = (df_train[delta_imp_columns] > 0).sum(1)
df_test['delta_imp_columns_count'] = (df_test[delta_imp_columns] > 0).sum(1)

# Balance > avg
# df_train['balance_greater_than_avg'] = (df_train[saldo_var_columns] > df_train[saldo_var_columns].mean()).sum(1)
# df_test['balance_greater_than_avg'] = (df_test[saldo_var_columns] > df_test[saldo_var_columns].mean()).sum(1)

def get_normalized(df, column):
    mean = df[column].mean()
    std = df[column].std()
    return (df[column] - mean) / std

# df_train['saldo_var30_normalized'] = get_normalized(df_train, 'saldo_var30')
# df_test['saldo_var30_normalized'] = get_normalized(df_test, 'saldo_var30')

df_train['var38_normalized'] = get_normalized(df_train, 'var38')
df_test['var38_normalized'] = get_normalized(df_test, 'var38')

df_train['saldo_var5_normalized'] = get_normalized(df_train, 'saldo_var5')
df_test['saldo_var5_normalized'] = get_normalized(df_test, 'saldo_var5')

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns
for i in range(len(c) - 1):
    v = df_train[c[i]].values
    for j in range(i + 1, len(c)):
        if np.array_equal(v, df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

df_train = df_train.replace(-999999, 0)
df_test = df_test.replace(-999999, 0)

df_train = df_train.replace(9999999999, 0)
df_test = df_test.replace(9999999999, 0)

# Income / Age
df_train['mean_wage_to_age'] = df_train['ind_var13'] / df_train['var15']
df_test['mean_wage_to_age'] = df_test['ind_var13'] / df_test['var15']

# var38 / Age
df_train['var38_over_age'] = df_train['var38'] / df_train['var15']
df_test['var38_over_age'] = df_test['var38'] / df_test['var15']

# var38 / Income
df_train['var38_over_income'] = df_train['var38'] / (df_train['ind_var13'] + 0.01)
df_test['var38_over_income'] = df_test['var38'] / (df_test['ind_var13'] + 0.01)

# ------------------------------
# Feature Selection
# ------------------------------
p = 77
X_train = df_train.drop(['TARGET','ID'],axis=1)
y_train = df_train['TARGET']

X_bin = Binarizer().fit_transform(scale(X_train))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y_train)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X_train, y_train)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X_train.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X_train.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X_train.columns, selected) if s]
print (features)

df_train['nonzeros'] = (df_train[features] > 0).sum(1)
df_test['nonzeros'] = (df_test[features] > 0).sum(1)

df_train['zeros'] = (df_train[features] == 0).sum(1)
df_test['zeros'] = (df_test[features] == 0).sum(1)

df_train['ones'] = (df_train[features] == 1).sum(1)
df_test['ones'] = (df_test[features] == 1).sum(1)

features = features + ['zeros', 'ones', 'nonzeros']

id_test = df_test.ID
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID', 'TARGET'], axis=1)[features].values
X_test = df_test.drop(['ID'], axis=1)[features].values

clf = xgb.XGBClassifier(
            missing=np.nan,
            gamma=0.8,
            max_depth=5,
            n_estimators=350,
            learning_rate=0.03,
            nthread=-1,
            subsample=0.75,
            colsample_bytree=0.75,
            seed=12345)
            
# clf = xgb.XGBClassifier(
#             missing=np.nan,
#             gamma=0.8,
#             max_depth=5,
#             n_estimators=350,
#             learning_rate=0.03,
#             nthread=-1,
#             subsample=0.75,
#             colsample_bytree=0.75,
#             seed=12345)
            
cv_folds = 5

xgb_param = clf.get_xgb_params()
xgtrain = xgb.DMatrix(X_train, label=y_train)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
                  early_stopping_rounds=20, metrics=['auc'], maximize=True, verbose_eval=True, show_stdv=False)
clf.set_params(n_estimators=cvresult.shape[0])
clf.fit(X_train, y_train, eval_metric='auc')

y_pred = clf.predict_proba(X_test)[:, 1]

print("Writing Submission")
submission = pd.DataFrame({"ID": id_test, "TARGET": y_pred})
submission.to_csv("submission.csv", index=False)

mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())
# ts.index = ts.reset_index()['index'].map(mapFeat)
ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)