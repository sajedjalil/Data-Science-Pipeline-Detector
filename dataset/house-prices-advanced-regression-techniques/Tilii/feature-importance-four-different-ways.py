import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import skew
    from datetime import datetime
    import xgboost as xgb
    import operator
    from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV, Lasso, LassoCV, LassoLars, LassoLarsCV, RandomizedLasso
    from sklearn.metrics import mean_squared_error
    from sklearn.cross_validation import cross_val_score
    #from sklearn.model_selection import cross_val_score

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    outfile.write('mapname,Feature\n')
    i = 0
    for feat in features:
        outfile.write('f{0},{1}\n'.format(i, feat))
        i = i + 1
    outfile.close()

# MOST OF THE DATA MANIPULATION CODE WAS BORROWED FROM https://www.kaggle.com/apapiu
# https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 10))
    return(rmse)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#print(train.head())
#pd.set_option('display.max_rows', 5000)
#pd.set_option('display.max_columns', 5000)

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
features = list(X_train.columns)
create_feature_map(features)
X_test = all_data[train.shape[0]:]
y = train.SalePrice

#Run RandomizedLasso
print("\n Running RandomizedLasso:")
model_rlasso = RandomizedLasso(alpha='bic', verbose=False, n_resampling=5000, random_state=1001, n_jobs=1)
model_rlasso.fit(X_train, y)
#Collect the scores into a dataframe and save
coef = pd.DataFrame(model_rlasso.scores_, columns = ['RandomizedLasso_score'])
coef['Feature'] = X_train.columns
coef['Relative score'] = coef['RandomizedLasso_score'] / coef['RandomizedLasso_score'].sum()
coef = coef.sort_values('Relative score', ascending=False)
coef = coef[['Feature', 'RandomizedLasso_score', 'Relative score']]
coef.to_csv("feature_importance_randomizedlasso.csv", index=False)
#Select scores to plot
imp_coef = pd.concat((coef.head(25), coef.tail(5)))
imp_coef.plot(kind = "barh", x='Feature', y='Relative score', legend=False, figsize=(8, 10))
plt.title('5 Least and 25 Most Important RandomizedLasso Features')
plt.xlabel('Relative RandomizedLasso score')
plt.savefig('feature_importance_randomizedlasso.png', bbox_inches='tight', pad_inches=0.5)
plt.show(block=False)
#plt.show()
train_new = model_rlasso.transform(X_train)
test_new = model_rlasso.transform(X_test)
print(" Running LassoCV using features selected by RandomizedLasso:")
model_lasso = LassoCV(eps=0.0000001, n_alphas=200, max_iter=10000, cv=10, precompute=True, random_state=1001)
model_lasso.fit(train_new, y)
print(" Best alpha value: %f" % model_lasso.alpha_ )
lasso = Lasso(alpha=model_lasso.alpha_, max_iter=10000, precompute=True, random_state=1001)
score = round((rmse_cv(lasso).mean()), 6)
print(score)
lasso.fit(train_new, y)
preds = np.expm1(lasso.predict(test_new))
solution = pd.DataFrame({"Id":test.Id, "SalePrice":preds})
now = datetime.now()
sub_file = 'submission_RandomizedLasso_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print(" Writing submission file: %s\n" % sub_file)
solution.to_csv(sub_file, index = False)

#Run LassoLars
print("\n Running LassoLars:")
model_llcv = LassoLarsCV(precompute='auto', max_iter=5000, verbose=False, cv=10, max_n_alphas=5000, n_jobs=1)
model_llcv.fit(X_train, y)
print(" Best alpha value: %f" % model_llcv.alpha_ )
llcv = LassoLars(alpha=model_llcv.alpha_, precompute='auto', max_iter=5000, verbose=False)
score = round((rmse_cv(llcv).mean()), 6)
print(score)
#Save the solution
llcv.fit(X_train, y)
preds = np.expm1(llcv.predict(X_test))
solution = pd.DataFrame({"Id":test.Id, "SalePrice":preds})
now = datetime.now()
sub_file = 'submission_LassoLars_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print(" Writing submission file: %s\n" % sub_file)
solution.to_csv(sub_file, index = False)
#Collect the scores into a dataframe and save
coef = pd.DataFrame(model_llcv.coef_, columns = ['LassoLarsCV_score'])
coef['Feature'] = X_train.columns
coef = coef.sort_values('LassoLarsCV_score', ascending=False)
coef = coef[['Feature', 'LassoLarsCV_score']]
coef.to_csv("feature_importance_larslassocv.csv", index=False)
#Select scores to plot
imp_coef = pd.concat((coef.head(25), coef.tail(5)))
imp_coef.plot(kind = "barh", x='Feature', y='LassoLarsCV_score', legend=False, figsize=(8, 10))
plt.title('5 Least and 25 Most Important LarsLassoCV Features')
plt.xlabel('LarsLassoCV score')
plt.savefig('feature_importance_larslassocv.png', bbox_inches='tight', pad_inches=0.5)
plt.show(block=False)
#plt.show()

#Run ElasticNet
print("\n Running ElasticNet:")
model_elnet = ElasticNetCV(l1_ratio=[0.25, 0.5, 0.75, 0.9], eps=0.0000001, n_alphas=100, max_iter=50000, cv=10, verbose=False, precompute=True, random_state=1001, n_jobs=1).fit(X_train, y)
print(" Best alpha value: %f" % model_elnet.alpha_ )
print(" Best l1_ratio value: %f" % model_elnet.l1_ratio_ )
elnet = ElasticNet(l1_ratio=model_elnet.l1_ratio_, alpha=model_elnet.alpha_, max_iter=50000, precompute=True, random_state=1001).fit(X_train, y)
score = round((rmse_cv(elnet).mean()), 6)
print(score)
#Collect the scores into a dataframe and save
coef = pd.DataFrame(model_elnet.coef_, columns = ['ElasticNet score'])
coef['Feature'] = X_train.columns
coef = coef.sort_values('ElasticNet score', ascending=False)
coef = coef[['Feature', 'ElasticNet score']]
coef.to_csv("feature_importance_elasticnet.csv", index=False)
#Select scores to plot
imp_coef = pd.concat((coef.head(25), coef.tail(5)))
imp_coef.plot(kind = "barh", x='Feature', y='ElasticNet score', legend=False, figsize=(8, 10))
plt.title('5 Least and 25 Most Important ElasticNet Features')
plt.xlabel('Relative ElasticNet score')
plt.savefig('feature_importance_elasticnet.png', bbox_inches='tight', pad_inches=0.5)
plt.show(block=False)
# plt.show()
preds = np.expm1(elnet.predict(X_test))
solution = pd.DataFrame({"Id":test.Id, "SalePrice":preds})
now = datetime.now()
sub_file = 'submission_ElasticNet_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing submission file: %s\n" % sub_file)
solution.to_csv(sub_file, index = False)

#Define XGBoost parameters ; these can certainly be tuned further
params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'rmse'
params['gamma'] = 0.01
params['eta'] = 0.01
params['colsample_bytree'] = 0.1491
params['subsample'] = 0.737
params['max_depth'] = 3
params['silent'] = 1
params['random_state'] = 1001
    
#Prepare the data
d_train = xgb.DMatrix(X_train, label=y)
watchlist = [(d_train, 'train')]
print("\n Running XGBoost:")
clf = xgb.train(params, d_train, 1000, watchlist, verbose_eval=False)
#Extract importance values into dataframe and save
importance = clf.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
df = pd.DataFrame(importance, columns=['mapname', 'fscore'])
df['Relative fscore'] = df['fscore'] / df['fscore'].sum()
f = pd.read_csv('xgb.fmap')
df = pd.merge(df, f, how='left', on='mapname', left_index=True)
df =df[['Feature', 'fscore', 'Relative fscore']]
df.to_csv("feature_importance_xgb.csv", index=False)
#Select scores to plot
df_coef = pd.concat((df.head(25), df.tail(5)))
df_coef.plot(kind='barh', x='Feature', y='Relative fscore', legend=False, figsize=(8, 10))
plt.title('5 Least and 25 Most Important XGBoost Features')
plt.xlabel('Relative feature importance')
plt.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=0.5)
plt.show(block=False)
#plt.show()
