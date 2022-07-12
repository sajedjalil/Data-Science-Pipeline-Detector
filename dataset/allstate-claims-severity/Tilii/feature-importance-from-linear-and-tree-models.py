import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import skew, boxcox
    from datetime import datetime
    import xgboost as xgb
    import operator
    from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV, Lasso, LassoCV, LassoLars, LassoLarsCV, RandomizedLasso
    from sklearn.metrics import mean_squared_error
    from sklearn.cross_validation import cross_val_score
    from sklearn.preprocessing import StandardScaler

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' %
              (tmin, round(tsec, 2)))

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    outfile.write('mapname,Feature\n')
    i = 0
    for feat in features:
        outfile.write('f{0},{1}\n'.format(i, feat))
        i = i + 1
    outfile.close()

def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def rmse_cv(model):
    rmse= -cross_val_score(model, X_train, y, scoring="mean_absolute_error", cv = 10)
    return(rmse)

start_time = timer(None)
print("\n Loading and transforming data:")
train_loader = pd.read_csv("../input/train.csv", dtype={'id': np.int32})
train = train_loader.drop(['id', 'loss'], axis=1)
features = train.columns
create_feature_map(features)
test_loader = pd.read_csv("../input/test.csv", dtype={'id': np.int32})
test = test_loader.drop(['id'], axis=1)
ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)
numeric_feats = train_test.dtypes[train_test.dtypes != "object"].index

# compute skew and do Box-Cox transformation
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[np.absolute(skewed_feats) > 0.25]
skewed_feats = skewed_feats.index

for feats in skewed_feats:
    train_test[feats] = train_test[feats] + 1
    train_test[feats], lam = boxcox(train_test[feats])
cats = [feat for feat in features if 'cat' in feat]

# factorize categorical features
for feat in cats:
    train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
X_train = np.array(train_test.iloc[:ntrain, :])
X_test = np.array(train_test.iloc[ntrain:, :])
y = np.log(np.array(train_loader['loss']))
timer(start_time)

#Run RandomizedLasso
# This section is commented out because it takes too long on Kaggle
# The results are similar to the other two linear models

#start_time = timer(None)
#print("\n Running RandomizedLasso:")
#model_rlasso = RandomizedLasso(alpha='bic', verbose=False, n_resampling=1000, random_state=101, n_jobs=1)
#model_rlasso.fit(X_train, y)
##Collect the scores into a dataframe and save
#coef = pd.DataFrame(model_rlasso.scores_, columns = ['RandomizedLasso_score'])
#coef['Feature'] = train.columns
#coef['Relative score'] = coef['RandomizedLasso_score'] / coef['RandomizedLasso_score'].sum()
#coef = coef.sort_values('Relative score', ascending=False)
#coef = coef[['Feature', 'RandomizedLasso_score', 'Relative score']]
#coef.to_csv("feature_importance_randomizedlasso.csv", index=False)
##Select scores to plot
#imp_coef = pd.concat((coef.head(25), coef.tail(5)))
#imp_coef.plot(kind = "barh", x='Feature', y='Relative score', legend=False, figsize=(8, 10))
#plt.title('5 Least and 25 Most Important RandomizedLasso Features')
#plt.xlabel('Relative RandomizedLasso score')
#plt.savefig('feature_importance_randomizedlasso.png', bbox_inches='tight', pad_inches=0.5)
#plt.show(block=False)
##plt.show()
#train_new = model_rlasso.transform(X_train)
#test_new = model_rlasso.transform(X_test)
#print(" Running LassoCV using features selected by RandomizedLasso:")
#model_lasso = LassoCV(eps=0.0000001, n_alphas=100, max_iter=10000, cv=5, precompute=True, random_state=101)
#model_lasso.fit(train_new, y)
#print(" Best alpha value: %f" % model_lasso.alpha_ )
#lasso = Lasso(alpha=model_lasso.alpha_, max_iter=10000, precompute=True, random_state=101)
#score = round((rmse_cv(lasso).mean()), 6)
#print(" CV Score: %f" % score)
#lasso.fit(train_new, y)
#preds = np.exp(lasso.predict(test_new))
#solution = pd.DataFrame({"id":test_loader.id, "loss":preds})
#now = datetime.now()
#sub_file = 'submission_RandomizedLasso_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
#print(" Writing submission file: %s\n" % sub_file)
#solution.to_csv(sub_file, index = False)
#timer(start_time)

#Run LassoLars
start_time = timer(None)
print("\n Running LassoLars:")
model_llcv = LassoLarsCV(precompute=True, max_iter=10000, verbose=False, cv=10, max_n_alphas=5000, n_jobs=1)
model_llcv.fit(X_train, y)
print(" Best alpha value: %.8f" % model_llcv.alpha_ )
llcv = LassoLars(alpha=model_llcv.alpha_, precompute=True, max_iter=5000, verbose=False)
score = round((rmse_cv(llcv).mean()), 6)
print(" CV Score: %f" % score)
#Save the solution
llcv.fit(X_train, y)
preds = np.exp(llcv.predict(X_test))
solution = pd.DataFrame({"id":test_loader.id, "loss":preds})
now = datetime.now()
sub_file = 'submission_LassoLars_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print(" Writing submission file: %s\n" % sub_file)
solution.to_csv(sub_file, index = False)
#Collect the scores into a dataframe and save
coef = pd.DataFrame(model_llcv.coef_, columns = ['LassoLarsCV_score'])
coef['Feature'] = train.columns
coef = coef.sort_values('LassoLarsCV_score', ascending=False)
coef = coef[['Feature', 'LassoLarsCV_score']]
coef.to_csv("feature_importance_larslassocv.csv", index=False)
#Select scores to plot
imp_coef = pd.concat((coef.head(25), coef.tail(5)))
imp_coef.plot(kind = "barh", x='Feature', y='LassoLarsCV_score', legend=False, figsize=(8, 10))
plt.title('5 Least and 25 Most Important LarsLassoCV Features')
plt.xlabel('Relative LarsLassoCV score')
plt.savefig('feature_importance_larslassocv.png', bbox_inches='tight', pad_inches=0.5)
plt.show(block=False)
#plt.show()
timer(start_time)

#Run ElasticNet
start_time = timer(None)
print("\n Running ElasticNet:")
model_elnet = ElasticNetCV(l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9], eps=0.0000001, n_alphas=100, max_iter=10000, cv=5, verbose=False, precompute=True, random_state=101, n_jobs=1).fit(X_train, y)
print(" Best alpha value: %f" % model_elnet.alpha_ )
print(" Best l1_ratio value: %f" % model_elnet.l1_ratio_ )
elnet = ElasticNet(l1_ratio=model_elnet.l1_ratio_, alpha=model_elnet.alpha_, max_iter=50000, precompute=True, random_state=101).fit(X_train, y)
score = round((rmse_cv(elnet).mean()), 6)
print(" CV Score: %f" % score)
#Collect the scores into a dataframe and save
coef = pd.DataFrame(model_elnet.coef_, columns = ['ElasticNet score'])
coef['Feature'] = train.columns
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
preds = np.exp(elnet.predict(X_test))
solution = pd.DataFrame({"id":test_loader.id, "loss":preds})
now = datetime.now()
sub_file = 'submission_ElasticNet_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing submission file: %s\n" % sub_file)
solution.to_csv(sub_file, index = False)
timer(start_time)

#Define XGBoost parameters ; these can certainly be tuned further
start_time = timer(None)
params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['gamma'] = 0.5290
params['eta'] = 0.1
params['colsample_bytree'] = 0.3085
params['subsample'] = 0.9930
params['max_depth'] = 7
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 101
    
#Prepare the data
d_train = xgb.DMatrix(X_train, label=y)
watchlist = [(d_train, 'train')]
print("\n Running XGBoost:")
clf = xgb.train(params, d_train, 300, watchlist, verbose_eval=False)
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
timer(start_time)
