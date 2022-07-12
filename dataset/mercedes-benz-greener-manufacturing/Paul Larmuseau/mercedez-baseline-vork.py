import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder


# read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

test['y'] = 102  # to make append possible
print
y_train = train["y"]

#find unique ?
kolom=train.columns 
kolom=[k for k in kolom if k not in ['ID','y']]
train_u = train.sort_values(by='y').duplicated(subset=kolom)
print(train_u)

totaal= train.append(test)
test = test.drop(['y'], axis=1)



# process columns, apply LabelEncoder to categorical features
for c in totaal.columns:
    if totaal[c].dtype == 'object':
        tempt = totaal[['y',c]]
        temp=tempt.groupby(c).mean().sort('y')
        templ=temp.index
        print(templ)
        aant=len(templ)
        train[c].replace(to_replace=templ, value=[x/aant for x in range(0,aant)], inplace=True, method='pad', axis=1)
        test[c].replace(to_replace=templ, value=[x/aant for x in range(0,aant)], inplace=True, method='pad', axis=1)        


# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))


##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA,TruncatedSVD,NMF, LatentDirichletAllocation,FactorAnalysis,MiniBatchDictionaryLearning
from sklearn.cluster import MiniBatchKMeans
n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

#MDB
mdb=MiniBatchDictionaryLearning(n_components=n_comp, alpha=0.1,n_iter=50, batch_size=3,random_state=42)
mdb_results_train = mdb.fit_transform(train.drop(["y"], axis=1)) #,y=y_train) #R2 +y 0.469
mdb_results_test = mdb.transform(test)

#mbk
mbk=MiniBatchKMeans(n_clusters=n_comp, tol=1e-3, batch_size=20,max_iter=50, random_state=42)
mbk_results_train = mbk.fit_transform(train.drop(["y"], axis=1)) #,y=y_train) #R2 +y 0.437
mbk_results_test = mbk.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]

    train['mbk_' + str(i)] = mbk_results_train[:,i-1]
    test['mbk_' + str(i)] = mbk_results_test[:, i-1]
    
    train['mdb_' + str(i)] = mdb_results_train[:,i-1]
    test['mdb_' + str(i)] = mdb_results_test[:, i-1]
    
#    train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
#    test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]
    
y_train = train["y"]
y_mean = np.mean(y_train)



### Regressor
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}


# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
#cv_result = xgb.cv(xgb_params, 
#                   dtrain, 
#                   num_boost_round=1000, # increase to have better results (~700)
#                   early_stopping_rounds=50,
#                   verbose_eval=10, 
#                   show_stdv=False
#                  )

#num_boost_rounds = len(cv_result)
#print('num_boost_rounds=' + str(num_boost_rounds))

num_boost_rounds = 1500
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# check f2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score
print(r2_score(model.predict(dtrain), dtrain.get_label()))

# make predictions and save results
y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('submission_baseLine.csv', index=False)
