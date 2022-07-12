import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor

# read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# process columns, apply LabelEncoder to categorical features

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

#print shape      
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))


##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA
n_comp = 10


# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)



# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]
    

y = np.array(train["y"])
X=np.array(train.drop('y', axis=1))

#LightGBM Regressor
model = LGBMRegressor(boosting_type='gbdt', num_leaves=10, max_depth=4, learning_rate=0.005, n_estimators=675, max_bin=25, subsample_for_bin=50000, min_split_gain=0, min_child_weight=5, min_child_samples=10, subsample=0.995, subsample_freq=1, colsample_bytree=1, reg_alpha=0, reg_lambda=0, seed=0, nthread=-1, silent=True)

#Fit to training data
model.fit(X, y)

#Generate Predictions
test['y']=model.predict(test)
output=test[['ID', 'y']]

#Save predictions to 'output.csv'
output.to_csv('output.csv', index=False)