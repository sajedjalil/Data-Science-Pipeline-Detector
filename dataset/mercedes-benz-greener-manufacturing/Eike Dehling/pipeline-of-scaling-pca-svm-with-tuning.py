import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = np.log1p(train['y'].values)
id_test = test['ID']

##
# Combine train/test for one-hot-encoding and other transformations
##
num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]

##
# Make the pipeline, this allows tuning of parameters of each component via GridsearchCV
##
pipe = make_pipeline(RobustScaler(),
                     PCA(n_components=0.975),
                     SVR(kernel='rbf', C=1.0, epsilon=0.05))

##
# Optimize hyper-params, uncomment to tune all of them..
##
gsc = GridSearchCV(
    estimator=pipe,
    param_grid={
        'pca__n_components': (0.95, 0.975, 0.99, None),
        #'svr__C': (0.5, 1.0, 2.0, 3.0),
        #'svr__epsilon': (0.01, 0.05, 0.1),
    },
    scoring='r2',
    cv=5
)

grid_result = gsc.fit(train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, test_stdev, train_mean, train_stdev, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['std_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['std_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f (%f) // Test : %f (%f) with: %r" % (train_mean, train_stdev, test_mean, test_stdev, param))

##
# Now train a model with the optimal params we found
## 
pipe.set_params(**grid_result.best_params_)

pipe.fit(train, y_train)

y_test = np.expm1(pipe.predict(test))

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('mercedes-submission.csv', index=False)