"""
Comparing results with SVM and Logistic Regression on 'Don't Overfit II' 
competition dataset. Employing StandardScaler for preprocessing of the
data as well as recursive feature elimination using cross validation and
Grid Search with cross validaion for feature and model selection 
respectively. Script stores results of both classifications in seperate
submission.csv's to be submitted to the competition leaderboard.

Author:
    Florian Eichin

Note: 
    I 'stole' the idea of using a recursive feature elimination from 
    Rishabh Singh's kernel, which can be found under 
    https://www.kaggle.com/rishrk007/don-t-overfit-contest/notebook
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# fetch training and test data
print('Loading data...', end='')
training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
X = training_data.iloc[0:, 2:]
y = training_data.iloc[0:, 1]
X_test = test_data.iloc[0:, 1:]
print('Done.')

# data preprocessing (so far only feature normalization)
print('Data preprocessing...', end='')
standard_scaler = StandardScaler()
standard_scaler.fit(X)
X_transformed = standard_scaler.transform(X)
X_test_transformed = standard_scaler.transform(X_test)
print('Done.')

# find best parameters for SVM
print('Searching for best parameters...', end='')
seed = 420
svm_classifier = SVC(kernel='linear', random_state=seed, probability=True)
parameters = {
    'C' : [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1],
    'shrinking' : [True, False],
    'tol' : [0.0000001, 0.00001, 0.0001, 0.001, 0.01]
}
grid_search_svm = GridSearchCV(svm_classifier,
                               param_grid = parameters,
                               scoring = 'roc_auc',
                               n_jobs = -1,
                               cv=3,
                               iid=False
                              )
grid_search_svm.fit(X_transformed, y)

# find best parameters for Logistic Regression [TODO]
logistic_classifier = LogisticRegression(random_state=seed, solver='liblinear')
parameters = {
    'C' : [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1],
    'tol' : [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.02, 0.03,
             0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    'penalty':['l1','l2']
}
grid_search_lr = GridSearchCV(logistic_classifier,
                              param_grid = parameters,
                              scoring = 'roc_auc',
                              n_jobs = -1,
                              cv=3,
                              iid=False
                             )
grid_search_lr.fit(X_transformed, y)
print('Done.\n')

print("=" * 70)
print("Best score for SVM is %.3f with following parameters:\n %s \n" % 
      (grid_search_svm.best_score_, str(grid_search_svm.best_params_)))
print("Best score for LogisticRegression is %.3f with following parameters:\n %s" % 
      (grid_search_lr.best_score_, str(grid_search_lr.best_params_)))
print("=" * 70, '\n')

# initialize with best parameters
print('Searching for best features and fitting classifiers...', end='')
svm_classifer = SVC(kernel='linear',
                    random_state=seed,
                    probability=True,
                    C=grid_search_svm.best_params_['C'],
                    tol=grid_search_svm.best_params_['tol'],
                    shrinking=grid_search_svm.best_params_['shrinking']
                   )
logistic_classifier = LogisticRegression(random_state=seed,
                                         solver='liblinear',
                                         C=grid_search_lr.best_params_['C'],
                                         tol=grid_search_lr.best_params_['tol'],
                                         penalty=grid_search_lr.best_params_['penalty']
                                        )

# make individual feature selection
feature_selector_svm = RFECV(svm_classifier, min_features_to_select=20, cv=3) 
feature_selector_lr = RFECV(logistic_classifier, min_features_to_select=20, cv=3)

# fit classifiers
feature_selector_svm.fit(X_transformed, y)
feature_selector_lr.fit(X_transformed, y)
print('Done.\n')

print('=' * 70)
print('Number of selected features for SVM: %d' 
      % sum(feature_selector_svm.ranking_ == 1))
print('Number of selected features for LogisticRegression: %d' 
      % sum(feature_selector_lr.ranking_ == 1))
print('=' * 70, '\n')

# make predictions and submit
print('Make predictions and submit...', end="")
prediction_svm = feature_selector_svm.predict(X_test_transformed)
prediction_lr = feature_selector_lr.predict(X_test_transformed)
# probabilities are P(class=1)
probas_svm = 1 - feature_selector_svm.predict_proba(X_test_transformed)
probas_lr = feature_selector_lr.predict_proba(X_test_transformed)

def make_submission(prediction, name):
    # submit prediction
    submission = pd.read_csv('../input/sample_submission.csv')
    submission["target"] = prediction
    submission.to_csv("submission" + name + ".csv", index=False)

for p, n in [(prediction_svm, '_pred_svm'), (prediction_lr, '_pred_lr'), 
             (probas_svm, '_prob_svm'), (probas_lr, '_prob_lr')]:
    make_submission(prediction=p, name=n)
print('Done.')