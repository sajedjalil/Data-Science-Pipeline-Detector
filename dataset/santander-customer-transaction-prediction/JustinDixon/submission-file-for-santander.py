# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sci 
import sklearn as sk
from sklearn import model_selection
import os

# helper functions
def split_data(split_proportion):
    X_train, X_validation, y_train, y_validation = model_selection.train_test_split(training_data.drop(["ID_code","target"], axis=1).values, training_data.target.values, test_size=split_proportion, random_state=42)
    return X_train, X_validation, y_train, y_validation
    
def score_predictions(model, X_train, X_validation, y_train, y_validation):
    probability_predictions = model.predict_proba(X_train)
    y_scores = []
    for i in probability_predictions:
        y_scores.append(i[1])
    train_score = sk.metrics.roc_auc_score(y_train, y_scores, 'weighted')
    probability_predictions = model.predict_proba(X_validation)
    y_scores = []
    for i in probability_predictions:
        y_scores.append(i[1])
    validation_score = sk.metrics.roc_auc_score(y_validation, y_scores, 'weighted')
    print('train score: ', train_score, ' validation score: ', validation_score)
    
def save_output_predictions(model):
    predictions = model.predict_proba(testing_data.drop(["ID_code"], axis=1).values)
    y_predictions = []
    for i in predictions:
        y_predictions.append(i[1])
    my_submission = pd.DataFrame({'ID_code': testing_data.ID_code, 'target': y_predictions})
    my_submission.to_csv('submission.csv', index=False)

# Models
def rf():
    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier(
                n_estimators = 250,
                bootstrap = True,
                criterion = 'gini',
                min_samples_leaf = 20,
                max_depth = 20,
                class_weight = 'balanced'
            ).fit(X_train, y_train)
    print(score_predictions(random_forest, X_train, X_validation, y_train, y_validation))
    return random_forest

def sgdlinear():
    from sklearn.linear_model import SGDClassifier
    linear_classifier_sgd = SGDClassifier(
        loss = 'log',
        max_iter=1000, 
        tol=1e-3,
        alpha = 0.0001,
        class_weight = 'balanced',
        epsilon = 0.1,
        learning_rate = 'optimal'
    ).fit(X_train, y_train)
    print(score_predictions(linear_classifier_sgd, X_train, X_validation, y_train, y_validation))
    return sgdlinear

def logisticregression():
    from sklearn.linear_model import LogisticRegression
    linear_classifier = LogisticRegression(
        class_weight = 'balanced',
        penalty = 'l2'
        ).fit(X_train, y_train)
    print(score_predictions(linear_classifier, X_train, X_validation, y_train, y_validation))
    return linear_classifier

# Run script
sample_submission = pd.read_csv("../input/sample_submission.csv")
training_data = pd.read_csv("../input/train.csv")
testing_data = pd.read_csv("../input/test.csv")
split_proportion=0.8
X_train, X_validation, y_train, y_validation = split_data(split_proportion)
linear_classifier = logisticregression()
save_output_predictions(linear_classifier)
