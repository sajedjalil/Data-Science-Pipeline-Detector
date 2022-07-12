__author__ = 'Ahmed Hani Ibrahim'

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
import csv
import json
import scipy.sparse
import numpy as np


def get_train_data():
    with open('../input/train.json') as r:
        data = json.load(r)
        r.close()

    return data


def get_test_data():
    with open('../input/test.json') as r:
        data = json.load(r)
        r.close()

    ids = [item['id'] for item in data]

    return data, ids


def get_training_data_matrix(data):
    labels = [item['cuisine'] for item in data]
    unique_labels = set(labels)
    ingredients = [item['ingredients'] for item in data]
    unique_ingredients = set(inner_item for outer_item in ingredients for inner_item in outer_item)

    training_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)))

    for i, item in enumerate(ingredients):
        for j, ing in enumerate(unique_ingredients):
            if ing in item:
                training_data_matrix[i, j] = 1

    return labels, training_data_matrix, unique_ingredients


def get_test_data_matrix(data, unique_ingredients):
    ingredients = [item['ingredients'] for item in data]
    test_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)))

    for i, item in enumerate(ingredients):
        for j, ing in enumerate(unique_ingredients):
            if ing in item:
                test_data_matrix[i, j] = 1

    return test_data_matrix



lr = LogisticRegression()
labels, training_data_matrix, unique_ingredients = get_training_data_matrix(get_train_data())
lr = lr.fit(training_data_matrix, labels)

print("Training Done")

test_data, ids = get_test_data()
test_data_matrix = get_test_data_matrix(test_data, unique_ingredients)

res = lr.predict(test_data_matrix)
print("Predicting Done")
submission = dict(zip(ids, res))

wr = csv.writer(open('Logistics_Regression_Result.csv', 'wt'))
wr.writerow(['id', 'cuisine'])

for first, second in submission.items():
    wr.writerow([first, second])

print("done")

#print(cross_val_score(lr, training_data_matrix, labels, cv=5).mean())



