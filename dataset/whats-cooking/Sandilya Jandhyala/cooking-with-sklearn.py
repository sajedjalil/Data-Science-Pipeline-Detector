import numpy as np
import pandas as pd
import json
import csv
import itertools as itools
import re
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectPercentile, chi2

from scipy.sparse import lil_matrix


def build_dfs(train, test, le):
    """ Cleans the data and builds data frames for both
        the training data and the test data.
    """

    # First remove the quantities from each ingredient
    def clean_string(inp):
        matches = re.search('(\(.*\))', inp)
        if matches:
            return matches.group(0).strip()
        else:
            return inp.strip()
    def clean_train_recipe(recipe):
        return {
            'id': recipe['id'],
            'cuisine': recipe['cuisine'],
            'ingredients': [clean_string(x) for x in recipe['ingredients']]
        }
    def clean_test_recipe(recipe):
        return {
            'id': recipe['id'],
            'ingredients': [clean_string(x) for x in recipe['ingredients']]
        }

    clean_train = list(map(clean_train_recipe, train))
    clean_test = list(map(clean_test_recipe, test))

    train_ingredients = set(itools.chain(*[e['ingredients']
                            for e in clean_train]))
    test_ingredients = set(itools.chain(*[e['ingredients']
                            for e in clean_test]))
    all_ingredients = list(train_ingredients | test_ingredients)

    with open('train.csv', 'w') as ouf:
        writer = csv.writer(ouf)
        writer.writerow(['id','cuisine']+all_ingredients)
        for i, recipe in enumerate(clean_train):
            c_num = le.transform([recipe['cuisine']])
            ings = list(map(lambda ingredient: int(ingredient in recipe['ingredients']),
                            all_ingredients))
            writer.writerow(np.concatenate([[recipe['id']], c_num, ings]))

    with open('test.csv', 'w') as ouf:
        writer = csv.writer(ouf)
        writer.writerow(['id']+all_ingredients)
        for i, recipe in enumerate(clean_test):
            ings = list(map(lambda ingredient: int(ingredient in recipe['ingredients']),
                            all_ingredients))
            writer.writerow(np.concatenate([[recipe['id']], ings]))


def csv_dims(fname):
    """ Returns the number of rows and columns in a CSV.
    """
    with open(fname, newline='') as inf:
        reader = csv.reader(inf)
        row_count = 0
        column_count = 0
        for row in reader:
            column_count = len(row)
            row_count += 1
        return (row_count, column_count)


def select_classifier(classifiers, train_df, train_target):
    """ Cross-validates a list of classifiers and returns the one with the
        best score.
    """
    scores = [(alg, cross_val_score(alg, train_df, train_target).mean())
              for alg in classifiers]
    for alg, score in scores:
        print('Cross-validation score of {} is {}'.format(
            type(alg).__name__,
            score
        ), flush=True)
    return max(scores, key=lambda x: x[1])[0]

def main():
    with open('../input/train.json') as inf:
        train = json.load(inf)
    with open('../input/test.json') as inf:
        test = json.load(inf)

    cuisines = list(set(e['cuisine'] for e in train))
    le = LabelEncoder()
    le.fit(cuisines)

    if (not os.path.isfile('train.csv')
        or not os.path.isfile('test.csv')):
        build_dfs(train, test, le)

    rows, columns = csv_dims('train.csv')

    train_df = lil_matrix((rows-1, columns-2))
    train_target = []

    print('Loading training data...', flush=True)
    with open('train.csv', newline='') as inf:
        reader = csv.reader(inf)
        next(reader)
        for i, row in enumerate(reader):
            r = list(row)
            train_df[i,:] = r[2:]
            train_target.append(r[1])
    train_df_raw  = train_df.tocsr()
    print('Done', flush=True)

    print('Performing feature selection...', flush=True)
    feature_selector = SelectPercentile(chi2, 30)
    train_df = feature_selector.fit_transform(train_df_raw, train_target)
    print('Done', flush=True)

    print('Loading test data...')
    raw_test_df = pd.read_csv('test.csv')
    test_df = feature_selector.transform(raw_test_df.iloc[:,1:])
    print('Done', flush=True)

    print('Starting cross-validation...')
    alg = select_classifier((
        BernoulliNB(),
        RandomForestClassifier(
            n_estimators=100,
            min_samples_split=20,
            min_samples_leaf=10
        ),
        LogisticRegression(multi_class='ovr')
    ), train_df, train_target)

    print('Selected {} as the best classifier to use'.format(
        type(alg).__name__))

    print('Fitting the classifier...', flush=True)
    alg.fit(train_df, train_target)
    print('Done', flush=True)

    print('Loading and predicting test results...', flush=True)
    raw_results = alg.predict(test_df)
    results = le.inverse_transform(raw_results.transpose().astype(int))
    print('Done', flush=True)

    submission = pd.DataFrame({
        'id': raw_test_df['id'],
        'cuisine': results
    })

    submission[['id', 'cuisine']].to_csv('first_submission.csv', index=False)


if __name__ == '__main__':
    main()
