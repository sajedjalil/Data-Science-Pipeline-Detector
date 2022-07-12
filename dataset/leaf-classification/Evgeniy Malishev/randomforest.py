import pandas as pd

from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
if __name__ == '__main__':
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    
    def encode(train, test):
        le = LabelEncoder().fit(train.species)
        labels = le.transform(train.species)
        classes = list(le.classes_)
        test_ids = test.id

        train = train.drop(['species', 'id'], axis=1)
        test = test.drop(['id'], axis=1)

        return train, labels, test, test_ids, classes

    train, labels, test, test_ids, classes = encode(train, test)
    train.head(1)

    sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

    for train_index, test_index in sss:
        X_train, X_test = train.values[train_index], train.values[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    alg_frst_model = RandomForestClassifier(random_state=1)
    alg_frst_params = [{
        "n_estimators": [50, 100, 150, 200, 300, 350, 400, 450, 500, 1000],
        "min_samples_split": [2, 4, 6, 8, 10, 12, 20, 40, 100],
        "min_samples_leaf": [1, 2, 4, 5,6,7,8,9,10,20,40,100]
    }]
    favorite_clf = GridSearchCV(alg_frst_model, alg_frst_params, scoring = 'accuracy',  refit=True, verbose=1, n_jobs=-1)


    favorite_clf.fit(X_train, y_train)
    test_predictions = favorite_clf.predict_proba(test)

    submission = pd.DataFrame(test_predictions, columns=classes)
    submission.insert(0, 'id', test_ids)
    submission.reset_index()

    submission.to_csv('submission_regressionHighScope.csv', index = False)
    submission.tail()