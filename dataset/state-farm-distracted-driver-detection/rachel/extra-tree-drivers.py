print('Importing libraries...')
import os
import numpy as np
import pandas as pd 
import multiprocessing
from scipy.misc import imread, imresize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score

labels = 'c0,c1,c2,c3,c4,c5,c6,c7,c8,c9'.split(',')

print('Defining data...')
def get_train():
    one_up = os.path.dirname(os.getcwd())
    labels = [i for i in os.listdir(os.path.join(one_up, 'input', 'train')) if 'c' in i]
    labels.sort()
    data = []
    
    for lab in labels:
        paths = os.listdir(os.path.join('..', 'input','train', lab))
        X = [(os.path.join(one_up, 'input', 'train', lab, i), lab) for i in paths]
        data.extend(X)
    import random
    random.shuffle(data) # since labels were sorted
    df = pd.DataFrame({'paths': [i[0] for i in data],
                       'target': [i[1] for i in data]})

    for cl in labels:
        df[cl] = df.target == cl
    df.drop('target', 1, inplace=True)
    return df

train = get_train().sample(2000)

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))

def getimage(x):
    return imresize(imread(x, 'L'), (100, 100)).flatten()

print('Loading training data...')
train['images'] = apply_by_multiprocessing(train.paths, getimage) 
X = np.array([i for i in train.images])


print('Training classifiers...')
classifiers = [ExtraTreesClassifier(n_jobs=-1, n_estimators=100) for i in labels]
targets = [train[i] for i in labels]


print("Fitting model...")
for i, clf, Y in zip(labels, classifiers, targets):
    clf.fit(X, Y)

# it may not be too interesting to look at:
print("clf (our classifier) is", clf)

# let's see how it does on the training data:
print("Training set accuracy:", clf.score(clf,Y))

from sklearn import cross_validation
kfold = cross_validation.KFold(len(X), n_folds=5)  # 5-fold cross-validation
cv_scores = cross_validation.cross_val_score(clf, X, 
               Y, cv=kfold)
print("\n\nFive-fold cross-validation scores:", cv_scores)


print('Fetching test data...')
def get_test():
    one_up = os.path.dirname(os.getcwd())
    paths = os.listdir(os.path.join('..', 'input','test'))
    x = [os.path.join(one_up, 'input', 'test', i) for i in paths]
    x.sort()
    df = pd.DataFrame({'paths': x,})
    return df

print("Cleaning up dataset...")
del(train)
del(X)
del(targets)

print('Loading images...')
test = get_test()
test['images'] = apply_by_multiprocessing(test.paths, getimage)
X = np.array([i for i in test.images])


print("Making predictions...")
results = []
for index, clf in enumerate(classifiers):
    results.append(clf.predict_proba(X)[:,1])

score = log_loss(clf, Y)
print('Score log_loss: ', score)

print('Creating predictions csv...')
c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = results
sub = pd.DataFrame({'img': test.paths.apply(lambda x:x.split('/')[-1]), 
'c0': c0,'c1': c1,'c2': c2, 'c3': c3,'c4': c4,'c5': c5, 'c6': c6,'c7': c7,'c8': c8, 'c9': c9,})
sub[['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']].to_csv('submission.csv', index=False)
print('Finsihed.')