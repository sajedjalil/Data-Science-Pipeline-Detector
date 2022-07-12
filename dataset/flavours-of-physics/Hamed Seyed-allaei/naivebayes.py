# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from sklearn import naive_bayes


# The competition datafiles are in the directory ../input
# List the files we have available to work with
#print("> ls ../input")
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Read competition data files:
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

sub = pd.read_csv("../input/sample_submission.csv")

variables = list(set([i for i in test]) - set([]))

print(variables)


train_xp = train.copy()
train_xp.reset_index(drop=True, inplace=True)

for i in train_xp:
    train_xp[i] = train_xp[i].order().reset_index(drop=True)
train_fp = train_xp.copy()
train_fp = train_fp.rank()
train_fp = 2.0 * train_fp/(train_fp.max()+1) - 1

def transform(data, v = variables, xp = train_xp, fp = train_fp):
    for i in v:
        data[i] = np.interp(data[i], xp[i], fp[i])
    return data

train = transform(train)
test  = transform(test)


clf = naive_bayes.GaussianNB()
clf.fit(train[variables], train['signal'])

sub['prediction'] = clf.predict_proba(test[variables])[:, 1]
sub.to_csv('GaussianNB.csv', index=False, sep=',')