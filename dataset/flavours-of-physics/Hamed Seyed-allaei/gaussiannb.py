# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
from sklearn import ensemble

# The competition datafiles are in the directory ../input
# List the files we have available to work with
#print("> ls ../input")
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Read competition data files:
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/sample_submission.csv")
#variables = list(set([i for i in test]) - set(['SPDhits']))
variables = list(set([i for i in test]) - set([]))

print(variables)

clf = ensemble.RandomForestClassifier(bootstrap=False, class_weight='auto', criterion='gini', max_features=3, n_estimators=228, random_state=0, max_depth=2)
clf.fit(train[variables], train['signal'])

sub['prediction'] = clf.predict_proba(test[variables])[:, 1]
sub.to_csv('RandomForest.csv', index=False, sep=',')

# Write summaries of the train and test sets to the log
#print('\nSummary of train dataset:\n')
#print(train.describe())
#print('\nSummary of test dataset:\n')
#print(test.describe())
