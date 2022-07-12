import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:

# Any files you write to the current directory get shown as outputs

features = list(train.columns)
features.remove('Cover_Type')
features.remove('Id')


#alg = RandomForestClassifier(random_state=1, n_estimators=200)
alg = GradientBoostingRegressor(random_state=1, n_estimators=500, min_samples_split=8, min_samples_leaf=4, learning_rate=0.1)


alg.fit(train[features], train['Cover_Type'])
predictions = alg.predict(test[features])
predictions = predictions.astype(int)
sub = pd.DataFrame({'Id': test['Id'], 'Cover_Type': predictions})
sub.to_csv('kaggle.csv', index=False)