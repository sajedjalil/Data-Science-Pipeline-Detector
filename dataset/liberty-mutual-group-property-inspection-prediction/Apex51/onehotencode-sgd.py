# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.linear_model import SGDRegressor
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.svm import SVR

# normalized gini by 0x0FFF
def gini(solution, submission):
    df = zip(solution, submission)
    df = sorted(df, key=lambda x: (x[1],x[0]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

# The competition datafiles are in the directory ../input
# Read competition data files
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

test_ind = test.index

train = train.values
test = test.values

# label encode the categorical variables
for i in range(train.shape[1]):
	if type(train[1,i]) is str:
		lbl = preprocessing.LabelEncoder()
		lbl.fit(list(train[:,i]) + list(test[:,i]))
		train[:,i] = lbl.transform(train[:,i])
		test[:,i] = lbl.transform(test[:,i])

train = train.astype(np.int64)
test = test.astype(np.int64)
y = labels.values.astype(float)

# OneHotEncoder
ohc = preprocessing.OneHotEncoder()
train = ohc.fit_transform(train)
test = ohc.transform(test)

sgd = SGDRegressor()
param_grid = {'alpha':[0.00005,0.0001,0.0005],
			'epsilon':[0.05,0.1,0.15]}

gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)

model = grid_search.GridSearchCV(estimator = sgd, param_grid=param_grid, scoring=gini_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
model.fit(train,y)

print("Best score: %0.3f" % model.best_score_)
a_best_s = model.best_score_
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
	print("\t%s: %r" % (param_name, best_parameters[param_name]))

best_model = model.best_estimator_
best_model.fit(train,y)

ypreds = best_model.predict(test)
preds = pd.DataFrame({"Id": test_ind, "Hazard": ypreds})
preds = preds.set_index('Id')
preds.to_csv('onehotencoder_sgd.csv')
