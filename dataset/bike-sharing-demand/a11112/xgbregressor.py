from numpy import argsort, log1p, expm1
from pandas import read_csv, DataFrame
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor

# Import Data

train = read_csv("../input/train.csv", parse_dates=[0])
test = read_csv("../input/test.csv", parse_dates=[0])

# Data Preprocessing

train_data = train.drop(labels=["count"], axis=1)
train_labels = train["count"]
test_data = test

for df in [train_data, test_data]:
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

train_data = train_data.drop(labels=["datetime", "casual", "registered"], axis=1)
test_data = test.drop(labels=["datetime"], axis=1)

# Feature Engineering

x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3, random_state=0)

rfr = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
rfr.fit(x_train, y_train)
y_pred = rfr.predict(x_test)

print("Simple RandomForestRegressor mse: ", mean_squared_error(y_test, y_pred))
indices = argsort(rfr.feature_importances_)[::-1]
print("Feature importance:")
for f in range(x_train.shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f], x_train.columns[indices[f]], rfr.feature_importances_[indices[f]]))

# Optimizing

# rfc = RandomForestRegressor(random_state=0)
# parameter_grid = {
#     "n_estimators": [10, 50, 100, 1000],
#     "criterion": ["mae", "mse"],
#     "max_features": ["auto", "sqrt", "log2", None],
#     "max_depth": [None, 5, 20, 100],
#     "min_samples_split": [2, 5, 7],
#     "max_leaf_nodes": [40, 60, 80]
# }
# scorer = make_scorer(mean_squared_error, greater_is_better=False)
# grid_search = GridSearchCV(rfc, param_grid=parameter_grid, scoring=scorer, cv=5)
# grid_search.fit(train_data, train_labels)
#
# print("Best score: %f" % grid_search.best_score_)
# print("Best parameters: %s" % grid_search.best_params_)

# Prediction

# regressor = RandomForestRegressor(random_state=0, n_jobs=-1, n_estimators=2700, min_samples_split=4, max_features=0.65,
#                             max_depth=17, oob_score=True)
#
# estimator = DecisionTreeRegressor(splitter="random", min_samples_split=4, max_features=0.65, max_depth=17)
# regressor = AdaBoostRegressor(base_estimator=estimator, learning_rate=1e-2, n_estimators=2700, random_state=0)

regressor = XGBRegressor(n_estimators=400, n_jobs=-1, random_state=0)
regressor.fit(train_data, log1p(train_labels))
pred = expm1(regressor.predict(test_data))

submission = DataFrame({"datetime": test["datetime"], "count": pred})
submission.to_csv("submission.csv", index=False)
