import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = train['y'].values
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]

gsc = GridSearchCV(
    estimator=RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
                                    min_samples_leaf=25, max_depth=3),
    param_grid={
        #'n_estimators': range(75,251,25),  # Best => 150-250
        #'max_features': range(150,500,50),  # Best => 200-400
        #'min_samples_leaf': range(15,30,5),  # Best => 20-25
        #'min_samples_split': range(15,30,5),  # Best => 15-20
        #'max_depth': range(2,6),  # Best => 3-4
    },
    scoring='r2',
    cv=3
)

#
# Commented because tuning is completee ;-)
#
# grid_result = gsc.fit(train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for test_mean, train_mean, param in zip(
#        grid_result.cv_results_['mean_test_score'],
#        grid_result.cv_results_['mean_train_score'],
#        grid_result.cv_results_['params']):
#    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))

model = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
                              min_samples_leaf=25, max_depth=3)

model.fit(train, y_train)

df_sub = pd.DataFrame({'ID': id_test, 'y': model.predict(test)})
df_sub.to_csv('submission.csv', index=False)