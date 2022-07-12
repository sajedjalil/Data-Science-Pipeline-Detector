# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.stats import skew

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from subprocess import check_output

# Any results you write to the current directory are saved as output.

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

print (train.head())

print("----------------------------")

print (train.isnull().values.any())

#------------------------------

plt.rcParams['figure.figsize'] = (20, 10)
cost = pd.DataFrame({"log of loss":np.log1p(train["loss"]),"loss":train["loss"]})
cost.hist(bins=50,color='orange')

plt.savefig("Fig0.png")

loss = train["loss"]
train=train.drop(["id"],axis=1)

ids = test['id']
test=test.drop(['id'], axis=1)

cols = train.columns
num_cols = train._get_numeric_data().columns
cat_cols=list(set(cols) - set(num_cols))

#------------------------------
# Plot numerical features

print (train[num_cols].skew())

shift_sk = train[num_cols].apply(lambda x: skew(x.dropna()))
shift_sk = shift_sk[shift_sk > 0.75]
shift_sk = shift_sk.index

train[shift_sk] = np.log1p(train[shift_sk])
train.hist(column=num_cols, bins=10, figsize=(20,20), xlabelsize = 7, color='red')

plt.savefig("Fig1.png")


#------------------------------

num_ft_test = test.dtypes[test.dtypes != "object"].index

sk_ft_test = test[num_ft_test].apply(lambda x: skew(x.dropna())) 
sk_ft_test = sk_ft_test[sk_ft_test > 0.75]
sk_ft_test = sk_ft_test.index

test[sk_ft_test] = np.log1p(test[sk_ft_test])

#------------------------------
# Plot categorical variables

fig_dims=(28,4)

n_cols=4
n_raws=28

for i in range(n_raws):
    for j in range(n_cols):
        plt.subplot2grid(fig_dims,(i,j))
        train[cat_cols[i*n_cols+j]].value_counts().plot(kind='bar',title=cat_cols[i*n_cols+j],color='green')

plt.savefig("Fig2.png")

#-------------------------------

train = pd.get_dummies(train)
train = train.fillna(train.median())

test = pd.get_dummies(test)
test = test.fillna(test.median())

#-------------------------------

# Plot the learning curves

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring,
                                                            n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

X = train.drop(['loss'], axis=1)
y = train.loss

X=X.astype(int)
y=y.astype(int)

y=y.reshape(len(y),1)

print (y.shape)
print (X.shape)

same_cols = [col for col in X.columns if col in test.columns]


#--------------------------

# Apply random forest clasifier to plot the learning curves

#X_train, X_test, y_train, y_test = train_test_split(X[same_cols], y, random_state=42)

#rfc = RandomForestClassifier(random_state=42, criterion='entropy', min_samples_split=5, oob_score=True)
#parameters = {'n_estimators':[500], 'min_samples_leaf':[12]}

#scoring = make_scorer(accuracy_score, greater_is_better=True)

#cl_rand_fr = GridSearchCV(rfc, param_grid=parameters, scoring=scoring)

#print(X_train.shape)
#print(y_train.shape)

#cl_rand_fr.fit(X_train, y_train)
#cl_rand_fr = cl_rand_fr.best_estimator_

# Show prediction accuracy score

#print (accuracy_score(y_test, cl_rand_fr.predict(X_test)))
#print (cl_rand_fr)
#plot_learning_curve(cl_rand_fr, 'Random Forest', X[same_cols], y, cv=4);
#plt.savefig("Fig3.png")

#--------------------------

X=train
X=train.drop(['loss'], axis=1)
y=train.loss

X=X.astype(int)
y=y.astype(int)

y=y.reshape(len(y),1)

print (y.shape)
print (X.shape)

Xt=test

X_train, X_test, y_train, y_test = train_test_split(X[same_cols], y)

reg_rand_fr = RandomForestRegressor(n_estimators=500, n_jobs=-1,random_state=42)

print (y_train.shape)
print (X_train.shape)

#reg_rand_fr.fit(X_train, y_train)
#y_pred = reg_rand_fr.predict(X_test)
#y_pred_test = reg_rand_fr.predict(Xt)

#plt.figure(figsize=(10, 5))
#plt.scatter(y_test, y_pred,s=30,color='black')
#plt.title('Predicted vs. Actual')
#plt.xlabel('Actual Sale Price')
#plt.ylabel('Predicted Sale Price')

#plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
#plt.tight_layout()

#plt.savefig("Fig3.png")

#--------------------------

# Submit result

#Result = [x - 1 for x in np.exp(y_pred_test)]

#submission = pd.DataFrame({
#        "Id": ids,
#        "loss": Result
#    })
#submission.to_csv('Test-Price.csv', index=False)

