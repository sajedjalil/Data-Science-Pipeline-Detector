import pandas as pd
import sklearn as sk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

#########################################################################################
#read train data
train = pd.read_csv("../input/train.csv", parse_dates=["datetime"])
train["hour"] = pd.DatetimeIndex(train['datetime']).hour
train["month"] = pd.DatetimeIndex(train['datetime']).month
train["day"] = pd.DatetimeIndex(train['datetime']).day
train["year"] = pd.DatetimeIndex(train['datetime']).year
train["temp"] =  train.temp*9.0/5.0+32
train["atemp"] = train.atemp*9.0/5.0+32

trainCV= train.ix[(train.day <19),] 
testCV = train.ix[(train.day >=19),]

#########################################################################################
#Feature selection
feature_cols = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'hour']

X_train = trainCV[feature_cols]
y_train = trainCV["count"]
X_test = testCV[feature_cols]
y_test = testCV["count"]

from sklearn.ensemble import ExtraTreesClassifier
params = {'n_estimators': 50, 'random_state': np.random.RandomState(1)}

clf = ExtraTreesClassifier(**params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('\nCorrelation coefficients:\n')
print(np.corrcoef(y_pred, y_test))
print('\nFeature importances:\n')
print(clf.feature_importances_ )

###############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)

# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)


# Set color transparency (0: transparent; 1 solid)
a = 0.7
# Create a colormap
customcmap = [(x/10.0,  x/50.0, 0.1) for x in range(len(clf.feature_importances_))]
plt.barh(pos, feature_importance[sorted_idx], align='center',  alpha=a, color=customcmap)
plt.yticks(pos, feature_cols)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
plt.savefig("featureImportance.png")


