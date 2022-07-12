import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import math

##AUTOMATICLY! LOVELY!!
########################
import pandas as pd
import sklearn as sk

def get_feature_mat(fname):
	#feature engineering in this funciton is applied to both test and train
	df 	= pd.read_csv("../input/"+fname)
	return(df)

train, test = [get_feature_mat(fname) for fname in ['train.csv', 'test.csv']]
print('\nSummary of train dataset:\n')
print(train.describe())
print('\nSummary of test dataset:\n')
print(test.describe())

#######################

train["month"] = pd.DatetimeIndex(train['datetime']).month
train["day"] = pd.DatetimeIndex(train['datetime']).day
train["hour"] = pd.DatetimeIndex(train['datetime']).hour
train["res_reg"] = np.nan
print('\nSummary of train dataset modified:\n')
print(train.describe())

train_month = train.ix[(train.workingday == 1) & (train.month == 2) & (train.day < 16),]
lmod1 = smf.ols(formula = "np.log10(registered+1) ~ C(hour) + temp + humidity + windspeed", data = train_month).fit()
print('\nSummary of Linear Regression model:\n')
print(lmod1.summary())
vallm1 = np.round(10**lmod1.predict(train_month)-1)
vallm1 = np.where(vallm1 >= 0, vallm1, 0)
train.ix[(train.workingday == 1) & (train.month == 2) & (train.day < 16),["res_reg"]] = train.ix[(train.workingday == 1) & (train.month == 2) & (train.day < 16),]["registered"] - vallm1

test_month = train.ix[(train.workingday == 1) & (train.month == 2) & (train.day >= 16),]
vallm1 = np.round(10**lmod1.predict(test_month)-1)
vallm1 = np.where(vallm1 >= 0, vallm1, 0)
train.ix[(train.workingday == 1) & (train.month == 2) & (train.day >= 16),["res_reg"]] = train.ix[(train.workingday == 1) & (train.month == 2) & (train.day >= 16),]["registered"] - vallm1

train_month = train.ix[(train.workingday == 1) & (train.month == 2) & (train.day < 16),]
test_month  = train.ix[(train.workingday == 1) & (train.month == 2) & (train.day >= 16),]

obs_log = np.log(test_month['registered']+1)
pred_log = np.log(vallm1+1)
print('\nRMSLE:\n')
print(math.sqrt(((obs_log-pred_log)**2).mean()))

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
f.suptitle("'The Fish'\nExample of Typical Nonstandardised Residuals per Hour for an OLS model (coloured by temp)\n", fontsize=12, fontweight='bold')
f.subplots_adjust(top=0.85)
#f.tight_layout()
ax1.scatter(train_month.hour, train_month.res_reg, c = train_month.temp, s=40, cmap='hot')
ax1.set_title('Residuals Train (less than 16 days)')
ax2.scatter(test_month.hour ,test_month.res_reg, c = test_month.temp, s=40, cmap='hot')
ax2.set_title('Residuals Test (between 16-19 days)')
plt.savefig("2_scatter.png")
