import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

def get_feature_mat(fname):
	#feature engineering in this funciton is applied to both test and train
	df 	= pd.read_csv("../input/"+fname)
	return(df)

train, test = [get_feature_mat(fname) for fname in ['train.csv', 'test.csv']]
print('\nSummary of train dataset:\n')
print(train.describe())
print('\nSummary of test dataset:\n')
print(test.describe())

plt.plot(train['revenue'])
plt.show()

