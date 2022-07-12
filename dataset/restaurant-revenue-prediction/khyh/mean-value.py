import pandas as pd
import sklearn as sk
import numpy as np

def get_feature_mat(fname):
	#feature engineering in this funciton is applied to both test and train
	df 	= pd.read_csv("../input/"+fname)
	return(df)

train, test = [get_feature_mat(fname) for fname in ['train.csv', 'test.csv']]

print(train.columns)
print(test.columns)

v = np.array(train['revenue'])
mean = np.mean(v)
std = np.mean(v)

df_submit = pd.read_csv("../input/sampleSubmission.csv")
df_submit.set_index('Id', inplace=True)

print(df_submit.columns)

df_submit['Prediction'] = mean
df_submit.to_csv('submit.csv')

#print('\nSummary of train dataset:\n')
#print(train.describe())
#print('\nSummary of test dataset:\n')
#print(test.describe())