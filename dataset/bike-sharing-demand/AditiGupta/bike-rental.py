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

df = pd.DataFrame(train)
print(df.head(7))
print(df.ix[:6,1:8])

#train.plot(kind='box', subplots=True, layout=(11,11), sharex=False, sharey=False)
#plt.show()
