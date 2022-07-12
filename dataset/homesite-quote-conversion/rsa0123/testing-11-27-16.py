import pandas as pd
import time as timer
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing 

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")



def print_hist(group_indicator, groups, train, y_field):
	train_y0 = train.loc[train[y_field] == 0]
	train_y1 = train.loc[train[y_field] == 1]
	train_y0.hist(column = groups[group_indicator],color = 'b', alpha = 0.7)
	plt.suptitle('y = 0')
	train_y1.hist(column = groups[group_indicator],color = 'r', alpha = 0.7)
	plt.suptitle('y = 1')
	plt.show()

def feaure_selection(group_indicator, groups, x_values):
	corr_1 = x_values[groups[1]].corr('pearson')
	print(corr_1[corr_1 > 0.85])
	
def classify(X_train, y_train, X_test, y_test):
	model = xgboost.XGBClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

	
#init
t0 = timer.time()


le = preprocessing.LabelEncoder()
y_field = 'QuoteConversion_Flag'
irrelevant_fields =['QuoteNumber', y_field]
sorting_fields = ['Field','Coverage','Sales','Personal','Property','Geographic']
groups = []
print(train.shape)
# train = train.head(n=100)
y = train[y_field].values
for field in sorting_fields:
	groups.append([col for col in train.select_dtypes(exclude = ['object']).columns if col.startswith(field)])
# print_hist(1, groups, train, y_field)
train.drop(irrelevant_fields,axis = 1, inplace = True)
x_values = train.select_dtypes(exclude = ['object'])
x_mean = x_values.mean(axis = 0)
x_var = x_values.var(axis = 0)
x_norm = (x_values - x_mean)/(x_values.std(axis=0))
x_norm = x_norm.fillna(-1)
X_train, X_val, y_train, y_val = train_test_split( x_norm, y, test_size=0.33, random_state=42)

l_data = le.fit_transform(train['SalesField7'])

classify(X_train, y_train, X_val, y_val)
print(train['PersonalField11'].max())
print(train['PersonalField11'].min())
t1 = timer.time()
print(t1 -t0)

