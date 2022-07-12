import pandas as pd
data=pd.read_csv("../input/train_V2.csv",nrows=70000)
x=data.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27]].values
y=data.iloc[:,[28]].values
#15th entry has to be converted to numeric entities
from sklearn.preprocessing import LabelEncoder
LC=LabelEncoder()
#x[:,15]=LC.fit_transform(x[:,15])
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
KNN=KNeighborsRegressor()
KNN.fit(x,y)
#VM=svm.SVR()
#VM.fit(x,y)
test=pd.read_csv('../input/test_V2.csv')
x_test=test.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27]].values
result_KNN=KNN.predict(x_test)
#result=VM.predict(x_test)
print(result_KNN)
import matplotlib.pyplot as plt 
plt.plot(result_KNN,color='r')
plt.show()



