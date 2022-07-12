# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets, linear_model
from sklearn.svm import SVR
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# get train & test csv files as a DataFrame
train_df = pd.read_csv("../input/train.csv" )
test_df    = pd.read_csv("../input/test.csv" )

# preview the data
# print(test_df["Sequence"])

print(test_df.tail())
# exit(0)

ii = 0

ANS = []



for seq in test_df["Sequence"]:
    X = []
    Y = []
    SEQ = seq.split(',')
    # for x in seq.split(','):
        # print(x)
    # print("\n")
    ii = ii + 1
    # print(ii)
    if ii > 20:
        break
    
    if len(SEQ) < 2:
        ANS.append(int(0))
        continue
    
    for i in range(0,len(SEQ) -1 ):
        x = float(SEQ[i])
        y = float(SEQ[i+1])
        X.append(x)
        Y.append(y)
        
        # print(str(X[i]) + " "  + str(Y[i]))
        
    # mod = linear_model.LinearRegression()
    
    plt.scatter(X, Y, color='darkorange', label='data')
    plt.hold('on')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    # plt.show()
    plt.savefig(str(ii) + 'books_read.png')
    
    
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    X = np.array(X).astype(np.float)
    X = X.reshape(-1,1)
    # mod.fit(X,)
    
    # ans = mod.predict()
    ans = svr_rbf.fit(X, np.array(Y).astype(np.float)).predict(float(SEQ[len(SEQ) - 1]))
    # print(ans)
    if ans != np.inf and ans != -np.inf and np.isnan(ans) == False:
        ANS.append(int(ans))
    else:
        ANS.append(int(0))
        
    if ii % 10000 == 0:
        print(str(ii) + " " + str(ans[0]))
    

        
'''submission = pd.DataFrame({
        "Id": test_df["Id"],
        "Last": np.array(ANS)
    })
submission.to_csv('titanic.csv', index=False)'''
        
        
        
        
        
        
        
        
        


# Any results you write to the current directory are saved as output.