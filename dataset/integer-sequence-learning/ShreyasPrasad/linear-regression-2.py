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
    # if ii > 1:
        # break
    
    if len(SEQ) < 2:
        ANS.append(int(0))
        continue
    
    elif len(SEQ) < 6:
        '''for i in range(0,len(SEQ) - 2 ):
            x = float(SEQ[i])
            y = float(SEQ[i+1])
            X.append(x)
            Y.append(y)
            
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
        X = np.array(X).astype(np.float)
        print (str(X.shape) +  str(ii))
        
        X = X.reshape(-1,1)
        # mod.fit(X,)
        # print (X)
        # ans = mod.predict()
        
        SVM = svr_rbf.fit(X, np.array(Y).astype(np.float))
        ans = SVM.predict([float(SEQ[len(SEQ) - 1])])
        # print(ans)
        if ans != np.inf and ans != -np.inf and np.isnan(ans) == False:
            ANS.append(int(ans))
        else:'''
        
        ANS.append(int(0))
            
        if ii % 10000 == 0:
            print(str(ii) + " " + str(ans[0]))
    
    else :
        
        for i in range(0,len(SEQ) - 5 ):
            x = [float(SEQ[i]),float(SEQ[i+1]),float(SEQ[i+2]),float(SEQ[i+3])]
            y = float(SEQ[i+4])
            X.append(x)
            Y.append(y)
            
        
    # mod = linear_model.LinearRegression()
    
        mod = linear_model.LinearRegression()
    
        # svr_rbf = SVR(kernel='poly', C=1e3, degree=2)
        
        X = np.array(X).astype(np.float)
        # print (str(X.shape) +  str(ii))
        
        # X = X.reshape(-1,1)
        # mod.fit(X,)
        # print (X)
        # ans = mod.predict()
        
        #  SVM = svr_rbf.fit(X, np.array(Y).astype(np.float))
        mod.fit(X,np.array(Y).astype(np.float))
        ans = mod.predict([float(SEQ[len(SEQ) - 4]),float(SEQ[len(SEQ) - 3]),float(SEQ[len(SEQ) - 2]),float(SEQ[len(SEQ) - 1])])
        # print(ans)
        if ans != np.inf and ans != -np.inf and np.isnan(ans) == False:
            ANS.append(int(ans))
        else:
            ANS.append(int(0))
            
        if ii % 10000 == 0:
            print(str(ii) + " " + str(ans[0]))
    

        
submission = pd.DataFrame({
        "Id": test_df["Id"],
        "Last": np.array(ANS)
    })
submission.to_csv('linear.csv', index=False)
        
        
        
        
        
        
        
        
        