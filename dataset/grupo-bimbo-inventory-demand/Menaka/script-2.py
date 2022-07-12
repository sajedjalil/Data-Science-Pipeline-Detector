# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold


# dataset_train = pd.read_csv("../input/test.csv")  
chunksize = 10 ** 6

dtype_train = {'Semana':int, 'Agencia_ID':int, 'Canal_ID':int, 'Ruta_SAK':int, 'Cliente_ID':int, 'Producto_ID':int, 'Venta_uni_hoy':int, 'Venta_hoy': float, 'Dev_uni_proxima': int, 'Dev_proxima': float, 'Demanda_uni_equil':int}

# The columns used to predict the target
#eg: predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
predictors = ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Venta_uni_hoy', 'Venta_hoy']

# Initialize our algorithm class
alg = LinearRegression()

actual_demand = []
predictions = [] #array to save predicted values

loop_count = 0
for chunk in pd.read_csv("../input/train.csv", chunksize=6*chunksize, dtype = dtype_train):
    # print(chunk.head(5))
    if(loop_count == 1):
        print("Final Set "+ str(loop_count) +"\n")
        break
    
    print("Set "+ str(loop_count) +"\n")
    
    #print(chunk.describe())
    #print(chunk["Producto_ID"].unique())
    
    # Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
    # We set random_state to ensure we get the same splits every time we run this.
    kf = KFold(chunk.shape[0], n_folds=3, random_state=1)
    
    predictions_temp = []
    
    for train, test in kf:
        # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
        train_predictors = (chunk[predictors].iloc[train,:])
        # The target we're using to train the algorithm.
        train_target = chunk["Demanda_uni_equil"].iloc[train]
        # Training the algorithm using the predictors and target.
        alg.fit(train_predictors, train_target)
        # We can now make predictions on the test fold
        test_predictions = alg.predict(chunk[predictors].iloc[test,:])
        predictions_temp.append(test_predictions)
        #predictions.append(test_predictions)
        
        
    #Evaluating Error - starts
    
    # The predictions are in three separate numpy arrays.  Concatenate them into one.
    # We concatenate them on axis 0, as they only have one axis.
    #predictions = np.concatenate(predictions, axis=0)
    # predictions_temp = np.concatenate(predictions_temp, axis=0)
    # predictions.extend(predictions_temp.tolist())
    predictions_temp = np.concatenate(predictions_temp, axis=0).tolist()
    predictions.extend(predictions_temp)
    predictions_2 = np.array(predictions)
    
    # Map predictions to outcomes (only possible outcomes are 1 and 0)
    #predictions[predictions < 0] = 0
    
    actual_demand.extend(chunk["Demanda_uni_equil"])
    #actual_demand = np.append(chunk["Demanda_uni_equil"])
    #np_actual_demand = np.array(chunk["Demanda_uni_equil"])
    np_actual_demand = np.array(actual_demand)
    #np_actual_demand = actual_demand
    
    #applying ceiling function eg: 8.33 -> 9
    #predictions_rounded_temp= [math.ceil(x) for x in predictions_temp]
    predictions_rounded= [math.ceil(x) for x in predictions_2]
    
    np_predictions = np.array(predictions_rounded)
    
    #print actual and predicted results...
    #results= np.column_stack((np_actual_demand,np_predictions))
    # for row in results:
    #     print("Actual Demand: "+str(row[0])+"   Predicted Demand: " +str(row[1]))
    
    # print("actual demand array shape: " + ''.join(map(str,np_actual_demand.shape)) )
    # print("prediction array shape: " + ''.join(map(str,np_predictions.shape))  )
    print("actual demand array shape: ")
    print(np_actual_demand.shape)
    print("prediction array shape: ")
    print(np_predictions.shape)
        
    accuracy = len(np_predictions[np_actual_demand == np_predictions]) / len(np_actual_demand)
    print("Accuracy (up to "+ str(loop_count) + "): " +str(accuracy)+ "\n")
    
    #Evaluating Error - ends
    
    loop_count +=1
    print("\n\n")
# print(dataset_train.head(5))
# print(dataset_train.describe())



