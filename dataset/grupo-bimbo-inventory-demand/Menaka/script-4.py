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
predictors = ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']

# Initialize our algorithm class
alg = LinearRegression()





# Traing the model  --- starts

loop_count = 0
for chunk in pd.read_csv("../input/train.csv", chunksize=6*chunksize, dtype = dtype_train):
    # print(chunk.head(5))
    if(loop_count == 1):
        print("Final Set "+ str(loop_count) +"\n")
        break
    
    print("Set "+ str(loop_count) +"\n")
    
    #print(chunk.describe())
    #print(chunk["Producto_ID"].unique())
    

    # Training the algorithm using the predictors and target.
    alg.fit(chunk[predictors], chunk["Demanda_uni_equil"])


    loop_count +=1
    print("\n\n")
    
    
# Traing the model  --- ends

    
# print(dataset_train.head(5))
# print(dataset_train.describe())

dtype_test = {'Semana':int, 'Agencia_ID':int, 'Canal_ID':int, 'Ruta_SAK':int, 'Cliente_ID':int, 'Producto_ID':int}

predictions = [] #array to save predicted values
row_id = []

loop_count = 0
for chunk in pd.read_csv("../input/test.csv", chunksize=2*chunksize, dtype = dtype_train):
    if(loop_count == 1):
        print("[Test]Final Set "+ str(loop_count) +"\n")
        break
    print("[Test]Set "+ str(loop_count) +"\n")
    
    # Make predictions using the test set.
    np_predictions_temp = alg.predict(chunk[predictors])
    
    #applying ceiling function eg: 8.33 -> 9
    predictions_rounded= [math.ceil(x) for x in np_predictions_temp]
    
    predictions.extend(predictions_rounded)
    

    row_id.extend(chunk["id"]) 
    
    loop_count +=1
    print("\n\n")
    
# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "id": row_id,
        "Demanda_uni_equil": predictions
    })    
    

