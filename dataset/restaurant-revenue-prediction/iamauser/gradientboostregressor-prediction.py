# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def main():
    '''
    main
    '''
    
    origTraindf = pd.read_csv('../input/train.csv', encoding='ISO-8859-1')
    trainY = origTraindf['revenue'].apply(np.log).astype(int).as_matrix()
    trainDF = origTraindf[origTraindf.columns[np.r_[5:42]]]
    trainX = trainDF.as_matrix().astype(np.float32)
    scaler = preprocessing.StandardScaler()
    trainX = scaler.fit_transform(trainX)

    origTestdf = pd.read_csv('../input/test.csv', encoding='ISO-8859-1')
    testDF = origTestdf[origTestdf.columns[np.r_[5:42]]]
    testX = testDF.as_matrix().astype(np.float32)
    scaler = preprocessing.StandardScaler()
    testX = scaler.fit_transform(testX)

    print("Starting Gradient Boost Regressor")
    regCLF = ensemble.GradientBoostingRegressor(n_estimators=5000,
                                                max_depth=5,
                                                min_samples_split=5,
                                                learning_rate=0.01,
                                                verbose=0)

    regCLF.fit(trainX, trainY)
    print("Training Score: %f" %regCLF.score(trainX, trainY))
    
    gbr_prediction = regCLF.predict(testX)
    gbr_prediction = np.exp(gbr_prediction)

    # Create a CSV for submission
    sampleSub = pd.DataFrame({
        "Id": origTestdf["Id"],
        "Prediction": gbr_prediction
    })
    sampleSub.to_csv('GBR.csv', header=True, index=False)

if __name__ == '__main__':
    main()




