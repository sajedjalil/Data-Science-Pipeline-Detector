import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


numRows = 6999251 #(length of test file) 

# vectorized rmsle calc from https://www.kaggle.com/jpopham91/caterpillar-tube-pricing/rmlse-vectorized/code
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))



#Get Curves For RMSLE as a Function of Over- and Under-Prediction Values
def calcOverPredUnderPredCurves(errorRange):   

    #Initialize Lists to hold RMSLE for range of OverPrediction and UnderPrediction Values
    overPredCurve = []
    underPredCurve= []
    
    #Set Arbitrary 'actual' data vector 
    actual = np.repeat(100, numRows)
    
    for e in range(errorRange):
        #Get (x, ) of (x,y) in (error, RMSLE) curves
        overPrediction = actual + e
        underPrediction = actual - e
        
        #Get (, y) of (x,y) in (error, RMSLE) curves
        overPredRMSLE = rmsle(overPrediction, actual)
        underPredRMSLE= rmsle(underPrediction, actual)
        
        overPredCurve.append(overPredRMSLE)
        underPredCurve.append(underPredRMSLE)
        
    return overPredCurve, underPredCurve
   
over, under = calcOverPredUnderPredCurves(100) 


#Plot Curves
sns.set_style("darkgrid")
plt.plot(over, label="over-prediction RMSLE")
plt.plot(under, label="under-prediction RMSLE")
plt.show()
