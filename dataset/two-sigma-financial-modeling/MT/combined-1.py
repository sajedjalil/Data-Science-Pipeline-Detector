import kagglegym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, LinearRegression, LassoLarsCV
from sklearn.ensemble import RandomForestRegressor


# The "environment" is our interface for code competitions
# Get all columns here 
env     = kagglegym.make()
allCols = env.reset().train.columns

class fitModel():
    '''
        This class is going to take train values
        and a particular type of model and take care of
        the prediction step and wil contain a fit
        step. 
        
        Remember to send in a copy of train because
        there is a high chance that it will be modified
        
        the model is a sklearn model like ElasticNetCV
        
        all other parameters are passed onto the model
    '''
    
    def __init__(self, model, train, columns):

        # first save the model ...
        self.model   = model
        self.columns = columns
        
        # Get the X, and y values, 
        y = np.array(train.y)
        
        X = train[columns]
        self.xMeans = X.mean(axis=0) # Remember to save this value
        self.xStd   = X.std(axis=0)  # Remember to save this value

        X = np.array(X.fillna( self.xMeans ))
        X = (X - np.array(self.xMeans))/np.array(self.xStd)
        
        # fit the model
        self.model.fit(X, y)
        
        return
    
    def predict(self, features):
        '''
            This function is going to return the predicted
            value of the function that we are trying to 
            predict, given the observations. 
        '''
        X = features[self.columns]
        X = np.array(X.fillna( self.xMeans ))
        X = (X - np.array(self.xMeans))/np.array(self.xStd)

        return self.model.predict(X)


columns = ['technical_30', 'technical_20',
             'technical_40',
            'fundamental_11', 'technical_19',
            'fundamental_51', 'fundamental_53']
            
print('Starting a new calculation for score')
rewards = []
env = kagglegym.make()
observation = env.reset()

print('fitting a model 1')
model_1 = fitModel(ElasticNetCV(), observation.train.copy(), columns)
print('fitting a model 2')
model_2 = fitModel(LassoLarsCV(), observation.train.copy(), columns)
print('Starting to fit a model')
while True:
    data = observation.features.copy()
    print ('predicting 1')
    prediction_1  = model_1.predict(data)
    print ('predicting 2')
    prediction_2  = model_2.predict(data)   
    target      = observation.target
    target['y'] = prediction_1*0.2 + prediction_2 * 0.8
    
    observation, reward, done, info = env.step(target)
    
    if done: 
        print("Public score: {}".format(info["public_score"]))
        break
        

    



