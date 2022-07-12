"""
Example Python 3 template for Two Sigma competition.

Useful as a starting point. To submit a new model, 
you need to:

Step 0) [Optional] Implement 'preprocess' to prepare the training set
Step 1) Implement 'train' to create your model
Step 2) Implement '__predict' to perform predictins using the model

* Tips:

- Step 0) can be used for initial cleaning or filtering steps 
(e.g. dealing with missing values, centering the data ...)

- In Step 1), the trained model should be stored in self.model . 
That way it could be retrieved easily when doing the actual predictions

- Feel free to add additional methods to the KaggleSubmission class
if additional functionality becomes necessary (CV, ensembling ...)
"""
__author__ = 'CarrDelling -- https://www.kaggle.com/carrdelling'

import kagglegym
import numpy as np
import pandas as pd
import logging as lgg

# configure logging format
lgg.basicConfig(level=lgg.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s')


class KaggleSubmission:
    """ Class representing a Kaggle submission 
    
        - Manage KaggleGym enviroment. 
        - Load & preprocess training data
        - Train models
        - Predict output for test data
        - Store results & check if the tests has ended
    """ 
    
    def __init__(self):
        """ Initialize enviroment and load the first batch of observations """
        
        self.enviroment = kagglegym.make()
        observation = self.enviroment.reset()
        
        self.training = observation.train
        self.test_X = observation.features
        self.test_Y = observation.target
        
        self.rewards = []
        self.completed = False
        
        self.model = None
        
    def preprocess(self):
        """ Preprocess the training dataset """
        
    def train(self):
        """ Train models based in the current training set """
    
    def __predict(self):
        """ Create predictions for the current test batch 
        
            Test features are in self.test_X
            Predictions are stored by updating the self.test_Y DataFrame
        """
        
    def step(self):
        """ Submit predictions and load the next test batch of observations """
        
        self.__predict()
        
        observation, reward, done, info = self.enviroment.step(self.test_Y)

        self.rewards.append(reward)
        self.completed = done

        if not done:    
            self.test_X = observation.features
            self.test_Y = observation.target
        else:
            self.public_score = info['public_score']
            
    def finish(self):
        """ Tests if the submission is complete """
        
        return self.completed
        
        
    def performance(self):
        """ Show performance achieved """
        
        if not self.rewards:
            message = "No score yet"
        else:
            mean_score = np.mean(self.rewards)
            median_score = np.median(self.rewards)
            min_score = np.min(self.rewards)
            max_score = np.max(self.rewards)
            std_score = np.std(self.rewards)
        
            message = "Score: {:5f} +- {:5f} Median: {:5f} Min/Max: {:5f} / {:5f}".format(
            mean_score, std_score, median_score, min_score, max_score)

        lgg.info(message)

if __name__ == "__main__":
    
    submission = KaggleSubmission()
    
    # preprocess training data
    submission.preprocess()
    
    # train initial models
    submission.train()
    
    # loop until the submission is complete
    while not submission.finish():
        
        submission.step()
        
        # remove the following line for disabling step-wise statistics on rewards
        submission.performance()
        
    # report results
    submission.performance()
    lgg.info('End run - Public score: {}'.format(submission.public_score))
