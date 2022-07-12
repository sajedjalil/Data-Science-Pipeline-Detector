"""
This python script is part of our support work for DiveIntoCode students 
https://diveintocode.jp/ai_curriculum

The goal is to show how to target encode categorical features 
and update a Gradient Descent Classifier one chunk at a time

This is sort of an online learning.

Test file is then read and predictions are made using the full averages and fitted classifier

I hope you'll find this useful

"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import SGDClassifier
import time
import gc


def get_train_file_name():
    return '../input/train.csv'


def get_test_file_name():
    return '../input/test.csv'

    
def get_feature_types():
    return {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
    }


def get_working_features():
    return ['ip', 'device', 'channel', 'os', 'app']
    

def get_target_feature():
    return 'is_attributed'


class AverageManager(object):
    """ Class that will manage target averages for selected feature and target encode these features """
    def __init__(self, features, target):
        """ Init an average manager for the given features and the specified target """
        # averages contains the average data
        self.averages = {
            f: None for f in features
        }
        # Prior contains the estimated prior of the target 
        self.prior = {'cum_sum': 0.0, 'nb_samples': 0.0}
        # Conatins the name of the target column in the DataFrames
        self.target = target
        
    def update_averages(self, df):
        """Update averages information using samples available in df"""
        # update prior
        self.prior['cum_sum'] += df[self.target].sum()
        self.prior['nb_samples'] += df.shape[0]
        
        for f_ in self.averages.keys():
            # Create the groupby
            the_group = df[[f_, self.target]].groupby(f_).agg(['sum', 'count'])
            the_group.columns = the_group.columns.droplevel(0)
    
            # Update the average
            if self.averages[f_] is None:
                self.averages[f_] = the_group
            else:
                # pandas .add method makes sure apps that are not in both the_group and current averages
                # take value of 0 before the addition takes place
                self.averages[f_] = the_group.add(self.averages[f_], fill_value=0.0)
            
            del the_group
            gc.collect()
            
    def apply_averages(self, df):
        """Apply calculated averages on df to target encode the features"""
        encoded = pd.DataFrame()
        for f_ in self.averages.keys():
            if self.averages[f_] is None:
                raise ValueError('Averages have not been fitted yet')
            self.averages[f_]['average'] = self.averages[f_]['sum'] / self.averages[f_]['count']
            encoded[f_] = df[f_].map(self.averages[f_]['average']).astype(np.float32)
            prior = self.prior['cum_sum'] / self.prior['nb_samples']
            encoded[f_].fillna(prior, inplace=True)
        
        return encoded
        

def train_classifier_on_averages():
    start_time=time.time()
    # Create average manager
    avg_man = AverageManager(features=get_working_features(), target=get_target_feature())
    
    # Init Classifier, very simple not tuning yet
    clf = SGDClassifier(loss='log', tol=1e-2, random_state=1)
    
    # Read train file 
    chunksize=20000000
    for i_chunk, df in enumerate(pd.read_csv(get_train_file_name(), 
                                             chunksize=chunksize, 
                                             dtype=get_feature_types(), 
                                             usecols=get_working_features() + [get_target_feature()])):
        # Udpate averages with the average manager
        avg_man.update_averages(df)
        
        # Apply averages usin the average manager
        target_encoding = avg_man.apply_averages(df)
        
        # Update the SGDClassifier using current target encoding and calling partial_fit
        clf.partial_fit(X=target_encoding, y=df[get_target_feature()], classes=[0, 1])
        
        # Get current predictions
        preds = clf.predict_proba(target_encoding)[:, 1]
        
        # Display the log_loss and AUC score on the current chunk
        print("Chunk %3d scores : loss %.6f auc %.6f [%5.1f min used so far]"
              % (i_chunk + 1, 
                 log_loss(df[get_target_feature()], preds),
                 roc_auc_score(df[get_target_feature()], preds),
                 (time.time() - start_time) / 60))
        
        del target_encoding
        gc.collect()
    
    return clf, avg_man
    

def predict_test_target(clf, avg_man):
    start_time=time.time()
    # Create place holder for the prediction
    predictions = None
    chunksize = 5000000
    # Read the test file by chunks
    for i_chunk, df in enumerate(pd.read_csv(get_test_file_name(), 
                                             chunksize=chunksize, 
                                             dtype=get_feature_types(), 
                                             usecols=get_working_features() + ['click_id'])):
        if predictions is None:
            # Get the click ids
            # double square brackets are used to return a DataFrame and not a Series
            predictions = df[['click_id']].copy() 
            # Encode df using average manager
            target_encoding = avg_man.apply_averages(df)
            # Predict probabilities with SGD Classifier
            predictions[get_target_feature()] = clf.predict_proba(target_encoding)[:, 1]
        else:
            # double square brackets are used to return a DataFrame and not a Series
            curr_preds = df[['click_id']].copy() 
            # Encode df using average manager
            target_encoding = avg_man.apply_averages(df)
            # Predict probabilities with SGD Classifier
            curr_preds[get_target_feature()] = clf.predict_proba(target_encoding)[:, 1]
            # Stack predictions and current predictions
            predictions: pd.DataFrame = pd.concat([predictions, curr_preds], axis=0)
            # free memory
            del curr_preds
            
        # Free memory by deleting the current DataFrame
        del df
        gc.collect()
    
        # Display the time we spent so far
        print("%3d Chunks have been read in %5.1f minute" 
              % (i_chunk + 1, (time.time() - start_time) / 60))

    return predictions

if __name__ == '__main__':
    gc.enable()
    
    clf, avg_man = train_classifier_on_averages()
    
    predictions = predict_test_target(clf, avg_man)
    
    predictions.to_csv('sgd_average_predictions.csv', float_format='%.6f', index=False)