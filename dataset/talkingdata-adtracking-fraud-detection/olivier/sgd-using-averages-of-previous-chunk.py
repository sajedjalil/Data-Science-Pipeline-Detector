"""
This python script is part of our support work for DiveIntoCode students 
https://diveintocode.jp/ai_curriculum

The goal is to show how to target encode categorical features 
and update a Gradient Descent Classifier one chunk at a time

This is sort of an online learning.

Test file is then read and predictions are made using the full averages and fitted classifier

In this version of gradient descent:
 - averages calculated on a chunk are used on the next one before training SGD
 - This trains SGD to cope with values in the current chunk being uncovered by the averages
 
This script gets a better score than the original SGD script and will be used for future updates

Added : 
 - Hour and Minute
 - display of day and hour to understand the evolution of the scores
 - support for averages on multiple columns  

"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import SGDClassifier
from collections import defaultdict
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
    return [
        ['ip'], ['device'], ['channel'], ['os'], ['app'], ['hour'], 
        ['app', 'channel'], ['app', 'os'], ['os', 'channel'], ['ip', 'app']
    ]
    

def get_used_features():
    return ['ip', 'device', 'channel', 'os', 'app', 'click_time']
    

def get_target_feature():
    return 'is_attributed'


def add_time_features(df):
    # Create hour, minute and second features
    df['day'] = df['click_time'].apply(lambda x: np.uint8(x.split()[0].split('-')[2]))
    df['hour'] = df['click_time'].apply(lambda x: np.uint8(x.split()[1].split(':')[0]))
    df['minute'] = df['click_time'].apply(lambda x: np.uint8(x.split()[1].split(':')[1]))
    df['second'] = df['click_time'].apply(lambda x: np.uint8(x.split()[1].split(':')[2]))
    # Drop click_time feature and free up some memory 
    # del df['click_time']
    # gc.collect()
    
    
class AverageManager(object):
    """ Class that will manage target averages for selected feature and target encode these features """
    def __init__(self, features, target):
        """ 
        Init an average manager for the given features and the specified target
        :param features : expected to be a list of list of features to group by the data
        :param target : name of the target feature
        """
        # Check features : 
        for f_ in features:
            if type(f_) != list:
                raise ValueError('Features are expected to be provided as a list')
        # averages contains the average data
        self.averages = {
            tuple(f_): None for f_ in features
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
            the_group = df[list(f_) + [self.target]].groupby(list(f_)).agg(['sum', 'count'])
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
            # Check averages are fitted
            if self.averages[f_] is None:
                raise ValueError('Averages have not been fitted yet')
            # Compute the average
            self.averages[f_]['average'] = self.averages[f_]['sum'] / self.averages[f_]['count']
            # Now we need to encode for potetially several columns
            feat_name = '_' + '_'.join(list(f_))
            # Compute feataure on df
            # df[feat_name] = df[list(f_)].apply(lambda row: '_'.join(row.astype(str)), axis=1, raw=True)
            add_str_feature(df, list(f_), feat_name)
            # Compute feature on the average
            the_average = self.averages[f_].reset_index()
            # the_average[feat_name] = the_average[list(f_)].apply(lambda row: '_'.join(row.astype(str)), axis=1, raw=True)
            add_str_feature(the_average, list(f_), feat_name)
            the_average.set_index(feat_name, inplace=True)
            # finally map
            encoded[feat_name] = df[feat_name].map(the_average['average']).astype(np.float32)
            prior = self.prior['cum_sum'] / self.prior['nb_samples']
            encoded[feat_name].fillna(prior, inplace=True)
            # Drop feat_name from df
            del df[feat_name]
            gc.collect()
        
        return encoded
        

def add_str_feature(df_, features, name):
    """
    It does the same as : 
    df[feat_name] = df[list(f_)].apply(lambda row: '_'.join(row.astype(str)), axis=1, raw=True)
    However:
     - The addition of series is faster than the apply statement 
     - apply(lambda x: str(x)) is faster than df_[f].astype(str)
     
    Without this function it would take 7.5 minutes to complete 3 chunks when it takes 3.5 minutes using it!
    """
    df_[name] = ''
    for f in features:
        df_[name] += df_[f].apply(lambda x: str(x)) + '_'
    
    
def train_classifier_on_averages():
    start_time=time.time()
    # Create average manager
    avg_man = AverageManager(features=get_working_features(), target=get_target_feature())
    
    # Init Classifier, very simple not tuning yet
    clf = SGDClassifier(loss='log', tol=1e-2, random_state=1)
    
    # Read train file 
    chunksize=5000000
    for i_chunk, df in enumerate(pd.read_csv(get_train_file_name(), 
                                             chunksize=chunksize, 
                                             dtype=get_feature_types(), 
                                             usecols=get_used_features() + [get_target_feature()])):
                                                 
        # Add time features
        add_time_features(df)
        
        # Find the most important day
        day = df['day'].value_counts().index[0]
        hour = df['hour'].value_counts().index[0]
        
        if i_chunk == 0:
            # This our first read, only update averages here
            # Udpate averages with the average manager
            avg_man.update_averages(df)
        
        else:
            # Use averages to target encode df
            target_encoding = avg_man.apply_averages(df)
            df['click_time'] = pd.to_datetime(df['click_time'])
            # Reusing the code from Andrii Sydorchuk
            # https://www.kaggle.com/asydorchuk/nextclick-calculation-without-hashing-trick
            # However as I don't know the future I set this to a shift of 1 ;-)
            df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
            target_encoding['time_diff'] = - (df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1)
                - df.click_time).astype(np.float32).fillna(0) / 20000
            print(target_encoding['time_diff'].max(), target_encoding['time_diff'].min())
            
            # Train the classifier
            clf.partial_fit(X=target_encoding, y=df[get_target_feature()], classes=[0, 1])
            
            # Update averages
            avg_man.update_averages(df)
        
            # Get current predictions
            preds = clf.predict_proba(target_encoding)[:, 1]
            
            # Display the log_loss and AUC score on the current chunk
            print("Chunk %3d scores (day%2d|hour%2d): loss %.6f auc %.6f [%5.1f min used so far]"
                  % (i_chunk + 1, day, hour,
                     log_loss(df[get_target_feature()], preds),
                     roc_auc_score(df[get_target_feature()], preds),
                     (time.time() - start_time) / 60))
        
            del target_encoding, preds
            
        del df    
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
                                             usecols=get_used_features() + ['click_id'])):
                                                 
        # Add time features
        add_time_features(df)
        
        if predictions is None:
            # Get the click ids
            # double square brackets are used to return a DataFrame and not a Series
            predictions = df[['click_id']].copy() 
            # Encode df using average manager
            target_encoding = avg_man.apply_averages(df)
            
            # Add time difference
            df['click_time'] = pd.to_datetime(df['click_time'])
            df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
            target_encoding['time_diff'] = - (df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) 
                                            - df.click_time).astype(np.float32).fillna(0) / 20000
            
            # Predict probabilities with SGD Classifier
            predictions[get_target_feature()] = clf.predict_proba(target_encoding)[:, 1]
        else:
            # double square brackets are used to return a DataFrame and not a Series
            curr_preds = df[['click_id']].copy() 
            # Encode df using average manager
            target_encoding = avg_man.apply_averages(df)
            
            # Add time difference
            df['click_time'] = pd.to_datetime(df['click_time'])
            df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
            target_encoding['time_diff'] = - (df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) 
                                            - df.click_time).astype(np.float32).fillna(0) / 20000
            
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