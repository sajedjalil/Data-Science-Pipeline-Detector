import numpy as np
import pandas as pd

import h2o
h2o.init(max_mem_size_GB=14)


# Settings
data_path = "../input/"
submission_path = 'submission/'
train = data_path+'train.csv'
test = data_path+'test.csv'
sample = data_path+'sample_submission.csv'

# rows to use from training data
nrows = 30000000

# data lists
train_h2o_types = ['numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'time', 'time', 'numeric']
test_h2o_types = ['numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'time']
category_cols = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']

# The columns to be used during training
X = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour', 'qty']
y = 'is_attributed'

#############################
#   CONVENIENCE FUNCTION    #
#############################
def process_data(df):
    """In-place pre-processing of train/test dataframe for analysis"""
    
    # Convert columns to factors
    for c in category_cols:
        if c in df.columns:
            df[c] = df[c].asfactor()
            
    # Extract time data
    df['day'] = df['click_time'].day()
    df['hour'] = df['click_time'].hour()
    
    # Add qty  (https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-val-auc-0-977)
    df['qty'] = df[['ip','day','hour']].merge(
        df[['ip','day','hour']].group_by(by=['ip','day','hour']).count().get_frame(), 
        all_x=True,
        by_x=['ip','day','hour'], 
        by_y=['ip','day','hour']
    )['nrow']

#####################
#   DATA LOADING    #
#####################

# Load data using h2o
train_data = h2o.import_file(train, col_types=train_h2o_types)

# Subset data using h2o to only use 'nrows' latest rows
total_rows = len(train_data)
train_data = train_data[total_rows-nrows:total_rows, :]

# Pre-process dataframe
process_data(train_data)

#######################
#   MODEL TRAINING    #
#######################

# Create classifier, balance classes
clf = h2o.estimators.random_forest.H2ORandomForestEstimator(balance_classes=True)

# Train classifier, delete train data afterwards
clf.train(X, y, training_frame=train_data)
del train_data

# Check model performance
clf.model_performance()

###################
#   Submission    #
###################

# Import test dataset
test_data = h2o.import_file(test, col_types=test_h2o_types)

# Process dataframe in place
process_data(test_data)

# Perform predictions
pred = clf.predict(test_data)

# Create the submission file
submission = pd.read_csv(sample)
submission[y] = h2o.as_list(pred['p1'])
submission.to_csv('h2o_randomforest.csv', index=False)