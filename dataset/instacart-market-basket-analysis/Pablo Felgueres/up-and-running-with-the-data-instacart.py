######################################################################################################################
# Author: Pablo Felgueres

# Date: July 19th, 2017

# Kaggle Competition: Instacart Market Basket Analysis
 
# Description: 
# This script will get you up and running with the competitions data.
# Handles preprocessing of data and returns pickled dataframes for train, validation and testing datasets.
# Code is self-explanatory but follow comments for clarity.
# Runs on 8GB of RAM easily.
#######################################################################################################################

import pandas as pd
import numpy as np
from os import path, listdir

class Preprocess(object):
    '''
    Handles preprocessing of data and returns pickle files for train, validation and testing datasets.

    Parameters
    ----------
    path: str
        Path to folder with data files.

    Output
    ------
    train: pickle
        File containing dataset to train models.
    validation: pickle
        File containing dataset to cross-validate models.
    test: pickle
        File containing dataset for testing (30% of comp's data, remaining 70% is the official test)
    '''

    def __init__(self, datapath = '../data/'):
        '''
        Create a folder called data and insert all comp files to it:
        Folder structure should look like this. 
        .
        +-- data
        |   +-- aisles.csv
        |   +-- departments.csv
        |   +-- order_products__prior.csv
        |   +-- etc
        +-- src
        |   +-- __init__.py
        |   +-- THIS-SCRIPT.py

        Initialize with string to data folder.
        '''
        self._files = [path.join(datapath, file) for file in sorted(listdir(datapath)) if file.endswith('csv')]

    def _load2df(self):
        '''
        Load data into dataframes
        '''
        #Specify dtypes for memory optimization
        dtypes_order_products ={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}

        dtypes_orders = {
        'order_id': np.int32,
        'user_id': np.int32,
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float16}

        dtypes_products = {
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8}

        # Load data to dataframes
        self._df_aisles = pd.read_csv(self._files[0])
        self._df_dpts = pd.read_csv(self._files[1])
        self.df_order_prior = pd.read_csv(self._files[2], dtype = dtypes_order_products)
        self.df_order_train = pd.read_csv(self._files[3], dtype = dtypes_order_products)
        self.df_orders = pd.read_csv(self._files[4], dtype = dtypes_orders)
        self.df_products = pd.read_csv(self._files[5], dtype = dtypes_products)

    def _products(self):
        '''
        Merge product-related dataframes.
        '''
        #Merge aisles, dpt and products -- product is left
        self.df_products = self.df_products.merge(self._df_aisles, left_on= 'aisle_id', right_on= 'aisle_id')
        self.df_products = self.df_products.merge(self._df_dpts, left_on = 'department_id', right_on= 'department_id')

    def _users(self):
        '''
        Get a list of users corresponding to the training and testing dataset.
        '''
        users_train_all = self.df_orders.loc[(self.df_orders.eval_set == "train")].user_id
        #The users test dataset is the hold out only for final submissions
        self.users_test = self.df_orders.loc[(self.df_orders.eval_set == "test")].user_id
        #Divide the train dataset into a training and validation dataset
        self.users_train = users_train_all.sample(frac = 0.8, random_state = 10)
        #validation dataset / users that are not present in the users train set
        self.users_val = users_train_all[~users_train_all.isin(self.users_train)]

    def _merger(self):
        '''
        Merge orders with details of priors.
        '''
        # Use all prior orders and merge to df_orders on order id.
        df_prior_order_details = pd.merge(left=self.df_order_prior,
                                 right=self.df_orders,
                                 how='left',
                                 on='order_id')

        # Add detail dataframe of products
        df_prior_order_details = pd.merge(left=df_prior_order_details,
                                 right=self.df_products,
                                 how='left',
                                 on='product_id')

        self.df_prior_order_details = df_prior_order_details

        # Merge details to df_order_train as well.

        df_train_order_details = pd.merge(left=self.df_order_train,
                                 right=self.df_orders,
                                 how='left',
                                 on='order_id')

        # Add detail dataframe of products
        df_train_order_details = pd.merge(left=df_train_order_details,
                                 right=self.df_products,
                                 how='left',
                                 on='product_id')

        self.df_train_order_details = df_train_order_details

    def _partition(self):
        '''
        Separate orders details into training, validation and test dataset.
        Pickle results for easy retrieval.
        '''
        #These are the priors for the three sets (Put in another way, these is your featurespace X in f(X))
        self.df_prior_order_details.loc[self.df_prior_order_details.user_id.isin(self.users_train)].to_pickle('../data/X_train.pickle')
        self.df_prior_order_details.loc[self.df_prior_order_details.user_id.isin(self.users_val)].to_pickle('../data/X_val.pickle')
        self.df_prior_order_details.loc[self.df_prior_order_details.user_id.isin(self.users_test)].to_pickle('../data/X_test.pickle')
        #Now response variables for the train and validation dataset only (test is not included in data for comp purposes)
        self.df_train_order_details.loc[self.df_train_order_details.user_id.isin(self.users_train)].to_pickle('../data/Y_train.pickle')
        self.df_train_order_details.loc[self.df_train_order_details.user_id.isin(self.users_val)].to_pickle('../data/Y_val.pickle')

    def fit(self):
        '''
        Fit preprocessing methods
        '''
        self._load2df()
        self._users()
        self._products()
        self._merger()
        self._partition()

if __name__ == '__main__':
    a = Preprocess()
    a.fit()