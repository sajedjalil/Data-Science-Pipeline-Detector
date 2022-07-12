"""
This script fits a LightGBM model with "optimal" params on the Titanic dataset. 
It uses a gridsearch to find a reasonable set of parameters, then fits a number 
of models with different subsets of the training data with these params using 
early stopping. This technique can improve prediction stability for noisy datasets.

Note that the preprocessing here is done on the combined train and test sets.
In the real world this is bad practice, but here know we'll never see any new data, 
so it's fine.

This version can score around 0.822 (top ~3%), but anything ~0.77 or above is 
due to overfitting or luck :)
"""

import re
from typing import Tuple, Union, Dict, List, Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def split_cabin_str(s: str) -> Tuple[str, float]:
    """
    Function to try and extract cabin letter and number from the cabin column.
    Runs a regular expression that finds letters and numbers in the
    string. These are held in match.group, if they exist.
    """

    match = re.match(r"([a-z]+)([0-9]+)", s, re.I)

    letter = ''
    number = -1.0
    if match is not None:
        letter = match.group(1)
        number = match.group(2)

    return letter, number


def process_cabin(s: Union[str, float]) -> Tuple[str, float, int]:
    """From a cabin string, try and extract letter, number, and number of cabins."""
    # Check contents
    if not isinstance(s, str):
        # If field is empty, return nothing
        letter = ''
        number = ''
        n_rooms = -1.0
    else:
        # If field isn't empty, split sting on space. Some strings contain
        # multiple cabins.
        s = s.split(' ')
        # Count the cabins based on number of splits
        n_rooms = len(s)
        # Just take first cabin for letter/number extraction
        s = s[0]

        letter, number = split_cabin_str(s)

    return letter, number, n_rooms


def split_name_str(s: str, title_map: Dict[str, str]) -> Tuple[str, str]:
    """
    Extract title from name, replace with value in title dictionary and both title 
    and surname.
    """

    # Remove '.' from name string
    s = s.replace('.', '')
    # Split on spaces
    s = s.split(' ')
    # get surname
    surname = s[0]

    # Get title - loop over title_map, if s matches a key, take the
    # corresponding value as the title
    title = [t for k, t in title_map.items() if str(k) in s]

    # If no matching keys in title dict, use 'Other'.
    if len(title) == 0:
        title = 'Other'
    else:
        # Title is a list, so extract contents
        title = title[0]

    # Return surname (stripping remaining ',') and title as string
    return surname.strip(','), title


def prepare_for_light_gbm(
        data, target_col: Optional[str] = None, 
        id_col: Optional[str] = None,
        drop_cols: Optional[List[str]] = None) -> Tuple[lgb.Dataset, pd.Series,
                                                        pd.Series, pd.DataFrame]:
    """
    Prepare a dataframe containing processed titanic data for modelling.

    Creates a lbg.Dataset using the columns as specified.

    :param data: Dataframe to process.
    :param target_col: The column containing the labels/targets
    :param id_col:
    :param drop_cols: List of columns to drop.
    :return: prepared lbb.Dataset.
    """
    # Drop target column
    if target_col is not None:
        labels = data[target_col]
        drop_cols = drop_cols + [target_col]
    else:
        labels = []

    if id_col is not None:
        ids = data[id_col]
        drop_cols = drop_cols + [id_col]
    else:
        ids = []

    if drop_cols is not None:
        data = data.drop(drop_cols, axis=1)

    # Create LGB mats
    lgb_data = lgb.Dataset(data, label=labels, free_raw_data=False,
                           feature_name=list(data.columns), 
                           categorical_feature='auto')

    return lgb_data, labels, ids, data


if __name__ == "__main__":
    # Import both data sets
    data_train = pd.read_csv('../input/train.csv')
    data_test = pd.read_csv('../input/test.csv')

    # And concatenate together
    n_train = data_train.shape[0]
    data_joined = pd.concat([data_train, data_test], axis=0)

    # Process cabins
    # Apply process_cabin function to each row in 'Cabin' column individually 
    # using pandas apply method.
    cabins = data_joined['Cabin'].apply(process_cabin)
    # Output tuple has 3 values for each row, convert this to pandas df
    cabins = cabins.apply(pd.Series)
    # And name the columns
    cabins.columns = ['cabin_letter', 'cabin_number', 'cabins_number_of']

    # Then concatenate these columns to the dataset
    data_joined = pd.concat([data_joined, cabins], axis=1)

    # Process family
    # Add some family features directly to new columns in the dataset
    # Size
    data_joined.loc[:, 'fSize'] = data_joined['SibSp'] + data_joined['Parch'] + 1
    # Ratio
    data_joined.loc[:, 'fRatio'] = ((data_joined['Parch'] + 1) 
                                    / (data_joined['SibSp'] + 1))
    # Adult?
    data_joined.loc[:, 'Adult'] = data_joined['Age'] > 18

    # Process titles
    # Extract titles from Name column, standardise
    title_map_ = {"Capt": "Officer", "Col": "Officer", "Major": "Officer", 
                  "Jonkheer": "Sir", "Don": "Sir", "Sir": "Sir", "Dr": "Dr", 
                  "Rev": "Rev", "theCountess": "Lady", "Dona": "Lady", 
                  "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs", "Mr": "Mr", 
                  "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
                  "Lady": "Lady"}

    # Apply functions to df and concatenate new columns as before
    cabins = data_joined['Name'].apply(split_name_str, args=[title_map_])
    cabins = cabins.apply(pd.Series)
    cabins.columns = ['Surname', 'Title']

    data_joined = pd.concat([data_joined, cabins], axis=1)

    # Process categorical columns
    # List of categorical columns to recode
    cat_cols = ['Sex', 'Embarked', 'cabin_letter', 'cabin_number', 'Surname', 
                'Title']

    # Recode
    for c in cat_cols:
        # Convert column to pd.Categorical
        data_joined.loc[:, c] = pd.Categorical(data_joined[c])
        # Extract the cat.codes and replace the column with these
        data_joined.loc[:, c] = data_joined[c].cat.codes
        # Convert the cat codes to categorical...
        data_joined.loc[:, c] = pd.Categorical(data_joined[c])

    # Age
    # Replace missing age values with median.
    # See other kernels for more sophisticated ways of doing this!
    data_joined.loc[data_joined.Age.isnull(), 'Age'] = \
        np.median(data_joined['Age'].loc[data_joined.Age.notnull()])

    # Split datasets
    train_processed = data_joined.iloc[0:n_train, :]
    test_processed = data_joined.iloc[n_train::, :]

    # Specify columns to drop
    columns_to_drop = ['Ticket', 'Cabin', 'Name']

    # Split training data in to training and validation sets.
    # Validation set is used for early stopping.
    train_split_df, valid_split_df = train_test_split(train_processed, 
                                                      test_size=0.4)

    # Prepare the data sets
    (train_lgb_dataset, train_labels,
     train_ids, train_split_df) = prepare_for_light_gbm(
        train_split_df, target_col='Survived', id_col='PassengerId',
        drop_cols=columns_to_drop)

    (valid_lgb_dataset, valid_labels,
     valid_ids, valid_split_df) = prepare_for_light_gbm(
        valid_split_df, target_col='Survived', id_col='PassengerId',
        drop_cols=columns_to_drop)

    test_lgb_dataset, _, _, test_df = prepare_for_light_gbm(
        test_processed, target_col='Survived', id_col='PassengerId',
        drop_cols=columns_to_drop)

    # Prepare data set using all the training data
    (train_valid_lgb_dataset, train_valid_labels,
     _, train_valid_df) = prepare_for_light_gbm(
        train_processed, target_col='Survived', id_col='PassengerId',
        drop_cols=columns_to_drop)

    # Set params
    # Scores ~0.784 (without tuning and early stopping)
    params = {'boosting_type': 'gbdt', 'max_depth': -1, 'objective': 'binary', 
              'num_leaves': 64, 'learning_rate': 0.05, 'max_bin': 512, 
              'subsample_for_bin': 200, 'subsample': 1, 'subsample_freq': 1,
              'colsample_bytree': 0.8, 'reg_alpha': 5, 'reg_lambda': 10, 
              'min_split_gain': 0.5, 'min_child_weight': 1, 
              'min_child_samples': 5, 'scale_pos_weight': 1, 'num_class': 1, 
              'metric': 'binary_error'}

    # Create parameters to search
    grid_params = {'learning_rate': [0.01], 'n_estimators': [8, 24],
                   'num_leaves': [6, 8, 12, 16], 'boosting_type': ['gbdt'], 
                   'objective': ['binary'], 'seed': [500],
                   'colsample_bytree': [0.65, 0.75, 0.8], 
                   'subsample': [0.7, 0.75], 'reg_alpha': [1, 2, 6],
                   'reg_lambda': [1, 2, 6]}

    # Create classifier to use. Note that parameters have to be input manually
    # not as a dict!
    mod = lgb.LGBMClassifier(**params)

    # To view the default model params:
    mod.get_params().keys()

    # Create the grid
    grid = GridSearchCV(mod, param_grid=grid_params, verbose=1, cv=5, n_jobs=-1)
    # Run the grid
    grid.fit(train_valid_df, train_valid_labels)

    # Print the best parameters found
    print(grid.best_params_)
    print(grid.best_score_)

    # Using parameters already set above, replace in the best from the grid search
    best_params = {k: grid.best_params_.get(k, v) for k, v in params.items()}
    best_params['verbosity'] = -1

    # Kit k models with early-stopping on different training/validation splits
    k = 5
    valid_preds, train_preds, test_preds = 0, 0, 0
    for m in range(k):
        print('Fitting model', m)

        # Prepare the data set for fold
        train_split_df, valid_split_df = train_test_split(train_processed, 
                                                          test_size=0.4)
        (train_lgb_dataset, train_labels,
         train_ids, train_split_df) = prepare_for_light_gbm(
            train_split_df, target_col='Survived', id_col='PassengerId',
            drop_cols=columns_to_drop)
        
        (valid_lgb_dataset, valid_labels,
         valid_ids, valid_split_df) = prepare_for_light_gbm(
            valid_split_df, target_col='Survived', id_col='PassengerId',
            drop_cols=columns_to_drop)
        
        # Train
        gbm = lgb.train(best_params, train_lgb_dataset, num_boost_round=100000,
                        valid_sets=[train_lgb_dataset, valid_lgb_dataset],
                        early_stopping_rounds=50, verbose_eval=50)

        # Plot importance
        lgb.plot_importance(gbm)
        plt.show()

        # Predict
        valid_preds += gbm.predict(valid_split_df, 
                                   num_iteration=gbm.best_iteration) / k
        train_preds += gbm.predict(train_split_df, 
                                   num_iteration=gbm.best_iteration) / k
        test_preds += gbm.predict(test_df, num_iteration=gbm.best_iteration) / k

    # Save submission
    sub = pd.DataFrame()
    sub['PassengerId'] = test_processed['PassengerId']
    sub['Survived'] = np.int32(test_preds > 0.5)
    sub.to_csv('sub2.csv', index=False)
