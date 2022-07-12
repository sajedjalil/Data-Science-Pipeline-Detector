# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# Clearing up memory
import gc

# Featuretools for automated feature engineering
import featuretools as ft

# Suppress pandas warnings
import warnings
warnings.filterwarnings('ignore')

print('Reading in data')

# Read in the full datasets
app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
cash = pd.read_csv('../input/POS_CASH_balance.csv')
credit = pd.read_csv('../input/credit_card_balance.csv')
previous = pd.read_csv('../input/previous_application.csv')
installments = pd.read_csv('../input/installments_payments.csv')

# Join the application dataframes together
app_test['set'] = 'test'
app_test['TARGET'] = -999
app_train['set'] = 'train'

# Append the dataframes (this is a row bind in R)
app = app_train.append(app_test, ignore_index = True)

# Create the entity set with an id
es = ft.EntitySet(id = 'applications')

# Add in all the entities

# Entities with a unique index
es = es.entity_from_dataframe(entity_id = 'app', dataframe = app, index = 'SK_ID_CURR')

es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, index = 'SK_ID_BUREAU')

es = es.entity_from_dataframe(entity_id = 'previous', dataframe = previous, index = 'SK_ID_PREV')

# Entities that do not have a unique index
es = es.entity_from_dataframe(entity_id = 'bureau_balance', dataframe = bureau_balance, 
                              make_index = True, index = 'bb_index')

es = es.entity_from_dataframe(entity_id = 'cash', dataframe = cash, 
                              make_index = True, index = 'cash_index')

es = es.entity_from_dataframe(entity_id = 'installments', dataframe = installments,
                              make_index = True, index = 'in_index')

es = es.entity_from_dataframe(entity_id = 'credit', dataframe = credit,
                              make_index = True, index = 'credit_index')


# Relationship between app and bureau
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

# Relationship between bureau and bureau balance
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

# Relationship between current app and previous apps
r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

# Relationships between previous apps and cash, installments, and credit
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

# Add in the defined relationships
es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                           r_previous_cash, r_previous_installments, r_previous_credit])
                           
print(es)
                           
print('Clearing up memory')

gc.enable()
# Clear up memory
del app, bureau, bureau_balance, cash, credit, installments, previous
gc.collect()

print('Deep Feature Synthesis in Progress')

# Default primitives from featuretools
default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]

# DFS for application features using a max depth of 2
feature_matrix, feature_names = ft.dfs(entityset = es, target_entity = 'app',
                       trans_primitives = default_trans_primitives,
                       agg_primitives=default_agg_primitives, 
                       max_depth = 2, features_only=False, verbose = True)
                       
# Reset the index to make SK_ID_CURR a column again                                      
feature_matrix = feature_matrix.reset_index()

print('Saving features')
feature_matrix.to_csv('feature_matrix.csv', index = False)