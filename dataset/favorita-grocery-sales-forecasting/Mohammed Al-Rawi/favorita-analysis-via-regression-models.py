# Starter code for multiple regressors implemented by Mohammed Al-Rawi
# Source code based on: Leandro dos Santos Coelho several regresor analysis,
# which is based on Forecasting Favorites, 1owl
# https://www.kaggle.com/the1owl/forecasting-favorites , version 10

# Author: Mohammed Al-Rawi
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning) # numpy NaNs are causing a few warnings, toggle this of or on, as you wish
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn import preprocessing, linear_model, metrics
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
import gc; gc.enable()
import time
np.random.seed(123456)

#################################### Functions #################################
# Encoding of values
def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder() 
            df[c] = lbl.fit_transform(df[c])
            print(c)
    return df

# df transform does new indecies for data and conversions for onpromotion and perishable
def df_transform(df):
    df['date'] = pd.to_datetime(df['date']) # this should be pulled out of classification
    df['yea'] = df['date'].dt.year
    df['mon'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek # this feature is needed to score peek days, like Saturday(s) and Sunday(s)
    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})
    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})
    df = df.fillna(-1)
    return df

# The error metric
def NWRMSLE(y, pred, w):
    return metrics.mean_squared_error(y, pred, sample_weight=w)**0.5

# Returns the regressor model and its name
def get_method(method_nbr):

    # set the seed to generate random numbers
    ra1 = 123456
    np.random.seed(ra1)
    

    print('\n method = ', method_nbr) # selection the model, denoted as r
    
    if (method_nbr==1):
        print('Multilayer perceptron (MLP) neural network 01')
        str_method = 'MLP model01'    
        r = MLPRegressor(hidden_layer_sizes=(2,), max_iter=40)

    if (method_nbr==2):
        print('Linear model (classical)')
        str_method = 'Linear model'    
        r = linear_model.LinearRegression(n_jobs=-1)
    
    if (method_nbr==3):
        print('Bagging Regressor -1')
        str_method = 'BaggingRegressor'
        rng = check_random_state(ra1)
        r = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                           n_estimators = 8,
                         # bootstrap = True,
                         # oob_score = True,
                           random_state = rng )
        
    if (method_nbr==4):
        print('Bagging Regressor -2')
        str_method = 'BaggingRegressor'
        rng = check_random_state(ra1)
        r = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                           n_estimators = 16,
                         # bootstrap = True,
                         #  oob_score = True,
                           random_state = rng )
        r = RANSACRegressor(random_state=ra1)
        
    if (method_nbr==5):
        print('Bagging Regressor')
        str_method = 'BaggingRegressor -3'
        rng = check_random_state(ra1)
        r = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                           n_estimators = 32,
                         # bootstrap = True,
                         #  oob_score = True,
                           random_state = rng )
    
    if (method_nbr==6):
        print('Random forest')
        str_method = 'RandomForest'
        r = RandomForestRegressor(n_estimators= 16, max_depth=16, n_jobs=-1, 
                                   random_state=ra1, verbose=0, warm_start=True)
    
    return r, str_method
    
################ End of get_method() ##############

# A function to normalize x
def normalize_feature(x, method="log1p"):
    x[x < 0.] = 0. # Removing negatives
    
    if method=="log1p":
        return np.log1p(x)
    return 0

def normalization_inverse(data, method="log1p"):
# inputs-
# data:- Could be train, validation, or test
    if method=="log1p":
        cut = 1e-15
      #  data[data_name_str] = (np.exp(data[data_name_str]) - 1).clip(lower=cut) # Removing the log transform from the data
        return (np.exp(data) - 1).clip(lower=cut) # Removing the log transform from the data
 

def read_as_df():
    print('Reading Favorita Data...')
    folder_path = '../input/'
    dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales': 'float32', 'onpromotion':str}
    data = {
        'tra': pd.read_csv(folder_path+'train.csv', dtype=dtypes, parse_dates=['date']),
        'tes': pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date']),
        'ite': pd.read_csv('../input/items.csv'),
        'sto': pd.read_csv('../input/stores.csv'),
        'trn': pd.read_csv('../input/transactions.csv', parse_dates=['date']),
        'hol': pd.read_csv('../input/holidays_events.csv', dtype={'transferred':'bool'}, parse_dates=['date']),
        'oil': pd.read_csv('../input/oil.csv', dtype={'dcoilwtico':'float32'}, parse_dates=['date']),
        }
    return data


def data_to_train_test(data):
    print('Data Processing...')
    # One could use all month 8, that is, day>=1, or, by removing the day condition
    train = data['tra'][(data['tra']['date'].dt.month == 8) & (data['tra']['date'].dt.day > 15)] # Picking only the second half of Month 8 from each year, as this matches the time interval of the test data
    del data['tra']; gc.collect();
    x= train['unit_sales'].values
    train['unit_sales'] = normalize_feature(x) # Squeezing the values' range using natural logarithm
    # test['unit_sales'] does not exist, this is what we are trying to predict

    # item_nbr scale should probably be resolved as it is used as input, so, normalization could be used
    # or, build a classifier for each item number, and a classifier based on the family of product for the unseen test set 
    data['ite'] = df_lbl_enc(data['ite']) # ??, not sure why this is used in here!!!
    train = pd.merge(train, data['ite'], how='left', on=['item_nbr'])
    test = pd.merge(data['tes'], data['ite'], how='left', on=['item_nbr'])
    del data['tes']; gc.collect();
    del data['ite']; gc.collect();
    

    train = pd.merge(train, data['trn'], how='left', on=['date','store_nbr']) # trn denotes transactions (and not training!)
    test = pd.merge(test, data['trn'], how='left', on=['date','store_nbr'])
    del data['trn']; gc.collect();
    x = train['transactions'].values
    train['transactions'] = normalize_feature(x) 
    # test['transactions'] does not exist

    data['sto'] = df_lbl_enc(data['sto'])
    train = pd.merge(train, data['sto'], how='left', on=['store_nbr'])
    test = pd.merge(test, data['sto'], how='left', on=['store_nbr'])
    del data['sto']; gc.collect();

    data['hol'] = data['hol'][data['hol']['locale'] == 'National'][['date','transferred']]
    data['hol']['transferred'] = data['hol']['transferred'].map({'False': 0, 'True': 1})
    train = pd.merge(train, data['hol'], how='left', on=['date'])
    test = pd.merge(test, data['hol'], how='left', on=['date'])
    del data['hol']; gc.collect();


    # Filling the missing values of oil prices, using interpolation
    data['oil'].interpolate(method='akima') 
    train = pd.merge(train, data['oil'], how='left', on=['date'])
    test = pd.merge(test, data['oil'], how='left', on=['date'])
    del data; del x; gc.collect();
    

    # Oil prices should also be normalized with logp1, or, another scaling/normalization 
    # could be used to all values, 0 mean and unit variance (probably)
    x = train['dcoilwtico'].values 
    train['dcoilwtico'] = normalize_feature(x)
    x = test['dcoilwtico'].values
    test['dcoilwtico'] = normalize_feature(x)
    del x; gc.collect();
    train = df_transform(train)
    test = df_transform(test)
    test.is_copy = False # to get rid of later copy warnings
    
    return train, test

#------------------- forecasting based on multiple regressors models ---------------------------
def run_regressors(trn_set, trn_label, validation_set, validation_label, test, regressors_list, save_output=False):
    print('\n Training, validation, then testing regressors approach ... please sit tight')
    col = [c for c in trn_set if c not in ['id', 'date', 'yea', 'mon', 'unit_sales','perishable','transactions']] 
    print('The input features are: ', col)
# The col index that'll be used to fetch the values in training, validation, and testing
# Here, we are removing id, unit_sales, perishale, and transactions from the indexing
# this is bcuz id, which is just a number, should not be relivant to the classification, unit_sales are the lables and transactions
# are the accumulated unit_sales 
# I am also removing the date, yea and mon, as we are picking only month 8, and year has no effect as it is fided per year, and oil prices will correlate with the days instead
# If one wants to consider all the days of the year, then, it is better to normalize the days to (1 to 365) using xx = datetime.toordinal(), 
## where xx should be set to 1 to 365 (i.e., by taking care of the seed/start of the ordinal year which is year=1, month =1, day=1), such treatement should 
# probably pay attention to the leap year (which has 365 days every four years).
    
    for method in regressors_list:
        r, str_method = get_method(method) # getting the model and its name
        r.fit(trn_set[col], trn_label) # Training the model r
        predicted_validation_label = r.predict(validation_set[col])
        err_val = NWRMSLE(validation_label, predicted_validation_label, validation_set['perishable']) # Finding the error metric
        print('Performance: NWRMSLE = ',  str(err_val)) # So, how much was the error score?
        test['unit_sales'] = r.predict(test[col])  # Testing the trained model on the competition test data and producing the unit_sales
        test['unit_sales']  = normalization_inverse(test['unit_sales']) # test['unit_sales'] = (np.exp(test['unit_sales']) - 1).clip(lower=cut) # Removing the log transform from the data
        output_file = 'Sub- ' + str_method + '-' + str(err_val) + '.csv' 
        if save_output:
            test[['id','unit_sales']].to_csv(output_file, index=False, float_format='%.2f') # Saving, only id and unit_sales, to CSV file


def split_train_test(train, method='keep_2016_for_validation'):
    if (method=='keep_2016_for_validation'):
        trn_set = train[(train['yea'] != 2016)] # Used for training
        trn_label = trn_set['unit_sales'].values 
        validation_set = train[(train['yea'] == 2016)] # Used for validation
        validation_label = validation_set['unit_sales'].values
        del train; gc.collect() # train is not used anymore
#       trn_label_trans = trn_set['transactions'].values  # use this only if you want to set transactions as labels instead of unit_sales
#       validation_label_trans = validation_set['transactions'].values

    return trn_set, trn_label, validation_set, validation_label


def split_store_nbr(train, store_nbr, item_nbr):
    
    train = train[ train['store_nbr'] == store_nbr ]
    trn_set = train[ train['item_nbr'] == item_nbr  ] # Used for training
    trn_label = trn_set['unit_sales'].values 
    del train; gc.collect() # train is not used anymore

    return trn_set, trn_label



######### Main program #################
print('Enhanced Favorita Analysis \n')
#  to get the unique items from a df ...  df['y'].unique
regressors_list = list([2,5])  
start_time = time.time() # store the initial time
data = read_as_df()
train, test = data_to_train_test(data)
trn_set, trn_label, validation_set, validation_label =  split_train_test(train, method='keep_2016_for_validation')
run_regressors(trn_set, trn_label, validation_set, validation_label, test, regressors_list, False)
print( "............. \n Voilà  .......\n ...") #      
print ("Total time %s min ",  (time.time() - start_time)/60 )