#----------------------------IMPORT ALL THE PACKAGES WE NEED-----------------------------------------------------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing # Used for label encoding of 'Object' columns
from sklearn.linear_model import Ridge # Ridge Regression package
from sklearn.model_selection import GridSearchCV # Grid Search for tuning the Ridge Regression

#--------------------------IMPORT DATA AND CREATE DATAFRAMES----------------------------------------------------

#Below we use Pandas read_csv function to create dataframes of the Train and Test data, we parse_dates for the 'timestamp' column
train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

#Create a id_test variable that we can use later to join with the prediction. So each prediction has the right id attached
id_test = test.id

#-------------------------CLEAN DATA----------------------------------------------------------------------------

#clean data, built from many different kernels on Kaggle over time during the competition.
bad_index = train[train.life_sq > train.full_sq].index
train.ix[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]
bad_index = test[test.life_sq > test.full_sq].index
test.ix[bad_index, "life_sq"] = np.NaN
bad_index = train[train.life_sq < 5].index
train.ix[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 5].index
test.ix[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index
train.ix[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 5].index
test.ix[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
train.ix[bad_index, "full_sq"] = np.NaN
bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
test.ix[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index
train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test[test.life_sq > 200].index
test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1500].index
train.ix[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year < 1500].index
test.ix[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index 
train.ix[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index 
test.ix[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.ix[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test.ix[bad_index, "num_room"] = np.NaN
bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.ix[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.ix[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.ix[bad_index, "max_floor"] = np.NaN
bad_index = test[test.max_floor == 0].index
test.ix[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.ix[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index
test.ix[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train.ix[bad_index, "floor"] = np.NaN
train.material.value_counts()
test.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index
train.ix[bad_index, "state"] = np.NaN
test.state.value_counts()

# brings error down a lot by removing extreme price per sqm
train.loc[train.full_sq == 0, 'full_sq'] = 50
train = train[train.price_doc/train.full_sq <= 600000]
train = train[train.price_doc/train.full_sq >= 10000]

#-------------------------------CREATE NEW VARIABLES FROM DATA WE HAVE---------------------------------------------

# Add month-year
month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
train['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
train['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
train['month'] = train.timestamp.dt.month
train['dow'] = train.timestamp.dt.dayofweek

test['month'] = test.timestamp.dt.month
test['dow'] = test.timestamp.dt.dayofweek

# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

#-------------------------CREATE THE NEW TEST AND TRAIN DATAFRAMES----------------------------------------------

y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

#-------------------------FILL IN MISSING VALUES IN DATAFRAMES--------------------------------------------------

# Most machine learning algorithms give errors when we have missing data or 'NaN' values so we need to fill them in with something
# below we use a function that fills in these values with the mean of the column for numerical columns
# or the most frequent value for object columns


#Import the package we need for the fill in function
from sklearn.base import TransformerMixin

#Define the fill in (Imputer) function
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

# Create a new dataframe that we can pass to the imputer function for the train data
x_train_imp = pd.DataFrame(x_train)

# Use the imputer function on the new train dataframe, we use fit_transform on the train dataframe
x_train_imputer = DataFrameImputer().fit_transform(x_train_imp)

# Create a new dataframe that we can pass to the imputer function for the test data
x_test_imp = pd.DataFrame(x_test)

# Use the imputer function on the new test dataframe, we use fit_transform on the test dataframe
x_test_imputer = DataFrameImputer().fit_transform(x_test_imp)

#----------------------------------TRANSFORM ANY OBJECTS TO NUMERICAL VALUES---------------------------------------------------

# Now we want to go through each column and turn any 'Object' column in to a 'Numerical' column. Most machine learning
# algorithms need to have all columns in numerical values, so we use the below to turn objects to numerical representations
# it is encoding each object to a given numerical value.
for c in x_train_imputer.columns:
    if x_train_imputer[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train_imputer[c].values)) 
        x_train_imputer[c] = lbl.transform(list(x_train_imputer[c].values))
        
        
for c in x_test_imputer.columns:
    if x_test_imputer[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test_imputer[c].values)) 
        x_test_imputer[c] = lbl.transform(list(x_test_imputer[c].values))
         

#------------------------------MAKE SURE OUR Y VARIABLE IS AS LINEAR AS POSSIBLE--------------------------------------

# The last thing we want to do is try and make our y_train variable as linear as possible so we use the log transformation 
# on the data. Helps to create more accurate predictions in regression type machine learning algorithms such as Ridge Regression
y_train_array = np.log(y_train.values)

#----------------------------USE GRIDSEARCHCV TO CHOOSE THE PARAMETERS OF OUR RIDGE REGRESSION MODEL---------------------

# When doing training uncomment the lines with three ###, also make sure to comment out anything from "TRAIN OUR RIDGE REGESSION
# MODEL" and below

#prepare a range of parameters to test
###alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
###fit_interceptOptions = ([True, False])
###solverOptions = (['svd', 'cholesky', 'sparse_cg', 'sag'])
#create and fit a ridge regression model, testing each alpha
###model = Ridge(normalize=True) #We have chosen to just normalize the data by default, you could GridsearchCV this is you wanted
###grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas, fit_intercept=fit_interceptOptions, solver=solverOptions))
###grid.fit(x_train_imputer, y_train)
###print(grid)
# summarize the results of the grid search
###print(grid.best_score_)
###print(grid.best_estimator_.alpha)
###print(grid.best_estimator_.fit_intercept)
###print(grid.best_estimator_.solver)

#---------------------------TRAIN OUR RIDGE REGRESSSION MODEL-------------------------------------------------------

model = Ridge(normalize=True, alpha=0.1, fit_intercept=True, solver='sparse_cg') #paramters tuned using GridSearchCV
model.fit(x_train_imputer, y_train_array)

#--------------------------USE OUR MODEL TO PREDICT THE Y VALUE OF THE TEST DATA----------------------------------------

prediction = model.predict(x_test_imputer)

#------------------------CREATE AN OUTPUT OF OUR MODELS PREDICTION IN CONTEST FORMAT-----------------------------------

# The only unique thing here is that since we used Log transformation of our Y variable in training the model
# we need to make sure to use np.exp to apply exponential transformation of our predictions so they are back in 
# the format that the original data was in.
output = pd.DataFrame({'id': id_test, 'price_doc': np.exp(prediction)})
output.head()

#-----------------------CREATE A CSV OF OUR PREDICTION FOR US TO DOWNLOAD AND SUBMIT------------------------------------

output.to_csv('MyRidgeCV.csv', index=False)





