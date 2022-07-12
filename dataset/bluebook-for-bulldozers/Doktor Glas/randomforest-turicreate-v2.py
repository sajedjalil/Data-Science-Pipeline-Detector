# Using magic commands
%matplotlib inline
%load_ext autoreload
%autoreload 2

# Imports of the functions and the libraries

import turicreate as tc
import turicreate.aggregate as agg
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import math
import dateutil.parser
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
import numpy as np

tc.visualization.set_target('browser')
Path = "FastAI/ML/bluebook-for-bulldozers/"

!ls {Path}

# Import data
sf = tc.SFrame.read_csv(Path+'/train.csv', header = True)
sf.head(10)
sf.shape

# Metric is RMSLE, so we will take a log of the sales price
sf['Saleprice_log'] = sf['SalePrice'].apply(lambda x: math.log(x))
sf = sf.remove_column('SalePrice')

# Handle dates
sf['saledate'] = sf['saledate'].apply(lambda x: dateutil.parser.parse(x))
sf = sf.split_datetime('saledate', limit=['year','month','day', 'weekday', 'isoweekday', 'tmweekday'])
sf.show()

# Split train and test
sf['label'] = sf['Saleprice_log']
sf = sf.remove_column('Saleprice_log')
sf['age'] = sf['YearMade'] - sf['saledate.year']

train_data = sf[0:320000]
test_data = sf[320000:len(sf)]
train_data.shape
test_data.shape
seed = 34532
# Let's use the turicreate's internal methods

model = tc.random_forest_regression.create(train_data, target='label', max_iterations= 100, random_seed =seed, verbose=True, max_depth= 40, metric = 'rmse')

model.predict(test_data)

test_data['label']
model.predict(train_data)

train_data['label']

results = model.evaluate(test_data, metric='rmse')
results


# Further iterations

sf_copy = sf
sf_tmp = tc.SFrame()
lst = []
for col in sf_copy.column_names():
    lst.append(((len(sf_copy)- tc.SArray.nnz(sf_copy[col]))/len(sf_copy))*100)

sf_tmp = sf_tmp.add_column(sf.column_names())
sf_tmp = sf_tmp.add_column(lst)

sf_tmp.sort('X2',ascending = False).print_rows(58)
# Many columns have missing values
## Let's treat the numeric missing values but before let's change the SFrame to DataFrame as SFrame isn't letting me work with the dtype

df = tc.SFrame.to_dataframe(sf_copy)

#Numerical missing values
df_1 = df.select_dtypes(include = ['int64', 'float64'])
for col in df_1.columns:
    df_1[col] = df_1[col].fillna(df_1[col].mean())
# or do this
df_1 = df_1.fillna(df_1.mean())

# Categorical data missing values
df_2 = df.select_dtypes(include=['object']).copy()
for col in df_2.columns:
    df_2[col] = df_2[col].astype('category')
print(df_2.dtypes)

for col in df_2.columns:
    df_2[col] = df_2[col].cat.codes


def cat_na(df):
    for dt, col in zip(df.dtypes, df):
        if str(dt) == 'category':
            df[col] = df[col].fillna(df[col].mode().iloc[0])

cat_na(df_2)

pd.DataFrame((df_1.isna().sum()/len(df_1)).sort_values(ascending=False), columns=['Null percent'])
df_f = pd.concat([df_1, df_2], axis =1)
df_f.shape
df_f.columns

## Train test split

train_f = df_f[0:320000]
test_f = df_f[320000:len(df_f)]
#train_data
y = train_f['label']
df_m = train_f.drop(['label'], axis = 1)

#test_data
y_test = test_f['label']
df_m_test = test_f.drop(['label'], axis = 1)

#Base line Model without any parameter tuning
# n_jobs = -1 means you want to parallelise the code. You are asking to create a separate job for each of the CPU your machine has.
m = RandomForestRegressor(n_jobs=-1)
m.fit(df_m, y)
m.score(df_m,y)


def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())

# Training predictions (to demonstrate overfitting)
train_rf_predictions = m.predict(df_m)

#train rmse
rmse(train_rf_predictions, y)

# Testing predictions (to determine performance)
rf_predictions = m.predict(df_m_test)
#test score
m.score(df_m_test, y_test)
#test rmse
rmse(rf_predictions, y_test)

# n_estimators means the number of trees we want to grow, generally 30 to 40 trees seem a good number
# min_samples_leaf is another paramter that can be tuned, it tells about minimum number of samples required at each leaf node for it to be forked further. I generally use 1, 3, 5
# max_depth controls the depth of the tree, otherwise the tree will try to achieve purity in each leaaf node
# max_features is about choosing a random subset of columns to grow and split each tree. 1 means all, 0.5 means half of the columns, sqrt and log 2 are also good options
param_grid = {
    'bootstrap': [True],
    'max_depth': [3, 5, None],
    'max_features': [0.5, 1, 'sqrt'],
    'min_samples_leaf': [1, 3],
    'n_estimators': [10, 20, 40]
}

def hyperparameter_opt (df, y):
    g_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid = param_grid , cv = 2, verbose = 1, n_jobs = -1)
    g_results = g_search.fit(df, y)
    params_selected = g_results.best_params_
    return params_selected, g_search

params_selected, g_search = hyperparameter_opt(df_m, y)

model_search = RandomForestRegressor(bootstrap=params_selected['bootstrap'], max_depth=params_selected["max_depth"], \
                            n_estimators=params_selected["n_estimators"], max_features=params_selected['max_features'],\
                            min_samples_leaf=params_selected['min_samples_leaf'], random_state=False, verbose=True)


model_search.fit(df_m, y)
model_search.score(df_m, y)

model_search.fit(df_m_test, y_test)
model_search.score(df_m_test, y_test)
 model_search_pred = model_search.predict(df_m_test)
 rmse(model_search_pred, y_test)

var_importances = list(model_search.feature_importances_)
feature_list = df_m.columns
feature_data = {'feature': feature_list, 'importances' : var_importances}
feature_importances = pd.DataFrame(feature_data)
sorted_features = feature_importances.sort_values('importances', ascending=False).head(20)


#plotting graph
x_values = list(range(len(sorted_features)))
x_values
# Make a bar chart
def plt_chart(x_values, importances, feature_list ):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8,8))
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Variable Importance'); plt.xlabel('Feature'); plt.title('RF variable Importances');
    plt.bar(x_values, importances, orientation = 'vertical')


plt_chart(x_values, sorted_features.importances, sorted_features.feature)


# Cumulative importances
cumlative_features = feature_importances.sort_values('importances', ascending=False)
cumulative_importances = np.cumsum(cumlative_features.importances)
x_c = list(range(len(cumlative_features)))

def cumulative_plot(x_c, cumulative_importances):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(20,10))
    # Draw line at 95% of importance retained
    plt.hlines(y = 0.95, xmin=0, xmax=len(cumlative_features), color = 'b', linestyles = 'dashed')
    # Format x ticks and labels
    plt.xticks(x_c, cumlative_features.feature, rotation = 'vertical')
    # Axis labels and title
    plt.xlabel('Feature'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances')
    # Make a line graph
    plt.plot(x_c, cumulative_importances, 'r-')

cumulative_plot(x_c, cumulative_importances)
