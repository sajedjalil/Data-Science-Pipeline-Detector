# This Python 3 environment comes with many helpful analytics libraries installed
'''
Airbnb compition for interview - I am able to discuss every line of code with good details 
Author 85% Ahmed Shehata 15% looking at some functionality to extract extra feature from ses 
'''
#Importing The Nescessary Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets, feature_extraction
import xgboost as xgb
import os

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#%matplotlib inline

# Input data files are available in the "../input/" directory.

print(os.listdir("../input"))


####Start uploading user data#######
## Loading both the training and testing file into pandas dataframe 
print("Upload user data")
test_users = pd.read_csv("../input/test_users.csv")
train_users = pd.read_csv("../input/train_users_2.csv")
print(train_users.shape) ## We have around 213k records rows and 16 columns - KAGGLE  BASED
print(test_users.shape) ## We have arounk 62k records and 15 columns -- Which is the prediction targe variable

# Exploring just the first 4 records
# NDF - Stands for no destination found .. hence date_first_booking will be null ; 
# Other -  means there was a destination found but its not in the list 
#train_users.head(n=4)

## Concat both training sets together -- In case we want to do same data preprocessing 
## Note:  The prepocessing of the testing set is not much of a change of -- just dropping unescesary features
#train_test_users = pd.concat([train_users, test_users], ignore_index=True)
# for testing and analyzing preprocessing functions
#train_test_users_copy1 = train_test_users.copy(deep = True)
## printing the shape of the concactenated pandas object
#print(train_test_users.shape)

## Plotting the percentage of country destination target variable to whole data set 
'''
# Very Important : All plots are available in seperate Python note book for easier output visualization
des_countries = train_users.country_destination.value_counts(dropna = False) / train_users.shape[0] * 100
des_countries.plot('bar', rot = 0)
plt.xlabel('Destination country')
plt.ylabel('Percentage of booking')
plt.show()
'''

#Knowing which columns in  the training and test set separate has null values
print('Train columns with null values:\n', train_users.isnull().sum())
print("-"*10)

print('Test columns with null values:\n', test_users.isnull().sum())
print("-"*10)

#print('TestandTestusers columns with null values:\n', train_test_users.isnull().sum())
#print("-"*10)
# Getting over all idea of the data set describe it !!
#train_users.describe(include = 'all')

#train_test_users.info()

## start correcting the data based on data analyze ... For next step trying to see which features we will include
## Start Look at the age feature --> It includes alot of null values --> see further if it has errors
#print(train_test_users.age.describe())
## oops maximum age is 2014 which is deffinietly wrong maybe the user put his year instead of age
## let get more info on the counts of which user has age above 100 --> hard to travel and access internet if your age older than 100 + there is very few people pass the 100 age in the word
#print("*"*50)
#train_users[train_test_users["age"] > 100].age.value_counts()
## for me see ages count that less than 5 

'''
# Plot for data anaylze only
## Let's try plot the age column to get better visualization
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
ax[0].set_title('Age =< 100')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Count')
train_test_users[train_test_users.age < 100].age.hist(bins = 10, ax = ax[0])

ax[1].set_title('Age > 100')
ax[1].set_xlabel('Age')
ax[1].set_ylabel('Count')
train_test_users[train_test_users.age >= 100].age.hist(bins = 10, ax = ax[1])

print(train_test_users.age.median())
'''
##Helper functions
## Make a function as we will encode alot of categorical data.
label = LabelEncoder()
def encode_features(df, features_to_encode):
    for each_feature in features_to_encode:  
      #print(each_feature)
      df[each_feature] = label.fit_transform(df[each_feature])
    return df

# helper function to deal with dates and split it 
def split_date_ym(df, date_column):
    df[date_column + '_Y'] = df[date_column].dt.year
    df[date_column + '_M'] = df[date_column].dt.month
    df[date_column + '_Q'] = df[date_column].dt.quarter
    df.drop(date_column, axis=1, inplace=True)
    return df    


################### PreProcessing data ####################
# Let's copy our train and test, incase we need to look at original data after processing 

#train_users_copy1 = train_users.copy(deep = True)
#test_users_copy1 =  test_users.copy(deep = True)    


# start with ages:-  From above 
# A- We have values of ages that seem has the year of birth 19xx, 
# B- We have values that are null and,
# C- We have values that are not reasonable i.e 1, 2001

# reformat timestamp_first_active as i need to fix issue number one, subtract when user active against birth year = age
train_users['timestamp_first_active'] = pd.to_datetime(train_users['timestamp_first_active'], format='%Y%m%d%H%M%S')
test_users['timestamp_first_active'] = pd.to_datetime(test_users['timestamp_first_active'], format='%Y%m%d%H%M%S')

# substitute ages bigger than 1999 by null as they are in 2000- Case C - above comments
train_users.loc[train_users.age > 1999, 'age'] = np.nan
test_users.loc[test_users.age > 1999, 'age'] = np.nan
# substiute > 1900 now they will not reach 2000- Case A
train_users.loc[(train_users.age > 1900), 'age'] =  pd.DatetimeIndex(train_users['timestamp_first_active']).year - train_users.age
test_users.loc[(test_users.age > 1900), 'age'] =  pd.DatetimeIndex(test_users['timestamp_first_active']).year - test_users.age
# Now it narrows to CASE 105 102
train_users.loc[train_users.age > 95, 'age'] = np.nan
test_users.loc[test_users.age > 95, 'age'] = np.nan
# case the user enter 1 2 years old
train_users.loc[train_users.age < 15, 'age'] = np.nan
test_users.loc[test_users.age < 15, 'age'] = np.nan
# etc case put all nulls to median since there is alot of outliers
train_users.loc[train_users.age.isnull(), 'age'] = train_users.age.median()
test_users.loc[test_users.age.isnull(), 'age'] = test_users.age.median()

## first_affiliate_tracked -->  contain null data
## convert to untracked
train_users.loc[train_users.first_affiliate_tracked.isnull(), 'first_affiliate_tracked'] = 'untracked'
test_users.loc[test_users.first_affiliate_tracked.isnull(), 'first_affiliate_tracked'] = 'untracked'

# Encode all categorical string featurs using helper function
# Lets now encode all the features 
features_to_encode = ['affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'first_browser', 'first_device_type', 'gender', 'language', 'signup_app', 'signup_flow', 'signup_method']
train_users = encode_features(train_users,features_to_encode)
test_users = encode_features(test_users,features_to_encode)


# Work with date column,  we have timestamp-firstactive ,date_accountcreated,datebooking
# have correct format and convert to datetime
train_users['date_account_created'] = pd.to_datetime(train_users['date_account_created'], format='%Y-%m-%d')
test_users['date_account_created'] = pd.to_datetime(test_users['date_account_created'], format='%Y-%m-%d')

train_users = split_date_ym(train_users,"date_account_created")
test_users = split_date_ym(test_users,"date_account_created")

# Do same for timestamp_first_active
train_users['timestamp_first_active'] = pd.to_datetime(train_users['timestamp_first_active'], format='%Y%m%d%H%M%S')
test_users['timestamp_first_active'] = pd.to_datetime(test_users['timestamp_first_active'], format='%Y%m%d%H%M%S')

train_users = split_date_ym(train_users,"timestamp_first_active")
test_users = split_date_ym(test_users,"timestamp_first_active")

## Let's drop the records for the date_first_booking as it is empty in the test data set
#  Since no country dest (target column) booking
# Drop datefirst booking
train_users.drop("date_first_booking", axis=1, inplace=True)
test_users.drop("date_first_booking", axis=1, inplace=True)

print("Fininsh The user data processing")

'''
print("Fininsh The user data processing")
print("train-user-processed")
print(train_users.head(n=3))
print("train-user-processed---> end")
print("test-user-processed")
print(test_users.head(n=3))
'''
###

## Starting the Session data preprocessing 

# Trying understand the session data -- 

#sessions
print('Working on Session data...')
sessions_path = "../input/sessions.csv"
df_sessions = pd.read_csv(sessions_path)
df_sessions['id'] = df_sessions['user_id']
df_sessions = df_sessions.drop(['user_id'],axis=1)

#########Preparing Session data########

#Filling nan with specific value ('NAN')
df_sessions.action = df_sessions.action.fillna('NAN')
df_sessions.action_type = df_sessions.action_type.fillna('NAN')
df_sessions.action_detail = df_sessions.action_detail.fillna('NAN')
df_sessions.device_type = df_sessions.device_type.fillna('NAN')

#Action values with low frequency are changed to 'OTHER'
act_freq = 100  #Threshold for frequency
act = dict(zip(*np.unique(df_sessions.action, return_counts=True)))
df_sessions.action = df_sessions.action.apply(lambda x: 'OTHER' if act[x] < act_freq else x)

#Computing value_counts. These are going to be used in the one-hot encoding
#based feature generation (following loop).
f_act = df_sessions.action.value_counts().argsort()
f_act_detail = df_sessions.action_detail.value_counts().argsort()
f_act_type = df_sessions.action_type.value_counts().argsort()
f_dev_type = df_sessions.device_type.value_counts().argsort()

#grouping session by id. We will compute features from all rows with the same id.
dgr_sess = df_sessions.groupby(['id'])

#Loop on dgr_sess to create all the features.
samples = []
cont = 0
ln = len(dgr_sess)
for g in dgr_sess:
    if cont%10000 == 0:
        print("%s from %s" %(cont, ln))
    gr = g[1]
    l = []
    
    #the id
    l.append(g[0])
    
    #The actual first feature is the number of values.
    l.append(len(gr))
    
    sev = gr.secs_elapsed.fillna(0).values   #These values are used later.
    
    #action features
    #(how many times each value occurs, numb of unique values, mean and std)
    c_act = [0] * len(f_act)
    for i,v in enumerate(gr.action.values):
        c_act[f_act[v]] += 1
    _, c_act_uqc = np.unique(gr.action.values, return_counts=True)
    c_act += [len(c_act_uqc), np.mean(c_act_uqc), np.std(c_act_uqc)]
    l = l + c_act
    
    #action_detail features
    #(how many times each value occurs, numb of unique values, mean and std)
    c_act_detail = [0] * len(f_act_detail)
    for i,v in enumerate(gr.action_detail.values):
        c_act_detail[f_act_detail[v]] += 1 
    _, c_act_det_uqc = np.unique(gr.action_detail.values, return_counts=True)
    c_act_detail += [len(c_act_det_uqc), np.mean(c_act_det_uqc), np.std(c_act_det_uqc)]
    l = l + c_act_detail
    
    #action_type features
    #(how many times each value occurs, numb of unique values, mean and std
    #+ log of the sum of secs_elapsed for each value)
    l_act_type = [0] * len(f_act_type)
    c_act_type = [0] * len(f_act_type)
    for i,v in enumerate(gr.action_type.values):
        l_act_type[f_act_type[v]] += sev[i]   
        c_act_type[f_act_type[v]] += 1  
    l_act_type = np.log(1 + np.array(l_act_type)).tolist()
    _, c_act_type_uqc = np.unique(gr.action_type.values, return_counts=True)
    c_act_type += [len(c_act_type_uqc), np.mean(c_act_type_uqc), np.std(c_act_type_uqc)]
    l = l + c_act_type + l_act_type    
    
    #device_type features
    #(how many times each value occurs, numb of unique values, mean and std)
    c_dev_type  = [0] * len(f_dev_type)
    for i,v in enumerate(gr.device_type .values):
        c_dev_type[f_dev_type[v]] += 1 
    c_dev_type.append(len(np.unique(gr.device_type.values)))
    _, c_dev_type_uqc = np.unique(gr.device_type.values, return_counts=True)
    c_dev_type += [len(c_dev_type_uqc), np.mean(c_dev_type_uqc), np.std(c_dev_type_uqc)]        
    l = l + c_dev_type    
    
    #secs_elapsed features        
    l_secs = [0] * 5 
    l_log = [0] * 15
    if len(sev) > 0:
        #Simple statistics about the secs_elapsed values.
        l_secs[0] = np.log(1 + np.sum(sev))
        l_secs[1] = np.log(1 + np.mean(sev)) 
        l_secs[2] = np.log(1 + np.std(sev))
        l_secs[3] = np.log(1 + np.median(sev))
        l_secs[4] = l_secs[0] / float(l[1])
        
        #Values are grouped in 15 intervals. Compute the number of values
        #in each interval.
        log_sev = np.log(1 + sev).astype(int)
        l_log = np.bincount(log_sev, minlength=15).tolist()                      
    l = l + l_secs + l_log
    
    #The list l has the feature values of one sample.
    samples.append(l)
    cont += 1

#Creating a dataframe with the computed features    
col_names = []    #name of the columns
for i in range(len(samples[0])-1):
    col_names.append('c_' + str(i)) 
#preparing objects    
samples = np.array(samples)
samp_ar = samples[:, 1:].astype(np.float16)
samp_id = samples[:, 0]   #The first element in obs is the id of the sample.

#creating the dataframe        
df_agg_sess = pd.DataFrame(samp_ar, columns=col_names)
df_agg_sess['id'] = samp_id
df_agg_sess.index = df_agg_sess.id

print("Done with extracting features from Session data")

# getting the target variable country destination

target_variable = train_users['country_destination']
# drop the column 
train_users = train_users.drop(['country_destination'], axis=1)

# wrong but have to continue 
train_test_users = pd.concat([train_users, test_users], ignore_index=True)
df_all = pd.merge(train_test_users, df_agg_sess, how='left')

df_all = df_all.drop(['id'], axis=1)
df_all = df_all.fillna(-2)


######Computing X, y and X_test ################
piv_train = len(target_variable) #Marker to split df_all into train + test
vals = df_all.values
le = LabelEncoder()

X = vals[:piv_train]
y = le.fit_transform(target_variable.values)
X_test = vals[piv_train:]
print('Shape X = %s, Shape X_test = %s'%(X.shape, X_test.shape))


##### Model -  selection and training ###
# ## Train the classifier

def ndcg_score(preds, dtrain):
    labels = dtrain.get_label()
    top = []

    for i in range(preds.shape[0]):
        top.append(np.argsort(preds[i])[::-1][:5])

    mat = np.reshape(np.repeat(labels,np.shape(top)[1]) == np.array(top).ravel(),np.array(top).shape).astype(int)
    score = np.mean(np.sum(mat/np.log2(np.arange(2, mat.shape[1] + 2)),axis = 1))
    return 'ndcg', score

xgtrain = xgb.DMatrix(X, label=y)

param = {
    'max_depth': 10,
    'learning_rate': 1,
    'n_estimators': 5,
    'objective': 'multi:softprob',
    'num_class': 12,
    'gamma': 0,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': 1,
    'base_score': 0.5,
    'missing': None,
    'silent': True,
    'nthread': 4,
    'seed': 42
}

# Do cross validation
num_round = 5
result = xgb.cv(param, xgtrain, num_boost_round=num_round, metrics=['mlogloss'], feval=ndcg_score)
result.to_csv('score_result.csv', index = False)



'''
##NN model
##### Model -  selection and training ###
# ## Train the classifier
# fix random seed for reproducibility
y=  np_utils.to_categorical(y)
train_x, test_x, train_y, test_y = model_selection.train_test_split(X,y,test_size = 0.1, random_state = 0)
#eed = 7
#umpy.random.seed(seed)
model = Sequential()
model.add(Dense(8, input_dim = X.shape[1] , activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(12, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_x, train_y, epochs = 100, batch_size = 100)

scores = model.evaluate(test_x,test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

'''
'''
#another version of xgboost

print("start model training")
XGB_model = xgb.XGBClassifier(objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0, learning_rate=0.1, n_estimators=150)
param_grid = {'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.3], 'n_estimators': [50, 100],'tree_method':['gpu_hist'], 'predictor':['gpu_predictor']}
model = model_selection.GridSearchCV(estimator=XGB_model, param_grid=param_grid, scoring='accuracy', verbose=10, iid=True, refit=True, cv=3)
model.fit(X, y)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
xgb.plot_importance(model)
# ## Store the model and the label encoder in a pickle
pickle.dump(model, open('model.p', 'wb'))
pickle.dump(le, open('labelencoder.p', 'wb'))
'''