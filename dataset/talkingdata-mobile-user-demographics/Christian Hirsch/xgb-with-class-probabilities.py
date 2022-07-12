# #Feature Engineering -- Brand and device model
# In the first round of feature engineering, we extract features from the phone brand and the device model. The most immediate is to label-encode these two columns. As an additional feature, we add the **regularized class histograms** proposed in [https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/brand-and-model-based-benchmarks]([https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/brand-and-model-based-benchmarks). The analysis of the feature importances in the xgb scores later will show that these crosstab-features indeed have strong predictive capabilities.
import numpy as np
import math
import operator
import matplotlib.pylab as plt
import pickle 
import pandas as pd

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost.sklearn import XGBClassifier

#path to data
DATA_PATH = "../input/"

#parameter for crosstab feature; minimum number of observations per bin
MIN_COUNT = 5

#small value to avoid log(0)
EPS = 0.001

#seed for randomness
SEED = 1747

#score function of xgb classifier
SCORING = 'mlogloss'

#params for xgb
HYPER_PARAMS = {
 'learning_rate': 0.05,
 'n_estimators': 150,
 'max_depth': 6,
 'subsample': 0.8,
 'colsample_bytree': 0.8,
 'max_delta_step': 1,
 'objective': 'multi:softmax',
 'nthread': 2,
 'seed': SEED,
  'missing': np.nan
}
# Next, we load train, test and merge with the phone data.
train = pd.read_csv('{0}gender_age_train.csv'.format(DATA_PATH)).loc[:, ['device_id', 'group']]
test = pd.read_csv('{0}gender_age_test.csv'.format(DATA_PATH))['device_id']
phone_brand = pd.read_csv('{0}phone_brand_device_model.csv'.format(DATA_PATH))

train_test = pd.concat([train['device_id'], test])
train_test_phone_raw = pd.merge(train_test.to_frame(), phone_brand, 'left', on = 'device_id').drop_duplicates()
# A very small number of devices are associated to several phones. We only pick the first of them.
train_test_phone = train_test_phone_raw.groupby('device_id', sort = False).first()
# ##Phone and device model
# The phone and brand model are features that are readily accessible. We perform a one-hot encoding on them.
phone_device_ohe = OneHotEncoder().fit_transform(train_test_phone.apply(lambda col: LabelEncoder().fit_transform(col)))
# ##Crosstab encoder
# In addition to plain ohe, we implement an idea from https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/brand-and-model-based-benchmarks, which suggests to characterize a brand or device model by its gender-age histogram. In order to obtain features that are not constraint to [0,1], we consider the logs of the class probabilities.
class CrossTabEncoder(BaseEstimator, TransformerMixin):
    """CrossTabEncoder
    A CrossTabEncoder characterizes a feature by its crosstab dataframe.
    
    Parameters
    ----------
    feature_name: name of the feature to be encoded
    min_count: minimum number of observations per bin required to count that bin
    """
    def __init__(self,  min_count):     
        self.crosstab = None
        self.min_count = min_count

    def fit(self, features, classes, y = None):
        """For each class of the considered feature, the empirical histogram for the prediction classes is computed. 
        
        Parameters
        ----------
        features : feature column used for the histogram computation
        classes : class column used for the histogram computation        
        """        
        raw_hist = pd.crosstab(features.iloc[:classes.shape[0]], classes)
        self.crosstab = raw_hist.apply(lambda row: compute_log_probs(row, self.min_count), axis = 1)           
        return self

    def transform(self, data):
        """The precomputed histograms are joined as features to the given data set.
        
        Parameters
        ----------
        X : array-like object
        
        Returns
        -------
        Transformed dataset.
        """
        return pd.merge(data.to_frame(), self.crosstab, 'left', 
                        left_on = data.name, right_index = True).drop(data.name, axis =1)    
    
def compute_log_probs(row, min_count):    
    """helper function for computing regularized log probabilities
    """
    row = row.apply(lambda x: max(x, min_count))
        
    #compute the log ratios of class probabilities and the popularity of the feature 
    row_sum = row.sum()
    row = (row/row_sum).apply(lambda y: math.log(y) - math.log(min_count/row_sum))
    row['popularity'] = row_sum
    
    return row
# We apply the crosstab encoder to the phone_brand and device_model features.
phone_cte = CrossTabEncoder(5).fit(train_test_phone['phone_brand'], train.set_index('device_id')['group'])
device_cte = CrossTabEncoder(5).fit(train_test_phone['device_model'], train.set_index('device_id')['group'])

phone_crosstab = phone_cte.transform(train_test_phone['phone_brand'])
device_crosstab = device_cte.transform(train_test_phone['device_model'])
# Finally, we collect all features.
phone_device_features = sparse.csr_matrix(sparse.hstack([phone_crosstab, device_crosstab, phone_device_ohe]))
# #Feature Engineering -- events data
# In the second round of feature engineering, we construct features from the 'events' data. This will result in features involving the time step and the location of the event.
# Next, we load the events data and merge with the train-test data.
events = pd.read_csv('{0}events.csv'.format(DATA_PATH))
train_test_events = pd.merge(train_test.to_frame(), events, 
                             'left', on = 'device_id').drop_duplicates().set_index('device_id')
# ##Event count 
# To begin with, we simply count the number of events per device id.
event_cnt = train_test_events.groupby(level = 0, sort = False)['event_id'].nunique()
# ##Time features 
# For each device id we are given a possibly empty set of timestamps when that device triggered an event. For each device id, we compute the crosstab of the hours at which an event was triggered. This provides us with a collection of 24 features, one for each hour.
hours = pd.to_datetime(train_test_events['timestamp']).dt.hour
hours_hist_raw = hours.to_frame().reset_index().pivot_table(index = 'device_id', columns = 'timestamp', 
                                                            aggfunc = len,  margins = True)
hours_hist = pd.merge(train_test.to_frame(),hours_hist_raw,'left',
                      left_on = 'device_id', right_index = True).set_index('device_id')
# To make the features comparable, we compute the log-likelihoods. 
hours_loglik = hours_hist.apply(lambda row: 
                             (row / row['All']).apply(lambda cell: math.log(cell/EPS + 1)), 
                             axis = 1).drop('All', axis = 1)
# ##Location features
# As first feature, we record the number of distinct locations.
loc_cnt = train_test_events.groupby(level = 0, sort = False)['latitude'].nunique()
# The second type of location-based feature is the average of the latitudes and longitudes. As a preprocessing step, we remove outliers.
filtered_lat_lon = train_test_events[abs(train_test_events['latitude'])>1].loc[:, ['latitude', 'longitude']]
mean_loc_raw = filtered_lat_lon.groupby(level = 0).aggregate('mean')
mean_loc = pd.merge(train_test.to_frame(), mean_loc_raw,'left',
                      left_on = 'device_id', right_index = True).set_index('device_id')
# Finally, we collect the event features
event_features = sparse.csr_matrix(pd.concat([event_cnt.to_frame(), hours_loglik, loc_cnt.to_frame(), mean_loc], axis = 1))
# ##TalkingData -- App data
# In the third round of feature engineering, we construct features on app usage. We load the app_events data and merge it with train_test_events. 
apps = pd.read_csv("{0}app_events.csv".format(DATA_PATH))
app_labels = pd.read_csv("{0}app_labels.csv".format(DATA_PATH))
train_test_app = pd.merge(train_test_events.reset_index().loc[:, ['device_id', 'event_id']], 
                          apps.loc[:, ['event_id', 'app_id']], 
                          'left', on = 'event_id').loc[:,['device_id', 'app_id']].drop_duplicates()
train_test_label = pd.merge(train_test_app, 
                            app_labels, 
                            'left', on = 'app_id').loc[:,['device_id', 'label_id']].drop_duplicates()
# Next, we introduce a function to create a bag of features...
def make_bag(data, index, feature_name):
    bag_of_features = data.groupby(index, as_index = False, 
                                   sort = False).aggregate(lambda x:  ' '.join(list(map(str, (list(x))))))
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,  preprocessor = None,    stop_words = None)
    return vectorizer.fit_transform(bag_of_features[feature_name])
# ... and apply it to the app and label data.
bag_of_apps = make_bag(train_test_app , 'device_id', 'app_id')
bag_of_labels = make_bag(train_test_label , 'device_id', 'label_id')
# Finally, we collect all features that we have constructed into a single sparse matrix
train_size = train.shape[0]
features = sparse.hstack([phone_device_features[:train_size, :], event_features[:train_size, :],
                            bag_of_apps[:train_size, :], bag_of_labels[:train_size, :]])
train_size = train.shape[0]
features = sparse.hstack([phone_device_features[:train_size, :], event_features[:train_size, :],
                            bag_of_apps[:train_size, :], bag_of_labels[:train_size, :]])
# #Feature Importance
# As a final step, we investigate the importance of the generated features using xgb feature scores.
X_train, X_val, y_train, y_val = train_test_split(features, 
                                                  LabelEncoder().fit_transform(train['group']),
                                                  train_size = 0.8, random_state = SEED)
xgb = XGBClassifier(**HYPER_PARAMS)
xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], eval_metric = SCORING, verbose = 10)
# Due to the large number of features (19861), we use the importance scores to pick out the relevant ones.
gbdt = xgb.booster()
importance = gbdt.get_fscore()
importance = sorted(importance.items(), key = operator.itemgetter(1), reverse = True)
df=pd.DataFrame(importance, columns = ['feature', 'fscore']).set_index('feature')
print(df.iloc[0:10])

fig, ax = plt.subplots(1, 1)
ax.get_xaxis().set_visible(False)  
df.iloc[0:100,:].plot.bar(ax=ax)
# The indexes of the most important features have the following interpretation 
# 
# - the most important feature is associated to a specific app id
# - f1756 corresponds to the number of events registered at the device
# - f13, f19 and f24 are crosstab-features associated with devices