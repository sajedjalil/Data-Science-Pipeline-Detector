# imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, classification_report
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import gc


# set seed and some constants
np.random.seed(1234)
input_dir = '../input/' # change when running on collab or elsewhere
chunksize = 500000  # num of rows to read from test file at once
train_sample=False   # whether to sample train data. Set to false to use all data
N = 100000   # no of samples to take from training data.


# feature engineering
def make_interactions(dataframe):
    poly = PolynomialFeatures(degree = 2, interaction_only = True, include_bias=False)
    result = poly.fit_transform(dataframe)[...,len(dataframe.columns):]
    return pd.DataFrame(result,
    columns=poly.get_feature_names(dataframe.columns)[-result.shape[1]:],
    index=dataframe.index)


def feature_engg(df):
    # make copy
    dataframe = pd.DataFrame(df)
    # drop experiment, time and seat and id(in test)
    for i in {'experiment','time','seat','id'}:
        if i in dataframe.columns:
            dataframe = dataframe.drop([i], axis = 1)
    
    phi_df = dataframe[['crew', 'ecg', 'r', 'gsr']].copy()
    if 'event' in dataframe.columns:
        phi_df = phi_df.join(dataframe['event'])
    
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_fp1','eeg_fp2','eeg_fz']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_f3','eeg_f4','eeg_f7','eeg_f8', 'eeg_fz']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_t3','eeg_t4','eeg_t5','eeg_t6']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_c3','eeg_c4','eeg_cz']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_p3','eeg_p4','eeg_poz','eeg_pz']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_o1','eeg_o2']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_pz','eeg_poz','eeg_cz','eeg_fz']]))
    phi_df = phi_df.join(make_interactions(dataframe[['crew', 'ecg']]))
    phi_df = phi_df.join(make_interactions(dataframe[['crew', 'r']]))
    phi_df = phi_df.join(make_interactions(dataframe[['crew', 'gsr']]))
    phi_df = phi_df.join(make_interactions(dataframe[['ecg', 'gsr']]))
    return phi_df
    

# feature importances
def feature_importance(forest, X):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(10,10))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices],rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.show()
    

# read training data
print("reading training data")
df = pd.read_csv(input_dir + "train.csv", dtype={'event':'category'})
df.head()


# apply feature engineering
print("modifying training data")
df = feature_engg(df)


# Get a smaller data sample
if train_sample:
    print("sampling data")
    rows = np.random.choice(df.index.values, N)
    sampled_df = df.iloc[rows]
    del df
    gc.collect()
else:
    sampled_df = df
    
sampled_df.shape


# create train-test split
print("splitting data")
X = sampled_df.drop('event', axis = 1)
y = sampled_df['event']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


# train
print("training model")
model = RandomForestClassifier(n_estimators=250, oob_score=True,
                               min_samples_leaf = 100, max_features=3,
                               n_jobs=-1, class_weight='balanced')

model.fit(X_train,y_train)
pred_train = model.predict_proba(X_train)
pred_test = model.predict_proba(X_test)

# print scores
print(model.score(X_train,y_train), model.oob_score_,
      log_loss(y_train,pred_train),model.score(X_test,y_test),
      log_loss(y_test,pred_test))
print(classification_report(y_test, model.predict(X_test)))

# print feature importance
feature_importance(model, X_train)

# save the model to disk (don't ask why)
print("saving model")
filename = 'model.sav'
joblib.dump(model, filename)


# predict, and hope it works!
prediction = None

# read test data
print("reading test data and predicting in batches")
iterator = pd.read_csv(input_dir + "test.csv", chunksize=chunksize)
for test in iterator:
    moddf = test.copy()
    # process
    moddf = feature_engg(moddf)
    # predict
    arr = model.predict_proba(moddf)
    if prediction is None:
        prediction = arr
    else:
        prediction = np.append(prediction, arr, axis = 0)
    
# save
pd.DataFrame(prediction).to_csv("submission.csv",index=True, index_label='id', header=model.classes_)