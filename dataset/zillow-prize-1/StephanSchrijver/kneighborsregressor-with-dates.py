import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Load trainingset and properites
print('Loading train 2016 and properties CSV ...')
train = pd.read_csv('../input/train_2016_v2.csv', parse_dates=["transactiondate"])
prop = pd.read_csv('../input/properties_2016.csv')

# True/False fillna with False
prop["hashottuborspa"].fillna(False, inplace=True)
prop["fireplaceflag"].fillna(False, inplace=True)

# True/False cols to numbers 1/0
prop["num_hashottuborspa"] = prop["hashottuborspa"].astype(int)
prop["num_fireplaceflag"] = prop["fireplaceflag"].astype(int)


# Merge property information in training so we have
# all information needed
training_set = train.merge(prop, on="parcelid", how="left")

# Training is now merged into training_set so it can be deleted
del train

# Adjust features
training_set["transaction_year"] = training_set["transactiondate"].dt.year
training_set["transaction_month"] = training_set["transactiondate"].dt.month


# Drop all features which are not floats
x_train = training_set.drop(["parcelid","logerror","hashottuborspa", "propertyzoningdesc","fireplaceflag","taxdelinquencyflag", "propertycountylandusecode", "transactiondate"], axis=1)

# Fill all NaN's with zero
x_train = x_train.fillna(0)

# Y-axis values are log error
y_train = training_set["logerror"].values

# Create regressor
reg = KNeighborsRegressor(n_neighbors = 100)
print("Fit data")
reg.fit(x_train, y_train)
print("Done fitting")

# Save used featers so we can use these columns later
used_features = x_train.columns

# x_train and y_train aren't used anymore so we can delete them safely
del x_train, y_train

# Read submission files and merge them with the house features
submission_df = pd.read_csv('../input/sample_submission.csv')
submission_df['parcelid'] = submission_df['ParcelId']
submission_with_features = submission_df.merge(prop, on="parcelid", how="left")

# Submission dataframe is merged into submissions_with_features so it can be deleted
del submission_df

submission_df = pd.read_csv('../input/sample_submission.csv')
for c in submission_df.columns[submission_df.columns != 'ParcelId']:
    date = pd.to_datetime(str(c), format='%Y%m')
    submission_with_features["transaction_year"] = date.year
    submission_with_features["transaction_month"] = date.month

    # Only use the features that are used with the trainingset
    x_pred = submission_with_features[used_features]
    x_pred = x_pred.fillna(0)

    print("Predicting for date", c)
    pred = reg.predict(x_pred)
    print("Done predicting for date", c)
    submission_df[c] = pred

# Now submission_with_features is not used anymore, delete
del submission_with_features

print("Begin write")
submission_df.to_csv('KNeighbours_with_date.csv', index=False, float_format='%.5g')
print("End write")