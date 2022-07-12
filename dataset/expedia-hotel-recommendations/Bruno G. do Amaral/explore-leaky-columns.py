import numpy as np
import pandas as pd

# Definitions
leaky_cols = ['user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance']
target_col = 'hotel_cluster'
id_col = 'id'
train_rows =   37670294 # Using wc -l train.csv
rows_to_train = 1000000 # Rows to train

# Load train
print("Reading train...")
train = pd.read_csv("../input/train.csv", usecols=leaky_cols + [target_col], skiprows=range(1, train_rows-rows_to_train)).dropna()

# Will use to fill nans in test
most_populars = train[target_col].value_counts()[:5].index

# Find the most common hotel for each leaky_col
print("Grouping leaky columns...")
model = train.groupby(leaky_cols).agg(lambda x: x.value_counts().index[0])
del train

# Apply to test
print("Reading test...")
test = pd.read_csv("../input/test.csv", usecols=leaky_cols + [id_col])

print("Joining model...")
result = test.join(model.reset_index(), how='left', rsuffix='train_')[[id_col, target_col]]
del test

print("Creating out-of-model predictions...")
# Create standard prediction for last 4 predictions
last4 = " " + " ".join([str(i) for i in most_populars[1:]])

# Fill nan with most popular and apply last 4 to each prediction
result[target_col].fillna(most_populars[0], inplace=True)
result[target_col] = result[target_col].apply(lambda x: str(x.astype(int)) + last4)

# Save results
print("Saving results...")
result.to_csv('submission.csv', header=True, index=False)

print("Done")