import numpy as np
import pandas as pd
from pandas import DataFrame

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

colNames = list(train.columns)

# Separate the data and labels.
X_train_df = train.iloc[:, 2:len(colNames)]
X_train_df["ID"] = DataFrame(train["ID"])
X_test_df = test.iloc[:,1:len(colNames)]
X_test_df["ID"] = DataFrame(test["ID"])
y = train.iloc[:,1]

# Combine train and test for preprocessing
train_and_test = [X_train_df, X_test_df]
X_combined = pd.concat(train_and_test, keys = ['train', 'test'])


# Replace missing values of numerical attributes with MEAN of their column (Possibly not very effective)
for i in range(X_combined.shape[1]):
    if X_combined.iloc[:,i].dtype == 'float64':
        X_combined.iloc[:,i].fillna(np.mean(X_combined.iloc[:,i]), inplace = True)

# Replace missing values of categorical attributes with MODE of their column 
for i in range(X_test_df.shape[1]):
    if X_combined.iloc[:,i].dtype == 'O':
        X_combined.iloc[:,i].fillna(max(dict(X_combined.iloc[:,i].value_counts())), inplace = True)
        
# Remove these categoricals
removedCategoricals = ['v3', 'v22', 'v30', 'v31', 'v52', 'v56', 'v91', 'v107', 'v112', 'v113', 'v123']
X_combined_clean = X_combined.drop(removedCategoricals, axis = 1)

# Convert strings of categorical columns to integers
categoricalColumnNames = list(X_combined_clean.select_dtypes(["O"]).columns)
categoricalColumnIndeces = []
for i in range(len(categoricalColumnNames)):
    categoricalColumnIndeces.append(X_combined_clean.columns.get_loc(categoricalColumnNames[i]))

for i in categoricalColumnIndeces:
    class_mapping = {label: index for index, label in enumerate(np.unique(X_combined_clean.iloc[:,i]))}
    X_combined_clean.iloc[:,i] = X_combined_clean.iloc[:,i].map(class_mapping)
  
# All categorical variables are nominal, so I am splitting
# all of them into dummy variables using One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = categoricalColumnIndeces, sparse = False)
X_combined_processed = ohe.fit_transform(X_combined_clean)

# Split back into Training and Test sets
X_processed_df = DataFrame(X_combined_processed)
X_train_processed = X_processed_df.iloc[:len(train),:]
X_test_processed = X_processed_df.iloc[len(train):,:]


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 600, random_state = 1, n_jobs = 2)
rf.fit(X_train_processed, y)

predictions = rf.predict_proba(X_test_processed)
predictions = DataFrame(predictions[:,1])
submission = DataFrame(test["ID"])
submission["PredictedProb"] = predictions
submission.to_csv("submission.csv", index = False)
X_test_processed = X_processed_df.iloc[len(train):,:]