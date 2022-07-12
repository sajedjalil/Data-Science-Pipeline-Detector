# Initial script
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import xgboost as xgb

def simple_load(people, train, test):

    # Merge people to the other data sets
    train = train.merge(people, on="people_id", suffixes=("_act", ""))
    test = test.merge(people, on="people_id", suffixes=("_act", ""))

    # Set index to activity id
    train = train.set_index("activity_id")
    test = test.set_index("activity_id")

    # Correct some data types
    for field in ["date_act", "date"]:
        train[field] = pd.to_datetime(train[field])
        test[field] = pd.to_datetime(test[field])

    return train, test


def group_decision(train, test, only_certain=True):
    # Exploit the leak revealed by Loiso and team to try and directly infer any labels that can be inferred
    # https://www.kaggle.com/c/predicting-red-hat-business-value/forums/t/22807/0-987-kernel-now-available-seems-like-leakage

    # Make a lookup dataframe, and copy those in first since we can be sure of them
    lookup = train.groupby(["group_1", "date_act"], as_index=False)["outcome"].mean()
    test = pd.merge(test.reset_index(), lookup, how="left", on=["group_1", "date_act"]).set_index("activity_id")

    # Create some date filling columns that we'll use after we append
    train["date_act_fillfw"] = train["date_act"]
    train["date_act_fillbw"] = train["date_act"]

    # Create some group filling columns for later use
    train["group_fillfw"] = train["group_1"]
    train["group_fillbw"] = train["group_1"]

    # Put the two data sets together and sort
    df = train.append(test)
    df = df.sort_values(by=["group_1", "date_act"])

    # Fill the dates
    df["date_act_fillfw"] = df["date_act_fillfw"].fillna(method="ffill")
    df["date_act_fillbw"] = df["date_act_fillbw"].fillna(method="bfill")

    # Fill labels
    df["outcome_fillfw"] = df["outcome"].fillna(method="ffill")
    df["outcome_fillbw"] = df["outcome"].fillna(method="bfill")

    # Fill the groups
    df["group_fillfw"] = df["group_fillfw"].fillna(method="ffill")
    df["group_fillbw"] = df["group_fillbw"].fillna(method="bfill")

    # Create int booleans for whether the fillers are from the same date
    df["fw_same_date"] = (df["date_act_fillfw"] == df["date_act"]).astype(int)
    df["bw_same_date"] = (df["date_act_fillbw"] == df["date_act"]).astype(int)

    # Create int booleans for whether the fillers are in the same group
    df["fw_same_group"] = (df["group_fillfw"] == df["group_1"]).astype(int)
    df["bw_same_group"] = (df["group_fillbw"] == df["group_1"]).astype(int)

    # Use the filled labels only if the labels were from the same group, unless we're at the end of the group
    df["interfill"] = (df["outcome_fillfw"] *
                       df["fw_same_group"] +
                       df["outcome_fillbw"] *
                       df["bw_same_group"]) / (df["fw_same_group"] +
                                               df["bw_same_group"])

    # If the labels are at the end of the group, cushion by 0.5
    df["needs cushion"] = (df["fw_same_group"] * df["bw_same_group"] - 1).abs()
    df["cushion"] = df["needs cushion"] * df["interfill"] * -0.1 + df["needs cushion"] * 0.05
    df["interfill"] = df["interfill"] + df["cushion"]

    # Fill everything
    df["outcome"] = df["outcome"].fillna(df["interfill"])

    if only_certain == True:
        # Drop anything we're not 100% certain of
        df = df[(df["outcome"] == 0.0) | (df["outcome"] == 1.0)]

    # Return outcomes to the original index
    test["outcome"] = df["outcome"]

    return test


def dummy_var_creator(mat):
    """Used to fill in the dummy variables for each categorical
    variable and create sparse series of the values"""
    return_columns = {}
    for colname in mat.columns.values:
        if colname == 'char_38':
            continue
        unique_vals = mat[colname].unique()
        for v in unique_vals:
            return_columns[colname + '_' + str(v)] = \
                pd.Series(np.where(mat[colname] == v, 1, 0),
                          index=mat.index).to_sparse(fill_value=0)
    return return_columns


def simple_load_no_index_change(people, train, test):

    # Merge people to the other data sets
    train = train.merge(people, on="people_id", suffixes=("_act", ""))
    test = test.merge(people, on="people_id", suffixes=("_act", ""))

    # Correct some data types
    for field in ["date_act", "date"]:
        train[field] = pd.to_datetime(train[field])
        test[field] = pd.to_datetime(test[field])

    return train, test

people = pd.read_csv("../input/people.csv")
orig_train = pd.read_csv("../input/act_train.csv")
orig_test = pd.read_csv("../input/act_test.csv")

train, test = simple_load(people, orig_train, orig_test)

filled_test = group_decision(train, test)

unimputed_rows = pd.isnull(filled_test['outcome']).nonzero()[0]
imputed_rows = pd.notnull(filled_test['outcome']).nonzero()[0]

unknown_test = filled_test.iloc[unimputed_rows]
known_test = filled_test.iloc[imputed_rows]

del filled_test

train, test = simple_load(people, orig_train, orig_test)

del orig_train
del orig_test

train_set = train.append(known_test)
print('Full train shape: ' + str(train_set.shape))

# Sort the columns so they are all lined up
all_cols = train_set.columns.tolist()
unknown_test = unknown_test[all_cols]
train_set = train_set[all_cols]

# Extract the outcomes
outcomes = train_set.iloc[:,52]

# Take out the outcomes column from the train and test set
print('Taking out outcomes...')
test_set = unknown_test.iloc[:,list(range(52)) + [53]]
train_set = train_set.iloc[:,list(range(52)) + [53]]

print('Getting rid of columns...')
# Get rid of columns that shouldn't be included in the final data
useless_columns = [3,49,50,51,52]
train_set = train_set.iloc[:,list(set(range(53)) - set(useless_columns))]
test_set = test_set.iloc[:,list(set(range(53)) - set(useless_columns))]

print('Creating sparse columns...')
# Create dummy variables for each variable
test_cols = dummy_var_creator(test_set)
train_cols = dummy_var_creator(train_set)

# Create data frames of the sparse series - hopefully taking up less memory this way
print('Creating sparse dataframes')
sparse_train = pd.DataFrame(data=train_cols)
sparse_test = pd.DataFrame(data=test_cols)

# Add on the columns that didn't get turned into dummy variables - only char_38 currently
sparse_train['char_38'] = train_set['char_38']
sparse_test['char_38'] = test_set['char_38']

del train_set
del test_set

# Get rid of any columns in the training set that never appear in the test set
columns_to_keep = []
for i in range(len(sparse_train.columns.values)):
    if sparse_train.columns.values[i] in sparse_test.columns.values:
        columns_to_keep.append(i)
sparse_train = sparse_train.iloc[:, columns_to_keep]

print('Creating final dataframes...')
all_cols = sparse_train.columns.tolist()
sparse_train = sparse_train[all_cols]
sparse_test = sparse_test[all_cols]

# Test to make sure the columns are in the same order
for i in range(len(sparse_train.columns.values)):
    if sparse_train.columns.values[i] != sparse_test.columns.values[i]:
        print("Error")

# Test to see if there are missing values
assert(sparse_train.isnull().sum().sum() == 0)
assert(sparse_test.isnull().sum().sum() == 0)

print('Creating models...')
xgb_reg = xgb.XGBRegressor(objective='auc')
AUC = make_scorer(roc_auc_score, greater_is_better=True)
param_grid = {'max_features': ['sqrt', 'log2'], 'colsample_bytree': [0.85],
              'max_depth': [8, 10, 20], 'n_estimators': [2000, 3000],
              'learning_rate': [0.01]}

model = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, n_jobs=4, cv=2, verbose=20, scoring=AUC)
print('About to fit the model....')
model.fit(sparse_train, outcomes)

predicted_outcomes = model.predict(sparse_test)

del sparse_train
del sparse_test

unknown_test['outcome'] = predicted_outcomes
sample_sub = pd.read_csv('../input/sample_submission.csv')

train, test = simple_load_no_index_change(people, orig_train, orig_test)

unknown_test_reset = unknown_test.reset_index()
known_test_reset = known_test.reset_index()

sample_sub_ordering = pd.DataFrame(sample_sub.iloc[:,0])

full_test = known_test_reset.append(unknown_test_reset)

full_test_outcomes = full_test.iloc[:,[0,54]]

submission_output = sample_sub_ordering.merge(full_test_outcomes, on='activity_id')

submission_output.to_csv('xgboost_submission1.csv', index=False)