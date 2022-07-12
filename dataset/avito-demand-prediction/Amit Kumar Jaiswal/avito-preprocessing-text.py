import pandas as pd

print("\nData Load Stage")
train_df = pd.read_csv('../input/train.csv', index_col='item_id')
traindex = train_df.index
test_df = pd.read_csv('../input/test.csv', index_col="item_id")
testdex = test_df.index

cols = "title"

print("Combine Train and Test")
train_df = train_df[[cols]].copy()
test_df = test_df[[cols]].copy()

train_df[cols] = train_df[cols].astype(str)
train_df[cols] = train_df[cols].astype(str).fillna('missing_text')  # FILL NA
train_df[cols] = train_df[cols].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently

test_df[cols] = test_df[cols].astype(str)
test_df[cols] = test_df[cols].astype(str).fillna('missing_text')  # FILL NA
test_df[cols] = test_df[cols].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently

train_df.to_csv('train_title_features.csv')
test_df.to_csv('test_title_features.csv')