import pandas as pd
import csv as csv
from sklearn.naive_bayes import GaussianNB

#################################################################
#
# Some snippets were copied from TomaszGrel's script...
# (mainly writing to csv and result computation)
#
#################################################################

# Number of data records to analyse
num_values = 5000

# Read the training data in using pandas read_json and create an
# empty df to unpack it in
raw_df_train = pd.read_json('../input/train.json')[:num_values]
df_train = pd.DataFrame(columns=['id', 'cuisine', 'ingredients', 'val'])

# Loop over each row and so that we can unpack each ingredient onto a
# separate row (so we can pivot them out in a minute.
# Tried to this with apply but failed!
for idx, row in raw_df_train.iterrows():
    ingredients = row['ingredients']
    d = {'id': pd.Series([row['id']] * len(ingredients)),
         'cuisine': pd.Series([row['cuisine']] * len(ingredients)),
         'ingredients': pd.Series(ingredients),
         'val': pd.Series([1] * len(ingredients))}
    df_train = pd.concat([df_train, pd.DataFrame(d)])

# Pivot the ingredients to their own column, here a 1 indicates that
# the recipe has that particular ingredient, and a 0 indicates otherwise
df_train = pd.pivot_table(df_train,
                          index=['id', 'cuisine'],
                          columns='ingredients',
                          values='val',
                          fill_value=0)

# Extract the cuisines, this is the data we're trying to guess
classes_train = list(df_train.index.get_level_values('cuisine'))
matrix_train = df_train.as_matrix()
# Set up our Naive Bayes algo and fit the data
gnb = GaussianNB()
gnb.fit(matrix_train, classes_train)

# Compute accuracy on training set
result_train = [(ref == res, ref, res) for (ref, res) in zip(classes_train, gnb.predict(matrix_train))]
accuracy_train = sum(r[0] for r in result_train) / float(len(result_train))

print('Training data accuracy: ' + str(accuracy_train) + '%')

# Now do the same with the test data as we did with the training data
raw_df_test = pd.read_json('../input/test.json')[:num_values]
df_test = pd.DataFrame(columns=['id', 'ingredients', 'val'])

for idx, row in raw_df_test.iterrows():
    ingredients = row['ingredients']
    d = {'id': pd.Series([row['id']] * len(ingredients)),
         'ingredients': pd.Series(ingredients),
         'val': pd.Series([1] * len(ingredients))}
    df_test = pd.concat([df_test, pd.DataFrame(d)])

df_test = pd.pivot_table(df_test,
                         index=['id'],
                         columns='ingredients',
                         values='val',
                         fill_value=0)

test_total_data = df_test.as_matrix().sum()

index_test = list(map(int, df_test.index.get_level_values('id')))
df_train = df_train.set_index(df_test.index)

# We need the test data to have the same columns as the training data set
# Obviously this might not be the case as there are different sets of
# ingredients in both (although) there is major overlap.
# Therefore I have basically copied the test data over the training data
# frame, and zeroed columns that don't exist in the test dataset.
# This is a bit of a munge and will skew the results but I'm hoping that
# there isn't too much feature loss
for col in df_train.columns.values:
    if col in df_test.columns.values:
        df_train[col] = df_test[col]
    else:
        df_train[col] = df_train[col].apply(lambda x: 0)

matrix_test = df_train.as_matrix()

# Quick test to see if we are losing much data when copying to the training dataframe
test_modified_data = matrix_test.sum()
pct_loss = (test_total_data - test_modified_data) / float(test_total_data) * 100
print('Loss of test data when munging to train format: ' + str(pct_loss) + '%')

# Run the gnb predict on the test data and print out the results!
result_test = gnb.predict(matrix_test)
ids = index_test
result_dict = dict(zip(ids, result_test))

writer = csv.writer(open('submission.csv', 'wt'))
writer.writerow(['id', 'cuisine'])
for key, value in result_dict.items():
    writer.writerow([key, value])

print('Result saved in file: submission.csv')
