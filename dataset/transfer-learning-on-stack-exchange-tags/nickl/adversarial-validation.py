# This is an implementation of Adversarial validation as described in
#   http://fastml.com/adversarial-validation-part-one/ and
#   http://fastml.com/adversarial-validation-part-two/
#
# All errors are mine!
#
# The basic idea is:
#   We have labelled training data
#   We have unlabelled test data
#   We'd like a lablled validation dataset
#
# So what we do is we train a classifier which can distingish data from the
# test set and training set, and then choose the data from the train set it
# misclassifies and use it as the validation set.
#
# The idea here is that this set of data must be similar to the test set in
# some way, and it is labelled data!
#
# In this particular case I haven't found it very helpful. The text of the
# questions is pretty similar to physics type questions, but the tags are so
# dramatically different that I haven't found any algorithm which behaves
# similarly on this validation set compared to the leaderboard.
#
# Feedback would be very welcome! Also, I don't really know Python, so...
#
#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline



print("Loading Data...")

# Load all the training data and combine them into a single DataFrame
df = pd.DataFrame()
for name in ['biology.csv', 'cooking.csv', 'crypto.csv', 'diy.csv', 'robotics.csv', 'travel.csv']:
    df_read = pd.read_csv("../input/{0}".format(name), quotechar='"')
    df = df.append(df_read)

# Now load the test set
df_test = pd.read_csv("../input/test.csv".format(name), quotechar='"')
df_test['tags'] = np.nan # set the test set tags column to NaN

# Append the test set to the train set
df = df.append(df_test)

# reset the index. Otherwise we get duplicate rows
df.reset_index(inplace=True)

print("Data Loaded. Shape is ", df.shape)

# label the data by if they have tags or not.
# Not entirely sure why I need to create the column using NaN first?
df['is_train'] = np.nan
df['is_train'] = pd.notnull(df.tags)



def clean_html(raw_html):
    if len(raw_html) > 0:
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext
    else:
        return ''


def process(row):
    cleanbody = clean_html(row['content'])
    text = row['title'] + " " + cleanbody
    return text

print('Cleaning Text Data...')

df['text'] = df.apply(process, axis=1)

print("Training Classifier...")

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', BernoulliNB()),
])

text_clf = text_clf.fit(df.text, df.is_train)

print("Classifier Trainied")

print("Using model to predict against training data...")

predicted = text_clf.predict(df.text)

print("Classifier accuracy on train set:", np.mean(predicted == df.is_train))

# Turn the predictions into a DataFrame
predicted_df = pd.DataFrame.from_dict(predicted)
predicted_df.columns = ['prediction']

# Join the predictions back to the original data frame
df_combined = pd.concat([df, predicted_df], axis=1)

df_combined_train = df_combined[df_combined.is_train == True]

misclassified_df = df_combined_train[df_combined_train.is_train != df_combined_train.prediction]
print("Number misclassified records: ", len(misclassified_df))

print("These misclassified records are the data set we want to use as training data, since they appear similar to the test set")

# Save the data
misclassified_df[["id", "title", "content", "tags"]].to_csv("output.csv", header=True, index=False)
print("Data saved to output.csv")
