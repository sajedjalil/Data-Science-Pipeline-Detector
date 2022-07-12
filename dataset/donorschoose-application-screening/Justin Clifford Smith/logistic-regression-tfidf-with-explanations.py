"""
The kernel contains the details of my tuned logistic regression solution with 
explanation about my various choices to help out beginners. 
For more on my data analysis please see my other Kernel for this challenge.
"""

# load in everything I plan to use.
import numpy as np 
import pandas as pd
#import re
#from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale,LabelBinarizer, StandardScaler

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
resources = pd.read_csv('../input/resources.csv')

# I'm fond of printing shape just to make sure that I have a sense of the data I'm loading in.
print(train.shape)
print(test.shape)
print(resources.shape)

# I want to create a new feature that is resource*price to get a cost.
# Then I want to combine all the costs to get a total cost.
id_cost = pd.DataFrame({'id': resources['id'], 'total_cost': resources['quantity'] * resources['price']})
id_total_cost = id_cost.groupby(id_cost['id'], sort=False).sum().reset_index()

# And now we'll append the total cost to our train and test data.
train_resources = pd.merge(train, id_total_cost, on='id', sort=False)
test_resources = pd.merge(test, id_total_cost, on='id', sort=False)

# I was struggling withe memory issues on my laptop first, so I was being
# particularly diligent about cleaning up files I was done with.
del train
del test
del resources
del id_cost
del id_total_cost

print(train_resources.shape)
print(test_resources.shape)

# Time to preprocess all the data.
print('==== Beginning preprocessing of the data ====')

# Scale all the numeric data so it between 0 and 1.
scaler = StandardScaler()
numeric_labels = ['teacher_number_of_previously_posted_projects', 'total_cost']
for label in numeric_labels:
    train_resources[label] = scaler.fit_transform(train_resources[label].astype(np.float64).values.reshape(-1, 1))
    test_resources[label] = scaler.transform(test_resources[label].astype(np.float64).values.reshape(-1, 1))

# Encode all the category labels as numbers.
category_labels = ['school_state', 'project_grade_category']
for label in category_labels:
    lb = LabelBinarizer()
    train_resources[label] = lb.fit_transform(train_resources[label])
    test_resources[label] = lb.transform(test_resources[label]) 

# Create a new feature which is the combination of categories and subcategories
# Note the space is in the quotes of the apply function, otherwise you'll create
# bogus words as the categories will run up into the subcategories.
train_resources['subjects'] = train_resources[['project_subject_categories', 'project_subject_subcategories']].apply(lambda x: ' '.join(x), axis=1)
test_resources['subjects'] = test_resources[['project_subject_categories', 'project_subject_subcategories']].apply(lambda x: ' '.join(x), axis=1)

# I've included the code that one would use if you wanted to stem words
# However I saw that this decreased my accuracy.
#ps = PorterStemmer()
#def wordPreProcess(sentence):
#    return ' '.join([ps.stem(x.lower()) for x in re.split('\W', sentence) if len(x) >= 1])

# This is the vectorizer for the subject labels.
subject_vectorizer = TfidfVectorizer(
        analyzer='word', # word was better than char
        norm='l2', # l2 was much better than l1
        token_pattern=r'\w{1,}', # This seems the most common and most sensible
        strip_accents='unicode',  # I didn't play with this.
        stop_words='english', # This had a small positive effect.
        #preprocessor=wordPreProcess, # as noted above, stemming did not help.
        max_df=1.0, # Anything less than 1.0 decreased accuracy
        min_df=0, # Smaller is better
        lowercase=True, # Better than false
        sublinear_tf=False, # Better than true
        ngram_range=(1,1), # This was optimal, likely due to the small number of words.
        max_features=15000) # Generally bigger is better to a point.
print('Fitting and transforming train subjects')
subject_vec_output_train = subject_vectorizer.fit_transform(train_resources['subjects'])
print('Transforming test subjects')
subject_vec_output_test = subject_vectorizer.transform(test_resources['subjects'])

# Fil all the NaN's in essays 3 and 4.
train_resources['project_essay_3'] = train_resources['project_essay_3'].fillna('') 
train_resources['project_essay_4'] = train_resources['project_essay_4'].fillna('')
test_resources['project_essay_3'] = test_resources['project_essay_3'].fillna('') 
test_resources['project_essay_4'] = test_resources['project_essay_4'].fillna('')  

# And combine essays just as a bove.
train_resources['essays'] = train_resources[['project_essay_1', 'project_essay_2','project_essay_3','project_essay_4']].apply(lambda x: ' '.join(x), axis=1)
test_resources['essays'] = test_resources[['project_essay_1', 'project_essay_2','project_essay_3','project_essay_4']].apply(lambda x: ' '.join(x), axis=1)

# Do a bit more cleaning.
for label in ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']:
    del train_resources[label]
    del test_resources[label]

# Vectorize the essays.
word_vectorizer = TfidfVectorizer(
        analyzer='word', 
        norm='l2', 
        token_pattern=r'\w{1,}',
        strip_accents='unicode', 
        #stop_words='english', # Unlike above, this had a negative effect!
        #preprocessor=wordPreProcess, # Surprisingly, stemming hurt here too.
        max_df=1.0, 
        min_df=0, 
        lowercase=True, 
        sublinear_tf=False,
        ngram_range=(1,2), # Better than (1,1) unlike above.
        max_features=15000) 
print('Fitting and transforming train')
word_vec_output_train = word_vectorizer.fit_transform(train_resources['essays'])
del train_resources['essays']
print('Transforming test')
word_vec_output_test = word_vectorizer.transform(test_resources['essays'])
del test_resources['essays']

print('==== Word vectorization complete ====')

# Creating lists of the dataframe labels I want to use.
useful_data_train = []
useful_data_test = []
for label in numeric_labels + category_labels:
    useful_data_train.append(train_resources[label])
    useful_data_test.append(test_resources[label])

print('==== Forming y ====')
# Based off the methods I was using (and testing, though not shown) it was easier to be a sparse matrix.
y = train_resources['project_is_approved'].to_sparse().as_matrix()
del train_resources

from scipy.sparse import hstack

print('==== Forming X ====')
X = pd.concat(useful_data_train, axis=1).to_sparse().as_matrix()
del useful_data_train
# hstack is your friend!!!
# pd.concat is very memory intensive and using it here instead of hstack
# was causing my computer to crash a lot.
X = hstack((X, word_vec_output_train, subject_vec_output_train))
print('==== Forming Xtest ====')
Xtest = pd.concat(useful_data_test, axis=1).to_sparse().as_matrix()
del useful_data_test
Xtest = hstack((Xtest, word_vec_output_test, subject_vec_output_test))

print('==== Ending preprocessing of the data ====')

# Here is the code I used for my simple grid search.
"""
from sklearn.model_selection import GridSearchCV
parameters = [{'penalty':['l1'],'solver':['liblinear', 'saga'], 'C':[ .1, .3, 1, 3]},
               {'penalty':['l2'],'solver':['liblinear', 'newton-cg', 'lbfgs', 'sag'], 'C':[ .1, .3, 1, 3]}]
lg = LogisticRegression()
clf = GridSearchCV(lg, parameters, scoring='roc_auc', verbose=1)
clf.fit(X, y)
print("Best parameters set found on training data:")
print(clf.best_params_)
print("Best score found using these parameters:")
print(clf.best_score_)
"""

# If I was really fancy, I'm sure I could output the parameters of my
# grid search directly into the defintion of my classifier.
print('=== Start cross-validation ====')
clf = LogisticRegression(C=1.0, 
                         penalty='l1', 
                         solver='liblinear',
                         max_iter=500,
                         n_jobs=1)
scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc', verbose=2)
print('Cross-validation score: {}'.format(scores))
print('Cross-validation score: {}'.format(sum(scores)/5))
print('=== Finished cross-validation ====')

print('==== Starting fitting ====')
clf.fit(X, y)
print('Starting predicting.')
pred = clf.predict_proba(Xtest)[:,1]
print('==== Finished predicting ====')

print('==== Creating submission file ====')
submission_id = pd.DataFrame({'id': test_resources["id"]})
submission = pd.concat([submission_id, pd.DataFrame({'project_is_approved': pred})], axis=1)
print(submission.head())
submission.to_csv('submission.csv', index=False)
print('==== All done! Have a nice day! ====')

# I hope this helped! The result won't put you up very high on the leaderboard but I think it will
# give you a good understanding of the process behind a simple logistic regression solution.
