"""
Use Facebook's fasttext algorithm to classify recipes from Yummly. 

Custom install fasttext from github:
    https://github.com/facebookresearch/fastText
    
I wrote a custom script to tune the model.

I use majority voting with 3 fasttext classifiers using 1-3 ngrams
"""

################
# PREPARATIONS #
################

# Import libraries/functions
import pandas as pd
import numpy as np 
import re
import fastText as ft
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from collections import Counter

# Import data
df = pd.read_json('../input/train.json')
df_test = pd.read_json('../input/test.json')

##################
# PRE-PROCESSING #
##################

# Concatenate ingredients for training and test data
df.ingredients = df.ingredients.apply(lambda x: ' '.join(x))
df_test.ingredients = df_test.ingredients.apply(lambda x: ' '.join(x))

# Concatenate labels and ingredients for fasttext formatted training data
df['labels_text'] = '__label__' + df.cuisine 
df.labels_text = df.labels_text.str.cat(df.ingredients, sep=' ')

# Write training data to a file as required by fasttext
training_file = open('train.txt','w')
training_file.writelines(df.labels_text + '\n')
training_file.close()

################
# MODEL TUNING #
################

# Function to do K-fold CV across different fasttext parameter values
def tune(Y, X, YX, k, lr, wordNgrams, epoch):
    # Record results
    results = []
    for lr_val in lr:
        for wordNgrams_val in wordNgrams:
            for epoch_val in epoch:  
                # K-fold CV
                kf = KFold(n_splits=k, shuffle=True)
                fold_results = []
                # For each fold
                for train_index, test_index in kf.split(X):
                    # Write the training data for this CV fold
                    training_file = open('train_cv.txt','w')
                    training_file.writelines(YX[train_index] + '\n')
                    training_file.close()
                    # Fit model for this set of parameter values
                    model = ft.FastText.train_supervised('train_cv.txt',
                                          lr=lr_val,
                                          wordNgrams=wordNgrams_val,
                                          epoch=epoch_val)
                    # Predict the holdout sample
                    pred = model.predict(X[test_index].tolist())
                    pred = pd.Series(pred[0]).apply(lambda x: re.sub('__label__', '', x[0]))
                    # Compute accuracy for this CV fold
                    fold_results.append(accuracy_score(Y[test_index], pred.values))
                # Compute mean accuracy across 10 folds
                mean_acc = pd.Series(fold_results).mean()
                print([lr_val, wordNgrams_val, epoch_val, mean_acc])
    # Add current parameter values and mean accuracy to results table
    results.append([lr_val, wordNgrams_val, epoch_val, mean_acc])         
    # Return as a DataFrame 
    results = pd.DataFrame(results)
    results.columns = ['lr','wordNgrams','epoch','mean_acc']
    return(results)
    
# Tune parameters using 10-fold CV
#results = tune(Y = df.cuisine,
#     X = df.ingredients,
#     YX = df.labels_text,
#     k = 10, 
#     lr = [0.05, 0.1, 0.2],
#     wordNgrams = [1,2,3],
#     epoch = [15,17,20])

#########################
# TRAINING & PREDICTION #
#########################

# Train final classifiers
classifier1 = ft.FastText.train_supervised('train.txt', lr=0.1, wordNgrams=1, epoch=15)
classifier2 = ft.FastText.train_supervised('train.txt', lr=0.1, wordNgrams=2, epoch=15)
classifier3 = ft.FastText.train_supervised('train.txt', lr=0.1, wordNgrams=3, epoch=15)

# Predict test data
predictions1 = classifier1.predict(df_test.ingredients.tolist())
predictions2 = classifier2.predict(df_test.ingredients.tolist())
predictions3 = classifier3.predict(df_test.ingredients.tolist())

# Combine predictions
majority_vote = np.array([])
for i in range(len(predictions1[0])):
    majority_vote = np.append(majority_vote, Counter([predictions1[0][i][0],
                                                   predictions2[0][i][0],
                                                   predictions3[0][i][0]]).most_common(1)[0][0])

# Write submission file
submit = pd.DataFrame({'id': df_test.id, 
                       'cuisine': pd.Series(majority_vote)})
submit.cuisine = submit.cuisine.apply(lambda x: re.sub('__label__', '', x))
submit.to_csv('submit.csv', index=False)

#######
# END #
#######
