import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer # Text preprocessing, tokenizing and filtering of stopwords
from sklearn.feature_extraction.text import TfidfTransformer # Word rates, downscale weights for words that occur in many documents
from sklearn.naive_bayes import MultinomialNB # Multinomial Naive Bayes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split # Split training set into training and validation sets
from sklearn import metrics
from sklearn.pipeline import Pipeline
from mlutilities.mlutilities import estimate, classificationResults

# Input data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Split training set into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(train.question_text, train.target, test_size=0.2, random_state=42)

# Estimate Function
def estimate(estimator, parameters, X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(estimator, parameters, cv=5, iid=False, n_jobs=-1)
    model.fit(X_train, y_train)
    print("BEST SCORE: %r; BEST PARAMETERS: %s" % (model.best_score_, model.best_params_))
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    params = model.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Score: %f (%f); Parameters: %r" % (mean, stdev, param))
    print("")
    return model
    
# Classification Results
def classificationResults(predicted, y_test):
    from sklearn import metrics
    print("Classification Report")
    print(metrics.classification_report(y_test, predicted))
    print("")
    print("Confusion Matrix")
    print(metrics.confusion_matrix(y_test, predicted))
    print("")

# Make a pipeline with count vectors, division of counts by length of question
pipe = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB())
 ])
 
# Parameters for Multinomial Naive Bayes Model
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3)
}

model = estimate(pipe, parameters, X_train, y_train)

predictions = model.predict(X_test)
classificationResults(predictions, y_test)

# Predict using test data and make submission file
predict_test = model.predict(test.question_text)
submission = pd.DataFrame({'qid':test['qid'],'prediction':predict_test})
submission.to_csv('submission.csv',index=False)