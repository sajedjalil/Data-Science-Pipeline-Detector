from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import json
import numpy
import time

def generate_text(data):
  text_data = [" ".join(doc['ingredients']).lower() for doc in data]
  return text_data 
  
def main():
  train = json.load(open('../input/train.json'))
  test = json.load(open('../input/test.json'))
  
  train_text = generate_text(train)
  test_text = generate_text(test)
  target = [doc['cuisine'] for doc in train]
  
  tfidf = TfidfVectorizer(binary=True)
  X = tfidf.fit_transform(train_text).astype('float16')
  X_test = tfidf.transform(test_text).astype('float16')
  
  lb = LabelEncoder()
  y = lb.fit_transform(target)
  
  classifier = SVC(C=250, # penalty parameter
    kernel='rbf', # kernel type, rbf working fine here
    degree=3, # default value
    gamma=1, # kernel coefficient
    coef0=1, # change to 1 from default value of 0.0
    shrinking=True, # using shrinking heuristics
    tol=0.001, # stopping criterion tolerance 
    probability=False, # no need to enable probability estimates
    cache_size=200, # 200 MB cache size
    class_weight=None, # all classes are treated equally 
    verbose=False, # print the logs 
    max_iter=-1, # no limit, let it run
    decision_function_shape=None, # will use one vs rest explicitly 
    random_state=None)
  model = OneVsRestClassifier(classifier, n_jobs=-1)
  
  train_model_start_secs = time.time()
  model.fit(X, y)
  train_model_elapsed_secs = time.time() - train_model_start_secs
  print("Model training finished, seconds elapsed: %f" % train_model_elapsed_secs)
  
  print ("Predict on test data ... ")
  predict_start_secs = time.time()
  y_test = model.predict(X_test)
  y_pred = lb.inverse_transform(y_test)
  predict_elapsed_secs = time.time() - predict_start_secs
  print("Prediction finished, seconds elapsed: %f" % predict_elapsed_secs)

  test_id = [doc['id'] for doc in test]
  sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
  sub.to_csv('svm_output.csv', index=False)

if __name__ == "__main__":
  main()


# Any results you write to the current directory are saved as output.