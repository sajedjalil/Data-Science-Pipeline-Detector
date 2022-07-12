import numpy as np
import pandas as pd
import math
import time
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn import preprocessing as pre
import gc

NR_MAX_TEXT_FEATURES = 100000
NAME_MIN_DF = 10

def print_duration (start_time, msg):
    print("[%d] %s" % (int(time.time() - start_time), msg))
    start_time = time.time()
    return start_time


def main():
    start_time = time.time()
    ### Read data
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    result = test[['id']].copy() 
    # toxic, severe_toxic, obscene, threat, insult, identity_hate
    y_toxic = train['toxic']
    
    # merge train and test for feature engineering process
    nrow_train = train.shape[0] # store the index where train rows begin
    data = pd.concat([train, test])
    # free import data structures again and run GC
    del train
    del test
    gc.collect()
    data['comment_text'].fillna(value='', inplace=True)
    start_time = print_duration(start_time, "Finished to read data")
                         
    scaler = pre.StandardScaler(with_mean=False)
    cv = CountVectorizer(min_df=NAME_MIN_DF)
    count_matrix = cv.fit_transform(data['comment_text'])
    count_matrix_sparse = count_matrix.tocsr()
    # standardize the matrix
    count_matrix_sparse = scaler.fit_transform(count_matrix_sparse)
    
    # split train from test rows
    X = count_matrix_sparse[:nrow_train]
    X_test = count_matrix_sparse[nrow_train:]
    start_time = print_duration(start_time, "Finished transform comment text")
    
    # try partial fit
    model_toxic = MLPRegressor(solver='adam', hidden_layer_sizes=(10,), random_state=1)
    batch_size = 10000
    total_rows = X.shape[0]
    duration = 0
    start_train = time.time()
    pos = 0
    while duration < 2500 and (pos+batch_size) <= total_rows:
        X_p = X[pos:pos+batch_size]
        y_p = y_toxic[pos:pos+batch_size]
        model_toxic.partial_fit(X_p, y_p)
        pos = pos + batch_size
        duration = time.time() - start_train # how long did we train so far?
        print("Pos %d/%d duration %d" % (pos, total_rows, duration))
    # end test partial fit  
    
    preditions = np.array()
    preditions[0] = model_toxic.predict(X_test)
    start_time = print_duration(start_time, "Finished to predict result")
    
    
    min_max_scaler = pre.MinMaxScaler()
    
    result['toxic'] = preditions[0]
    result.to_csv('submission.csv', encoding='utf-8', index=False)
    start_time = print_duration(start_time, "Finished to store result")
    
if __name__ == '__main__':
    main()