import numpy as np
from sklearn import  preprocessing
#import xgboost as xgb
#import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
import random
import gc
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re, string

SEED = 42  # always use a seed for randomized procedures

def bagged_set(X_t,y_c,model, seed, estimators, xt,word_vec, update_seed=True):
    
   # create array object to hold predictions
   baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]

   X_t = word_vec.transform(X_t)
   xt = word_vec.transform(xt)
   
   #loop for as many times as we want bags
   for n in range (0, estimators):
        m,r = get_mdl(y_c)
        model.fit(X_t,y_c)
        #preds=model.predict_proba(xt)[:,1] # predict probabilities
        preds=m.predict_proba(xt.multiply(r))[:,1]
        # update bag's array
        for j in range (0, (xt.shape[0])):           
            baggedpred[j]+=preds[j]
   # divide with number of bags to create an average estimate            
   for j in range (0, len(baggedpred)): 
        baggedpred[j]/=float(estimators)
   # return probabilities            
   return np.array(baggedpred) 
   
   
# using numpy to print results
def printfilcsve(X, filename):
    np.savetxt(filename,X)

# used for nbsvm
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def tokenize(s):
    re_tok = re.compile('([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


def main():
    
    filename="logit_word_char_tfidf" # name prefix
    #model= xgb.XGBClassifier(max_depth=5, n_estimators=100, colsample_bytree=0.8,subsample=0.8, nthread=10, learning_rate=0.1)
    #model = RGFClassifier(max_leaf=500,algorithm="RGF",test_interval=100, loss="LS")
    #model=CatBoostClassifier(iterations=80, depth=3, learning_rate=0.1, loss_function='Logloss')
    #model=lgb.LGBMClassifier(num_leaves=150,objective='binary',max_depth=6,learning_rate=.01,max_bin=400,auc='binary_logloss')
    model = LogisticRegression(C=4, penalty="l2")

    # === load data in memory === #
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    train = train.fillna("unknown")
    test = test.fillna("unknown")

    print("loading data")
    X = train['comment_text']
    y_tr = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    X_test = test['comment_text']
    merge=pd.concat([train['comment_text'],test['comment_text']],axis=0)
    
    #transform_com = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
    #           smooth_idf=1, sublinear_tf=1).fit(merge)

    word_vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize, min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,smooth_idf=1, sublinear_tf=1 )
    #word_vec = TfidfVectorizer(
    #sublinear_tf=True,
    #strip_accents='unicode',
    #analyzer='word',
    #token_pattern=r'\w{1,}',
    #stop_words='english',
    #ngram_range=(1, 3),
    #max_features=10000)
    word_vec.fit(merge)
   
    print("TfidfVectorizer run")
    #create arrays to hold cv an dtest predictions
    train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ]

    # === training & metrics === #
    
    bagging=1 # number of models trained with different seeds
    n = 2  # number of folds in strattified cv
    col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_stacker_1 = np.zeros((train.shape[0], len(col)))

    for ii, j in enumerate(col):
        print('fit '+j)
        y = y_tr[j]
        mean_auc = 0.0
        kfolder=StratifiedKFold(y, n_folds= n,shuffle=True, random_state=SEED)     
        i=0
        for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object
            # creaning and validation sets
            X_train, X_cv = X[train_index], X[test_index]
            y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]

            # train model and make predictions 
            preds=bagged_set(X_train,y_train,model, SEED , bagging, X_cv,word_vec,update_seed=True)
            gc.collect()
        
            # compute AUC metric for this CV fold
            roc_auc = roc_auc_score(y_cv, preds)
            print("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
            mean_auc += roc_auc
        
            no=0
            for real_index in test_index:
                 train_stacker[real_index]=(preds[no])
                 no+=1
            i+=1
        train_stacker_1[:,ii]=train_stacker
        mean_auc/=n
        print((" Average AUC: %f" % (mean_auc) ))
    print (" printing train datasets ")
    printfilcsve(np.array(train_stacker_1), filename + ".train.csv")          

    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    train_stacker_2 = np.zeros((test.shape[0], len(col)))
    for ii, j in enumerate(col):
        print('fit '+j)
        y = y_tr[j]
        preds=bagged_set(X, y,model, SEED, bagging, X_test,word_vec, update_seed=True)
        train_stacker_2[:,ii]=preds
    
    #create submission file 
    printfilcsve(np.array(train_stacker_2), filename+ ".test.csv")  

if __name__ == '__main__':
    main()
