"""
Created on Thu Jul 27 19:59:46 2017

@author: suresh
"""
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy
import pandas as pd
import tqdm
# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import *
import string
import xgboost as xgb

train = pd.read_csv('../input/training_variants')
test = pd.read_csv('../input/test_variants')
trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
#y = train['Class'].values
#train = train.drop(['Class'], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values

def cleanup(text):
    text = text.lower()
    text= text.translate(str.maketrans("","", string.punctuation))
    return text
train['Text'] = train['Text'].apply(cleanup)
test['Text'] = test['Text'].apply(cleanup)
allText = train['Text'].append(test['Text'],ignore_index=True)
def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

sentences = constructLabeledSentences(allText)
model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=8,iter=100,seed=1)

model.build_vocab(sentences)

model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

#model.save('./docEmbeddings.d2v')

train_arrays = numpy.zeros((train.shape[0], 100))
#train_arrays = numpy.empty((0,100), float)
train_labels = numpy.zeros(train.shape[0])
#train_labels = numpy.empty((0,1),int)
#for i in range(train.shape[0]):
#    try:
#        train_arrays[i] = model['Text_'+str(i)]
#        #train_arrays = numpy.append(train_arrays, numpy.array([model[str(i)]]), axis=0)
#        train_labels[i] = train["Class"][i]
#        #train_labels = numpy.append(train_labels,numpy.array[[train["Class"][i]]])
#    except:
#        pass
for i in range(train.shape[0]):
    train_arrays[i] = model.docvecs['Text_'+str(i)]
    #train_arrays = numpy.append(train_arrays, numpy.array([model[str(i)]]), axis=0)
    train_labels[i] = train["Class"][i]
    #train_labels = numpy.append(train_labels,numpy.array[[train["Class"][i]]])
X_train, X_test, y_train, y_test = train_test_split(train_arrays, train_labels, test_size=0.2, random_state=42)
#svm scores 0.91 on LB with 200D representation
def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf',probability=True)
    svm.fit(X, y)
    return svm
#svm = train_svm(X_train, y_train)

#pred = svm.predict(X_test)

# Output the hit-rate and the confusion matrix for each model
#print(svm.score(X_test, y_test))
#print(confusion_matrix(pred, y_test))
test_arrays = numpy.zeros((test.shape[0], 100))
for i in range(train.shape[0],allText.shape[0]):
    test_arrays[i-train.shape[0]] = model.docvecs['Text_'+str(i)]
#submission = svm.predict(test_arrays)
#submission_prob = svm.predict_proba(test_arrays)
#sub= pd.DataFrame(submission_prob)
#sub.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']
#sub.insert( 0,"ID",test["ID"]) 
#sub.to_csv("submission_GensimDoc2Vec.csv",index=False)
def getClassCounts (similarDocs):
    indices = [sd[0] for sd in similarDocs]
    cts = [0.0]*9
    for text in indices:
        sp = int(text.split("_")[1])
        if sp<(train.shape[0]):
            cts[int(train_labels[sp])-1]+=1.0
        else:
            cts[6]+=1.0
    return cts
#nearest neighbor - scores 0.97 on LB with 200D representation
#nearestNeighborSubmission = numpy.zeros((test.shape[0],9))
#for i in range(train.shape[0],allText.shape[0]):
#    ms = model.docvecs.most_similar('Text_'+str(i))
#    cts = getClassCounts(ms)
#    cts = [c / sum(cts) for c in cts]
#    nearestNeighborSubmission[i-train.shape[0]] = numpy.array(cts)
#subNN = pd.DataFrame(nearestNeighborSubmission)
#subNN.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']
#subNN.insert(0,"ID",test["ID"])
#subNN.to_csv("submission_GensimDoc2VecNN.csv",index=False)

###xgboost
y = train_labels - 1 #fix for zero bound array

denom = 0
#following xgboost snippet taken from other available kernels
#with 5 folds, scores 0.79 on LB with 200D representation
fold = 1 #Change to 5, 1 for Kaggle Limits
for i in range(fold):
    params = {
        'eta': 0.03333,
        'max_depth': 4,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train_arrays, y, test_size=0.18, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print(score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test_arrays), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test_arrays), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)
preds /= denom
submission_xgb = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission_xgb['ID'] = test["ID"]
submission_xgb.to_csv('submission_xgb.csv', index=False)