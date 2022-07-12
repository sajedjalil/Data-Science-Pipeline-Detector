
import subprocess
# Change tensorflow version to 1.15
subprocess.run('pip install tensorflow==1.15 tensorflow-gpu==1.15', shell=True, check=True)
# Install bert-as-service
subprocess.run('pip install bert-serving-server', shell=True, check=True)
subprocess.run('pip install bert-serving-client', shell=True, check=True)
# Download and unzip the pre-trained model if necessary
subprocess.run('wget http://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip', shell=True, check=True)
subprocess.run('unzip uncased_L-12_H-768_A-12.zip', shell=True, check=True)

# Import Package
import os
import time
import pandas as pd, numpy as np
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from bert_serving.client import BertClient

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load Data
Xtrain = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
Xtest = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
Xtrain["comment_text"].fillna("fillna")
Xtest["comment_text"].fillna("fillna")

print('Train shape:', Xtrain.shape)
print('Tetst shape:', Xtest.shape)

Xtrain['filter'] = 0

for cn in class_names:
    print()
    print(Xtrain[cn].value_counts())
    Xtrain['filter'] += Xtrain[cn]

# %% [code]
print(Xtrain['filter'].value_counts())

# %% [code]
# down sampling
X_f1 = Xtrain[Xtrain['filter'] > 0].copy()
X_f0 = Xtrain[Xtrain['filter'] == 0].copy()

X_f0 = X_f0.sample(frac=0.15, replace=False)

Xtrain_ = X_f0.append(X_f1)

del Xtrain_['filter']

for cn in class_names:
    print()
    print(Xtrain_[cn].value_counts())
    
Xtr = Xtrain_[['comment_text']]
ytr = Xtrain_[class_names]

Xts = Xtest[['comment_text']]

xtrain, xtest, ytrain, ytest = train_test_split(
    Xtr,ytr,test_size=0.20)
print(xtrain.shape)

for cn in class_names:
    print()
    print('Train', '-'*20)
    print(ytrain[cn].value_counts())
    
for cn in class_names:
    print()
    print('Valid', '-'*20)
    print(ytest[cn].value_counts())


# Start the BERT server
bert_command = 'bert-serving-start -model_dir uncased_L-12_H-768_A-12 -max_seq_len 50'
process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)

# # Starting BertServer from Python script
# from bert_serving.server.helper import get_args_parser
# from bert_serving.server import BertServer
# args = get_args_parser().parse_args(['-model_dir', 'uncased_L-12_H-768_A-12',
#                                      '-port', '5555',
#                                      '-port_out', '5556',
#                                      '-max_seq_len', 'NONE',
#                                      '-cpu', 'False'])
# server = BertServer(args)
# server.start()

# Start the BERT client
bc = BertClient()

train_comment_text = xtrain['comment_text'].tolist()
valid_comment_text = xtest['comment_text'].tolist()
test_comment_text = Xts['comment_text'].tolist()

train_comment_text = [x if len(x)>5 else 'ts,dr' + x for x in train_comment_text ]
valid_comment_text = [x if len(x)>5 else 'ts,dr' + x for x in valid_comment_text ]
test_comment_text = [x if len(x)>5 else 'ts,dr' + x for x in test_comment_text ]


train_tokenize = bc.encode(train_comment_text)
valid_tokenize = bc.encode(valid_comment_text)
test_tokenize = bc.encode(test_comment_text)

# Create Submission file
submission = pd.DataFrame.from_dict({'id': Xtest['id']})

# Build a Classifier
for class_name in class_names:
    print()
    print(class_name)
    print('-' * 30)
    train_target = ytrain[class_name]
    
    print('Fitting.....')
    clf = xgb.XGBClassifier(max_depth=7, random_state=0)
    clf.fit(train_tokenize, train_target)
    print('Finish.')
    # valid
    pred = clf.predict(valid_tokenize)
    print(pred)
    print()
    acc = accuracy_score(ytest[class_name], pred)
    print('Accuracy:', acc)
    print()
    print(confusion_matrix(ytest[class_name], pred))
    # submit
#     submission[class_name] = clf.predict_proba(test_tokenize)[:, 1]
    submission[class_name] = clf.predict(test_tokenize)

submission.to_csv('submission.csv', index=False)