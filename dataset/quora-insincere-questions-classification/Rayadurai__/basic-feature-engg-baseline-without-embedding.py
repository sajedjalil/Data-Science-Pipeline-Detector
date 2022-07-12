import numpy as np 
import pandas as pd 

import os,re,gc
import warnings
from sklearn.model_selection import train_test_split,StratifiedKFold,StratifiedShuffleSplit
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize,TweetTokenizer   
import string

from keras.models import Model
from keras.layers import Dense, Input,concatenate,LSTM, Embedding, Dropout, Activation,  Conv1D,CuDNNGRU,CuDNNLSTM
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from tqdm import tqdm
tqdm.pandas()
print(os.listdir("../input"))
pd.options.display.max_columns=50
pd.options.display.max_colwidth=100
pd.options.mode.chained_assignment = None

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()

train = pd.read_csv('../input/train.csv')#,nrows=30000)
test = pd.read_csv('../input/test.csv')#,nrows=30000)

eng_stopwords = set(stopwords.words("english"))
def rawdata_features(df):
    col = 'question_text'
    df['count_sent'] = (df[col].apply(lambda x: len(re.findall(r'\n',str(x)))+1)).fillna(0)
    df['count_word'] = df[col].apply(lambda x : len(str(x).split()))
    nums = re.compile(r"[+-]?\d+(?:\,\d+)?")
    df['count_numbers'] = df[col].apply(lambda x: len( nums.findall(x)))
    df['count_char']=df[col].apply(lambda x: len(str(x)))
    df['count_punc'] = df[col].apply(lambda x : len([i for i in str(x) if i in string.punctuation]))
    df['count_caps'] = df[col].apply(lambda x : len(re.findall(r'[A-Z]',x)))
    df["count_title"] = df[col].apply(lambda x: len([i for i in str(x).split() if i.istitle()]))
    df["count_stopwords"] = df[col].apply(lambda x: len([i for i in str(x).lower().split() if i in eng_stopwords]))
    df['count_eng_words'] = df[col].apply(lambda x: len([i for i in str(x).split() if  wordnet.synsets(i)]))
    return df

train = rawdata_features(train)
test = rawdata_features(test)
 
lem = WordNetLemmatizer()
eng_alpha = set(string.ascii_lowercase)

def clean_text(df):
    col = 'cleaned_questions'
    df[col] = df['question_text'].apply(lambda x : ' '.join( re.findall(r'[a-zA-Z]+',x)))
    df[col] = df['question_text'].apply(str.lower)
    df[col] = df['question_text'].apply(lambda x: 
                            ' '.join([lem.lemmatize(lem.lemmatize(i),'v') for i in x.split() if set(i).issubset(eng_alpha)]))
    return df

    
train = clean_text(train)
test = clean_text(test)
       
count_features = [i for i in train.columns if 'count' in i]
train[count_features] = (train[count_features] - train[count_features].mean()) / train[count_features].std()
test[count_features] = (test[count_features] - test[count_features].mean()) / test[count_features].std()

maxlen = 100
max_features = 50000
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train['cleaned_questions'].values) + list(test['cleaned_questions'].values))
train_vec = tokenizer.texts_to_sequences(train['cleaned_questions'].values)
test_vec = tokenizer.texts_to_sequences(test['cleaned_questions'].values)
train_vec = sequence.pad_sequences(train_vec, maxlen=maxlen)
test_vec = sequence.pad_sequences(test_vec, maxlen=maxlen)

train_target = train.target

train_data = np.concatenate((train_vec,train[count_features].values),axis=1) 
test_data = np.concatenate((test_vec,test[count_features].values),axis=1)
print(train_data.shape)


X_train, X_val, y_train, y_val = train_test_split(train_data,
                                                  train_target, 
                                                  test_size=0.1, 
                                                  random_state=22, 
                                                  stratify=train_target)


from keras import backend as K
def f1_score(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    if c3 == 0:
        return 0
    precision = c1 / c2
    recall = c1 / c3
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score 


def LSTM_model():
    inp = Input(shape=(X_train.shape[1], ))
    x = Embedding(max_features, 100)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    x = Dense(64, activation="relu")(conc)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy',f1_score])
    print(model.summary())
    return model

batch_size = 1024
epochs = 2
class_weight= (abs(y_train-1).value_counts()/len(y_train)).to_dict()
print(class_weight)

model = LSTM_model()
model.fit(X_train, 
          y_train, 
          batch_size=batch_size, 
          epochs=epochs, 
          #class_weight=class_weight,
          validation_data=(X_val, y_val))
          
test_pred = model.predict(test_data,batch_size=1024)
val_pred = model.predict(X_val,batch_size=1024)    

best_threshold = 0.01
best_score = 0.0
for threshold in range(1, 100):
    threshold = threshold / 100
    score = metrics.f1_score(y_val, val_pred > threshold)
    if score > best_score:
        best_threshold = threshold
        best_score = score
print(best_score)
test_pred = (test_pred > best_threshold).astype(int)

submission = pd.read_csv('../input/sample_submission.csv')
submission['prediction'] = test_pred
submission.to_csv('submission.csv', index=False)
