import pandas as pd
import numpy as np
import gc
import os

train = pd.read_csv("../input/donorschoose-application-screening/train.csv")
test = pd.read_csv("../input/donorschoose-application-screening/test.csv")
resources = pd.read_csv("../input/donorschoose-application-screening/resources.csv")
len_train = len(train)
data=train.append(test)

resources['total_price'] = resources.quantity * resources.price
mean_total_price = pd.DataFrame(resources.groupby('id').total_price.mean())
sum_total_price = pd.DataFrame(resources.groupby('id').total_price.sum())
count_total_price = pd.DataFrame(resources.groupby('id').total_price.count())
mean_total_price['id'] = mean_total_price.index
sum_total_price['id'] = mean_total_price.index
count_total_price['id'] = mean_total_price.index


char_cols = ['project_title', 'project_essay_1', 'project_essay_2',
             'project_essay_3', 'project_essay_4', 'project_resource_summary']
def create_features(df):
    df = pd.merge(df, mean_total_price, on='id')
    df = pd.merge(df, sum_total_price, on='id')
    df = pd.merge(df, count_total_price, on='id')
    df['year'] = df.project_submitted_datetime.apply(lambda x: x.split("-")[0])
    df['month'] = df.project_submitted_datetime.apply(lambda x: x.split("-")[1])
    for col in char_cols:
        df[col] = df[col].fillna("NA")
    df['text'] = df.apply(lambda x: " ".join(x[col] for col in char_cols), axis=1)
    return df
    
data = create_features(data)
data['teacher_prefix'] = data['teacher_prefix'].fillna('Teacher')
from sklearn.preprocessing import LabelEncoder
temp_data = data[["teacher_prefix", "school_state", "year", "month", "project_grade_category",
                "project_subject_categories", "project_subject_subcategories"]].apply(LabelEncoder().fit_transform)
data[["teacher_prefix", "school_state", "year", "month", "project_grade_category",
                "project_subject_categories", "project_subject_subcategories"]]=temp_data[["teacher_prefix", "school_state", "year", "month", "project_grade_category",
                "project_subject_categories", "project_subject_subcategories"]]
char_cols = ['project_title', 'project_essay_1', 'project_essay_2','project_essay_3', 'project_essay_4', 'project_resource_summary']
data.drop(char_cols,1,inplace=True)
data.drop(['project_submitted_datetime'],1,inplace=True)

train = data[0:len_train]
test = data[len_train:]


num_features = ["teacher_number_of_previously_posted_projects", "total_price_x", "total_price_y", "total_price"]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train[num_features])
X_test_num = scaler.transform(test[num_features])

category_features = ["teacher_prefix", "school_state", "year", "month", "project_grade_category",
                "project_subject_categories", "project_subject_subcategories"]
X_train_cat = np.array(train[category_features], dtype=np.int)
X_test_cat = np.array(test[category_features], dtype=np.int)


import re
from keras.preprocessing import text, sequence
max_features = 100000  # 50000
maxlen = 300
tokenizer = text.Tokenizer(num_words=max_features)
def preprocess1(string):
    '''
    :param string:
    :return:
    '''
    string = re.sub(r'(\")', ' ', string)
    string = re.sub(r'(\r)', ' ', string)
    string = re.sub(r'(\n)', ' ', string)
    string = re.sub(r'(\r\n)', ' ', string)
    string = re.sub(r'(\\)', ' ', string)
    string = re.sub(r'\t', ' ', string)
    string = re.sub(r'\:', ' ', string)
    string = re.sub(r'\"\"\"\"', ' ', string)
    string = re.sub(r'_', ' ', string)
    string = re.sub(r'\+', ' ', string)
    string = re.sub(r'\=', ' ', string)

    return string

train["text"] = train["text"].apply(preprocess1)
test["text"] = test["text"].apply(preprocess1)
#fit_on_texts 
tokenizer.fit_on_texts(train["text"].tolist() + test["text"].tolist())
list_tokenized_train = tokenizer.texts_to_sequences(train["text"].tolist())
list_tokenized_test = tokenizer.texts_to_sequences(test["text"].tolist())
X_train_words = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test_words = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
#lable
X_train_target = train.project_is_approved


train.drop(['teacher_id'],1,inplace=True)
test.drop(['teacher_id'],1,inplace=True)


num_features = ["teacher_number_of_previously_posted_projects", "total_price_x", "total_price_y", "total_price"]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train[num_features])
X_test_num = scaler.transform(test[num_features])


category_features = ["teacher_prefix", "school_state", "year", "month", "project_grade_category",
                "project_subject_categories", "project_subject_subcategories"]
X_train_cat = np.array(train[category_features], dtype=np.int)
X_test_cat = np.array(test[category_features], dtype=np.int)

# max_teacher_prefix = np.max([train['teacher_prefix'].max(), test['teacher_prefix'].max()])+1
max_school_state = np.max([train['school_state'].max(), test['school_state'].max()])+1
# max_year = np.max([train['year'].max(), test['year'].max()])+1
max_month = np.max([train['month'].max(), test['month'].max()])+1
max_project_grade_category = np.max([train['project_grade_category'].max(), test['project_grade_category'].max()])+1
max_project_subject_categories = np.max([train['project_subject_categories'].max(), test['project_subject_categories'].max()])+1
max_project_subject_subcategories = np.max([train['project_subject_subcategories'].max(), test['project_subject_subcategories'].max()])+1




import re
from keras.preprocessing import text, sequence
max_features = 100000  # 50000
maxlen = 300
tokenizer = text.Tokenizer(num_words=max_features)

#fit_on_texts 
tokenizer.fit_on_texts(train["text"].tolist() + test["text"].tolist())
list_tokenized_train = tokenizer.texts_to_sequences(train["text"].tolist())
list_tokenized_test = tokenizer.texts_to_sequences(test["text"].tolist())
X_train_words = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test_words = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
#lable
X_train_target = train.project_is_approved

import os
EMBEDDING_DIM = 100
GLOVE_DIR = '../input/glove-global-vectors-for-word-representation/'
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.%dd.txt'%EMBEDDING_DIM),encoding='utf8')
for line in f:
    if len(line)>20:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()
#embeddings_index :  word:vector
print('Found %s word vectors.' % len(embeddings_index))
embed_size =EMBEDDING_DIM
word_index = tokenizer.word_index
# prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


def get_keras_train_data(dataset):
    X = {
        # 'teacher_prefix': np.array(dataset.teacher_prefix),
        'school_state': np.array(dataset.school_state),
        # 'year': np.array(dataset.year),
        'month': np.array(dataset.month),
        'project_grade_category': np.array(dataset.project_grade_category),
        'project_subject_categories': np.array(dataset.project_subject_categories),
        'project_subject_subcategories': np.array(dataset.project_subject_subcategories),
        'num' : X_train_num,
        'words':X_train_words,

    }
    return X

def get_keras_test_data(dataset):
    X = {
        # 'teacher_prefix': np.array(dataset.teacher_prefix),
        'school_state': np.array(dataset.school_state),
        # 'year': np.array(dataset.year),
        'month': np.array(dataset.month),
        'project_grade_category': np.array(dataset.project_grade_category),
        'project_subject_categories': np.array(dataset.project_subject_categories),
        'project_subject_subcategories': np.array(dataset.project_subject_subcategories),
        'num' : X_test_num,
        'words':X_test_words,

    }
    return X


X_train = get_keras_train_data(train)
X_test = get_keras_test_data(test)

from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Convolution1D, \
    GlobalMaxPool1D, SpatialDropout1D, CuDNNGRU, Bidirectional, PReLU, GRU,LSTM
from keras.models import Model
from keras import optimizers


def get_model():
    emb_n = 5
    # in_teacher_prefix = Input(shape=[1], name = 'teacher_prefix')
    # emb_teacher_prefix = Embedding(max_teacher_prefix, 1,trainable=True)(in_teacher_prefix)

    in_school_state = Input(shape=[1], name = 'school_state')
    emb_school_state = Embedding(max_school_state, emb_n,trainable=True)(in_school_state)

    # in_year= Input(shape=[1], name = 'year')
    # emb_year = Embedding(max_year, 2,trainable=True)(in_year)

    in_month = Input(shape=[1], name = 'month')
    emb_month = Embedding(max_month, 2,trainable=True)(in_month)

    in_project_grade_category = Input(shape=[1], name = 'project_grade_category')
    emb_project_grade_category = Embedding(max_project_grade_category, emb_n,trainable=True)(in_project_grade_category)

    in_project_subject_categories = Input(shape=[1], name = 'project_subject_categories')
    emb_project_subject_categories = Embedding(max_project_subject_categories, emb_n,trainable=True)(in_project_subject_categories)

    in_project_subject_subcategories = Input(shape=[1], name = 'project_subject_subcategories')
    emb_project_subject_subcategories = Embedding(max_project_subject_subcategories, emb_n,trainable=True)(in_project_subject_subcategories)

    x_cat = concatenate([ (emb_school_state), (emb_month), (emb_project_grade_category),
                     (emb_project_subject_categories), (emb_project_subject_subcategories)],axis=-1)

    x_cat = Dropout(0.3)(x_cat)
    x_cat = Flatten()(x_cat)

    input_num = Input((len(num_features),),name= 'num')
    input_words = Input((maxlen,),name= 'words')

    x_words = Embedding(max_features, 100,
                            weights=[embedding_matrix],
                            trainable=True)(input_words)
    x_words = SpatialDropout1D(0.3)(x_words)
    # x_words = Convolution1D(100, 3, activation="relu")(x_words)
    x_words = Bidirectional(CuDNNGRU(100, return_sequences=True))(x_words)
    x_words = Convolution1D(100, 3, activation="relu")(x_words)
    x_words = GlobalMaxPool1D()(x_words)

    x_cat = Dense(50, activation="relu")(x_cat)
    x_num = Dense(50, activation="relu")(input_num)

    x = concatenate([x_cat, x_num, x_words])

    x = Dense(50, activation="relu")(x)
    x = Dropout(0.25)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[in_school_state,in_month ,in_project_grade_category,in_project_subject_categories,in_project_subject_subcategories,input_num,input_words], outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = get_model()
model.summary()

from keras.callbacks import *
from sklearn.metrics import roc_auc_score

file_path = 'simpleRNN3.h5'
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True,
                             mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=4)
lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=2,
                               verbose=1,
                               epsilon=1e-4,
                               mode='min')
callbacks_list = [checkpoint, early, lr_reduced]
history = model.fit(X_train, X_train_target, validation_split=0.1,
                    verbose=2, callbacks=callbacks_list,
                    epochs=15, batch_size=512)
del X_train_cat, X_train_num, X_train_words, X_train_target
model.load_weights(file_path)
pred_test = model.predict(X_test, batch_size=2000)

test["project_is_approved"] = pred_test
test[['id', 'project_is_approved']].to_csv("submission.csv", index=False)


