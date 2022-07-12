# Forked from https://www.kaggle.com/CVxTz/keras-baseline-feature-hashing-cnn
'''
The first trick: Using kfold.
The second trick: Train model overfitting a bit. I  train 20 epochs with the whole train data for every fold 
and let it overfitting a bit.

After several hours training on 1080ti gpu, I got a 0.812x lb score. 
I can not train a single model over 0.82 score. Maybe someone can do that.
Hope these can help others training some better models.

'''


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(os.listdir("../input"))
print('Good luck!')

'''
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
resources = pd.read_csv("../input/resources.csv")
train = train.sort_values(by="project_submitted_datetime")

teachers_train = list(set(train.teacher_id.values))
teachers_test = list(set(test.teacher_id.values))
inter = set(teachers_train).intersection(teachers_test)

char_cols = ['project_subject_categories', 'project_subject_subcategories',
       'project_title', 'project_essay_1', 'project_essay_2',
       'project_essay_3', 'project_essay_4', 'project_resource_summary']

resources['total_price'] = resources.quantity * resources.price

mean_total_price = pd.DataFrame(resources.groupby('id').total_price.mean())
sum_total_price = pd.DataFrame(resources.groupby('id').total_price.sum())
count_total_price = pd.DataFrame(resources.groupby('id').total_price.count())
mean_total_price['id'] = mean_total_price.index
sum_total_price['id'] = mean_total_price.index
count_total_price['id'] = mean_total_price.index


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


train = create_features(train)
test = create_features(test)

cat_features = ["teacher_prefix", "school_state", "year", "month", "project_grade_category", "project_subject_categories", "project_subject_subcategories"]
#"teacher_id",
num_features = ["teacher_number_of_previously_posted_projects", "total_price_x", "total_price_y", "total_price"]
cat_features_hash = [col+"_hash" for col in cat_features]

max_size=15000#0
def feature_hash(df, max_size=max_size):
    for col in cat_features:
        df[col+"_hash"] = df[col].apply(lambda x: hash(x)%max_size)
    return df

train = feature_hash(train)
test = feature_hash(test)

from sklearn.preprocessing import StandardScaler
#from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import text, sequence
import re

max_features = 100000#50000
maxlen = 300
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train[num_features])
X_test_num = scaler.transform(test[num_features])
X_train_cat = np.array(train[cat_features_hash], dtype=np.int)
X_test_cat = np.array(test[cat_features_hash], dtype=np.int)
tokenizer = text.Tokenizer(num_words=max_features)

def preprocess1(string):
    
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
    string = re.sub(r'\�', ' ', string)


    string = string.lower()  # If strings not lowercased
    #
    string = re.sub(r'\'re', ' are ', string)
    string = re.sub(r'n\'t', ' not ', string)
    string = re.sub(r'\'ve', ' have ', string)
    string = re.sub(r'\'m', ' am ', string)
    string = re.sub(r' u ', ' you ', string)
    string = re.sub(r' yr ', ' your ', string)
    string = re.sub(r' ca not', ' can not ', string)
    string = re.sub(r' wil ', ' will ', string)
    string = re.sub(r'dont', 'do not ', string)
    string = re.sub(r'didnt', 'do not ', string)
    string = re.sub(r'cant', 'can not ', string)
    string = re.sub(r'cannot', 'can not ', string)

    string = re.sub(r'don\'t', 'do not', string)
    string = re.sub(r'doesn\'t', 'do not', string)
    string = re.sub(r'didn\'t', 'do not', string)
    string = re.sub(r'hasn\'t', 'has not', string)
    string = re.sub(r'haven\'t', 'has not', string)
    string = re.sub(r'won\'t', 'will not', string)
    string = re.sub(r'wouldn\'t', 'will not', string)
    string = re.sub(r'can\'t', 'can not', string)
    string = re.sub(r'cannot', 'can not', string)
    string = re.sub(r'i\'m', 'i am', string)
    string = re.sub(r'i\'ll', 'i will', string)
    string = re.sub(r'you\'ll', 'you will', string)
    string = re.sub(r'we\'ll', 'we will', string)
    string = re.sub(r'it\'s', 'it is', string)
    string = re.sub(r'its', 'it is', string)


    return string

train["text"]=train["text"].apply(preprocess1)
test["text"]=test["text"].apply(preprocess1)
train_ids = train['id']
test_ids = test['id']


from common import punctuationWithSpace
print("before do_punctuation:",train['text'][0])
train["text"]=train["text"].apply(punctuationWithSpace)
test["text"]=test["text"].apply(punctuationWithSpace)
print("after do_punctuation:",train['text'][0])

tokenizer.fit_on_texts(train["text"].tolist()+test["text"].tolist())
list_tokenized_train = tokenizer.texts_to_sequences(train["text"].tolist())
list_tokenized_test = tokenizer.texts_to_sequences(test["text"].tolist())
X_train_words = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test_words = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


X_train_target = train.project_is_approved

EMBEDDING_FILE = 'E:/kaggle/toxicComment/toxic-master/crawl-300d-2M.vec'
embed_size=300
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Convolution1D, \
    GlobalMaxPool1D, SpatialDropout1D, CuDNNGRU, Bidirectional, PReLU,GlobalAvgPool1D
from keras.models import Model, Layer
from keras import optimizers

from keras.layers import Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D

from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def get_model3_v3():
    input_cat = Input((len(cat_features_hash),))
    input_num = Input((len(num_features),))
    input_words = Input((maxlen,))

    x_cat = Embedding(max_size, 10)(input_cat)

    x_cat = SpatialDropout1D(0.3)(x_cat)
    x_cat = Flatten()(x_cat)

    x_words = Embedding(max_features, 300,
                        weights=[embedding_matrix],
                        trainable=False)(input_words)
    x_words = SpatialDropout1D(0.3)(x_words)
    x_words1 = Bidirectional(CuDNNGRU(50, return_sequences=True))(x_words)  # 50

    attenion = Attention(maxlen)(x_words1)
    gl=GlobalMaxPool1D()(x_words1)
    gl_aver=GlobalAvgPool1D()(x_words1)

    x_cat = Dense(100, kernel_initializer='he_normal')(x_cat)
    x_cat = PReLU()(x_cat)
    x_num = Dense(100, kernel_initializer='he_normal')(input_num)
    x_num = PReLU()(x_num)

    x = concatenate([x_cat, x_num, attenion,gl,gl_aver])

    x = Dense(50, kernel_initializer='he_normal')(x)
    x = PReLU()(x)
    x = Dropout(0.25)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_cat, input_num, input_words], outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model



from sklearn.model_selection import KFold
from keras.callbacks import *
n_folds = 5
kf = KFold(n_splits=n_folds, random_state=1234, shuffle=False)
i = 0
loss_total = 0
acc_total = 0
pred_test=0.
x_test=[X_test_cat, X_test_num, X_test_words]
for index_train, index_valid in kf.split(train['total_price_x']):
    print ("Running Fold: {}".format(i))
    x_train = [X_train_cat[index_train], X_train_num[index_train], X_train_words[index_train]]
    y_train = X_train_target[index_train]
    x_valid = [X_train_cat[index_valid], X_train_num[index_valid], X_train_words[index_valid]]
    y_valid = X_train_target[index_valid]
    file_path = 'weights/simpleRNN_attention_v3_{}.h5'.format(i)
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
    model = get_model3_v3()

    history = model.fit([X_train_cat, X_train_num, X_train_words], X_train_target,
                        validation_data=(x_valid,y_valid),
                        verbose=2,callbacks=callbacks_list,
              epochs=20, batch_size=256)#epochs=100 overfitting too much

    model.load_weights(file_path)
    loss, acc = model.evaluate(x_valid, y_valid, verbose=0)
    loss_total += loss
    acc_total += acc

    

    print('start to predict on test')
    preds = model.predict(x_test, batch_size=2000)


    pred_test += preds

    i += 1

print ("Avg loss = {}, avg acc = {}".format(loss_total/n_folds, acc_total/n_folds))
test["project_is_approved"] = pred_test/n_folds

test[['id', 'project_is_approved']].to_csv("gru_attention_v3_subm_5fold.csv", index=False)
'''
'''
epochs=20 ,lb 0.81298
Avg loss = 0.30144214344412246, avg acc = 0.8772297891036906


'''

