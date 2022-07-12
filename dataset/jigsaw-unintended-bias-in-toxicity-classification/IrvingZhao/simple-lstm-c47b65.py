#differnt batch size

import numpy as np
import pandas as pd
import gc
import pickle
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNGRU,CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler


EMBEDDING_FILES = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]
NUM_MODELS = 2
#BATCH_SIZE = 1024
BATCH_SIZE = 256
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32') #Convert the input to an array.


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path) 
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix

def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]

def build_model(embedding_matrix, num_aux_targets, loss_weight):
    words = Input(shape=(MAX_LEN,))# return tensor  instantiate a Keras tensor
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    #x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    # x = Bidirectional(CuDNNGRU(GRU_UNITS, return_sequences=True))(x) 
    # x = Bidirectional(CuDNNGRU(GRU_UNITS, return_sequences=True))(x) 
    

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    dense = Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(dense)])
    dense1 = Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(dense1)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result, aux_result])
    
    #model.compile(loss='binary_crossentropy', optimizer='Adagrad')
    model.compile(loss=[custom_loss,'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer='adam')
    return model
    

def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv').head(1000)
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv').head(1000)

#loss
TOXICITY_COLUMN = 'target'
identity_columns = ['asian', 'atheist',
    'bisexual', 'black', 'buddhist', 'christian', 'female',
    'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',
    'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
    'muslim', 'other_disability', 'other_gender',
    'other_race_or_ethnicity', 'other_religion',
    'other_sexual_orientation', 'physical_disability',
    'psychiatric_or_mental_illness', 'transgender', 'white']
    

subgroup_bool_train = train[identity_columns].fillna(0)>=0.5
toxic_bool_train = train[TOXICITY_COLUMN].fillna(0)>=0.5
subgroup_negative_mask = subgroup_bool_train.values.sum(axis=1).astype(bool) & ~toxic_bool_train
# Overall
weightss = np.ones((len(train),)) #int

# Subgroup negative
weightss += subgroup_negative_mask

loss_weight = 1.0 / weightss.mean()

x_train = preprocess(train['comment_text'])
#y_train = np.where(train['target'] >= 0.5, 1, 0)
y_train = np.vstack([(train['target'].values>=0.5).astype(np.int),weightss]).T
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
x_test = preprocess(test['comment_text'])

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)#sentence to words
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)#fill sentence to MAX_LEN
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1) #tokenizer.word_index: words to index
    
checkpoint_predictions = []
weights = []

submission_id = test['id']

with open('temporary.pickle', mode='wb') as f:
    pickle.dump(x_test, f) # use temporary file to reduce memory

del identity_columns, weightss, tokenizer, train, test
gc.collect()



for model_idx in range(NUM_MODELS):
    model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weight)
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            [y_train, y_aux_train],
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=2,
            callbacks=[
                LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))
            ]
        )
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        weights.append(2 ** global_epoch)

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

submission = pd.DataFrame.from_dict({
    'id': submission_id,
    'prediction': predictions
})

submission.to_csv('submission.csv', index=False)