import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.preprocessing import sequence, text
from gensim.models import KeyedVectors

#embeddings 
EMBEDDING_FILES = [
    '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim',
    '../input/gensim-embeddings-dataset/glove.840B.300d.gensim'
]


#hyper-parameter 
num_nodes = 128
batch_size = 256
num_unrolling = 50
dropout = 0.2
MAX_LEN = 220
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = num_nodes  * 4

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]

def build_model(matrix_embed):
    words = Input(shape=(None,))

    x = Embedding(*matrix_embed.shape, weights=[matrix_embed], trainable=False)(words)
    x = Bidirectional(LSTM(num_nodes, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def build_embed_matrix(embed, tokens):
    embed_vec = KeyedVectors.load(embed, mmap='r')
    embed_matrix = np.zeros((len(tokens) + 1,300))
    for word, i in tokens.items():
        for candidate in [word, word.lower()]:
            if candidate in embed_vec:
                embed_matrix[i]= embed_vec[candidate]
    return embed_matrix



#load code 
train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

tokenizer = text.Tokenizer(
    num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)

aux_columns = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

x_train = train_df['comment_text'].astype(str)
y_train = train_df['target'].values
y_aux_train = train_df[aux_columns].values

x_test = test_df['comment_text'].astype(str)
tokenizer.fit_on_texts(list(x_train) + list(x_test))

for column in  identity_columns + ['target']:
    train_df[column] = np.where(train_df[column]>= 0.5, True, False)
    
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

sample_weights = np.ones(len(x_train), dtype=np.float32)
sample_weights += train_df[identity_columns].sum(axis=1)
sample_weights += train_df['target'] * (~train_df[identity_columns]).sum(axis=1)
sample_weights += (~train_df['target']) * train_df[identity_columns].sum(axis=1) * 5
sample_weights /= sample_weights.mean()

embed_matrix = build_embed_matrix('../input/gensim-embeddings-dataset/crawl-300d-2M.gensim', tokenizer.word_index)

model = build_model(embed_matrix)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=1,
    verbose=2
)
predictions = model.predict(x_test, batch_size=2048)
submission = pd.DataFrame({'id': test_df.id, 'prediction': list(predictions.flatten())})
submission.to_csv('submission.csv', index=False)