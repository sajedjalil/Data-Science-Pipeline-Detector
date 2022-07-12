import numpy as np
import pandas as pd 
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import operator
from tqdm import tqdm
import gc

epochs=25
batch_size=128
max_words=100000
max_seq_size=256


train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_df  = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
sub_df   = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
train_df = train_df[["target", "comment_text"]]

# Use for combine the vector files that have given 
def combine_embedding(vec_files):
    
    # convert victor to float16 to make it use less memory
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float16')

    # make our embed smaller by get_coefs
    def optimize_embedding(embedding): 
        optimized_embedding = {}
        for word in embedding.vocab:
            optimized_embedding[word] = np.asarray(embedding[word], dtype='float16')
        return optimized_embedding

    
    # load embed vector from file
    def load_embed(file):
        print("Loading {}".format(file))

        if file == '../input/quoratextemb/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
            return dict(get_coefs(*o.strip().split(" ")) for o in open(file) if len(o) > 100)

        elif file == '../input/quoratextemb/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin':
            return optimize_embedding(KeyedVectors.load_word2vec_format(file, binary=True))
        
        elif file == '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec':
            return optimize_embedding(KeyedVectors.load_word2vec_format(file))

        else:
            return dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
        
    combined_embedding = {}
    for file in vec_files:
        combined_embedding.update(load_embed(file))
    return combined_embedding
    
    
vec_files = [
    "../input/quoratextemb/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec", 
    "../input/quoratextemb/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin",
    "../input/quoratextemb/embeddings/glove.840B.300d/glove.840B.300d.txt",
    "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
]

contraction_mapping = {
    "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 
    "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": 
    "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", 
    "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 
    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", 
    "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", 
    "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", 
    "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 
    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 
    "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", 
    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 
    "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have",
    "Trump's": "trump is", "Obama's": "obama is", "Canada's": "canada is", "today's": "today is"
}
known_contractions = ["ain't", "aren't", "can't", "'cause", "could've", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's", "how'd", "how's", "I'd", "I'd've", "I'll", "I'm", "I've", "isn't", "it'd", "it'll", "it's", "let's", "ma'am", "must've", "o'clock", "oughtn't", "she'd", "she'll", "she's", "should've", "shouldn't", "that's", "there's", "here's", "they'd", "they'll", "they're", "they've", "wasn't", "we'd", "we'll", "we're", "we've", "weren't", "what're", "what's", "what've", "where'd", "where's", "who'll", "who's", "who've", "won't", "would've", "wouldn't", "wouldn't've", "y'all", "you'd", "you'll", "you're", "you've", "Obama's", "today's"]
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
specail_signs = { "…": "...", "₂": "2"}
small_caps_mapping = { 
    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ғ": "f", "ɢ": "g", "ʜ": "h", "ɪ": "i", 
    "ᴊ": "j", "ᴋ": "k", "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ǫ": "q", "ʀ": "r", 
    "s": "s", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "x": "x", "ʏ": "y", "ᴢ": "z"
}

for cont in known_contractions:
    contraction_mapping.pop(cont)

def clean_contractions(text):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    words = [contraction_mapping[word] if word in contraction_mapping else word for word in text.split(" ")]
    return ' '.join(words)

def clean_special_chars(text):
    for s in specail_signs: 
        text = text.replace(s, specail_signs[s])
    for p in punct:
        text = text.replace(p, f' {p} ')
    return text

def clean_small_caps(text):
    for char in small_caps_mapping:
        text = text.replace(char, small_caps_mapping[char])
    return text
    
def clean_up_text_with_all_process(text):
    text = text.lower()
    text = clean_contractions(text)
    text = clean_special_chars(text)
    text = clean_small_caps(text)
    return text
    
train_df["comment_text"] = train_df["comment_text"].apply(lambda text: clean_up_text_with_all_process(text))
test_df["comment_text"] = test_df["comment_text"].apply(lambda text: clean_up_text_with_all_process(text))

tranformer = Tokenizer(lower = True, filters='', num_words=max_words)
tranformer.fit_on_texts( list(train_df["comment_text"].values) + list(test_df["comment_text"].values) )
transformed_x = tranformer.texts_to_sequences(train_df["comment_text"].values)
transformed_x = pad_sequences(transformed_x, maxlen = max_seq_size)
x_predict = tranformer.texts_to_sequences(test_df["comment_text"])
x_predict = pad_sequences(x_predict, maxlen = max_seq_size)

def build_embedding_matrix(word_index, total_vocab, embedding_size):
    embedding_index = combine_embedding(vec_files)
    matrix = np.zeros((total_vocab, embedding_size))
    for word, index in word_index.items():
        try:
            matrix[index] = embedding_index[word]
        except KeyError:
            pass
    return matrix
    
word_index = tranformer.word_index
total_vocab = len(word_index) + 1
embedding_size = 300
embedding_matrix = build_embedding_matrix(tranformer.word_index, total_vocab, embedding_size)


y = (train_df['target'].values > 0.5).astype(int)
x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, random_state=10, test_size=0.15)


from tensorflow.nn import relu, sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from tensorflow.keras.layers import CuDNNGRU, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D
sequence_input = Input(shape=(max_seq_size,), dtype='int32')
embedding_layer = Embedding(total_vocab,
                            embedding_size,
                            weights=[embedding_matrix],
                            input_length=max_seq_size,
                            trainable=False)

x_layer = embedding_layer(sequence_input)
x_layer = SpatialDropout1D(0.2)(x_layer)
x_layer = Bidirectional(CuDNNGRU(64, return_sequences=True))(x_layer)   
x_layer = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_layer)

avg_pool1 = GlobalAveragePooling1D()(x_layer)
max_pool1 = GlobalMaxPooling1D()(x_layer)     

x_layer = concatenate([avg_pool1, max_pool1])

preds = Dense(1, activation=sigmoid)(x_layer)

model = Model(sequence_input, preds)
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=callbacks)
y_predict = model.predict(x_predict)

sub_df["prediction"] = y_predict
sub_df.to_csv("submission.csv", index=False)

