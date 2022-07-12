# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import datetime
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import CuDNNLSTM, CuDNNGRU, LSTM, Dense, Embedding, Bidirectional
from keras import Sequential
from keras.models import clone_model
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


train_nrows = None
test_nrows = None
max_length = 65
embedding_size = 300
dictionary_size = 120000
threshold = 0.33
num_models = 11
num_models_best = 11
num_epochs = 5


def get_time():
    return '[' + datetime.datetime.now().strftime("%H:%M:%S") + ']'


def multireplace(text, replacements):
    substrs = sorted(replacements, key=len, reverse=True)
    for p in substrs:
        text = text.replace(p, replacements[p])
    return text


def preprocess_text(text):
    if text.startswith('"'):
        text = text[1:]
    if text.endswith('"'):
        text = text[:-1]
    if text.endswith('?'):
        text = text[:-1]

    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
         
    contraction_mapping = {"what's":"what is", "What's":"What is", "i'm":"i am", "I'm":"i am", "isn't":"is not", "Isn't":"is not", "i've":"i have", "I've":"i have", "you've":"you have", "aren't":"are not", "Aren't":"are not", "won't":"will not", "Won't":"will not", "they're":"they are", "They're":"they are", "he's":"he is", "He's":"he is", "haven't":"have not", "shouldn't":"should not", "Shouldn't":"should not", "wouldn't":"would not", "Wouldn't":"would not", "who's":"who is", "Who's":"who is", "there's":"there is", "There's":"there is", "wasn't":"was not", "Wasn't":"was not", "she's":"she is", "hasn't":"has not", "Hasn't":"has not", "couldn't":"could not", "we're":"we are", "We're":"we are", "i'll":"i will", "I'll":"i will", "i'd":"i would", "I'd":"i would", "how's":"how is", "How's":"how is", "let's":"let us", "Let's":"let us", "weren't":"were not", "Weren't":"were not", "they've":"they have", "we've":"we have", "We've":"we have", "hadn't":"had not", "Hadn't":"had not", "you'd":"you would", "where's":"where is", "Where's":"where is", "'the":"the", "'The":"the", "'i":"i", "'I":"i", "would've":"would have", "“the":"the", "“The":"the", "“i":"i", "“I":"i","they'll":"they will", "They'll":"they will", "he'll":"he will", "He'll":"he will", "they'd":"they would", "They'd":"they would", "you'":"you ", "etc…":"etc ", "couldn't":"could not", "Couldn't":"could not", "it'll":"it will", "he'd":"he would", "could've":"could have", "Could've":"could have", "we'll":"we will"}
    text = multireplace(text, contraction_mapping)

    posessive_mappings = {"Trump's":"trump", "trump's":"trump", "Obama's":"obama", "obama's":"obama", "Google's":"google", "google's":"google", "India's":"india", "india's":"india", "Russia's":"russia", "russia's":"russia", "Israel's":"israel", "israel's":"israel", "Korea's":"korea", "korea's":"korea", "China's":"china", "china's":"china", "America's":"america", "america's":"america", "canada's":"canada", "Canada's":"canada", "pakistan's":"pakistan", "Pakistan's":"pakistan", "iran's":"iran", "Iran's":"iran", "japan's":"japan", "Japan's":"japan", "UK's":"uk", "uk's":"uk", "britain's":"britain", "Britain's":"britain", "usa's":"usa", "USA's":"usa", "germany's":"germany", "Germany's":"germany", "australia's":"australia", "Australia's":"australia", "someone's":"someone", "else's":"else", "today's":"today", "people's":"people", "women's":"women", "men's":"men", "world's":"world", "earth's":"earth", "Earth's":"earth", "country's":"country", "person's":"person", "quora's":"quora", "Quora's":"quora", "man's":"man", "woman's":"woman", "God's":"God", "company's":"company", "father's":"father", "mother's":"mother", "child's":"child", "girl's":"girl", "boy's":"boy", "wife's":"wife", "husband's":"husband", "year's":"year", "dog's":"dog", "friend's":"friend", "children's":"children", "driver's":"driver", "government's":"government", "everyone's":"everyone", "girlfriend's":"girlfriend", "boyfriend's":"boyfriend", "other's":"other", "modi's":"modi", "Modi's":"modi", "son's":"son", "daughter's":"daughter", "sister's":"sister", "cat's":"cat", "asperger's":"asperger", "Asperger's":"asperger", "alzheimer's":"alzheimer", "Alzheimer's":"alzheimer", "jehovah's":"jehovah", "Jehovah's":"jehovah", "einstein's":"einstein", "Einstein's":"einstein", "clinton's":"clinton", "Clinton's":"clinton", "king's":"king", "life's":"life", "parents'":"parents", "hitler's":"hitler", "Hitler's":"hitler", "newton's":"newton", "Newton's":"newton", "amazon's":"amazon", "Amazon's":"amazon", "xavier's":"xavier", "Xavier's":"xavier", "king's":"king", "King's":"king", "university's":"university", "University's":"university", "student's":"student", "Putin's":"putin", "putin's":"putin", "mom's":"mom", "baby's":"baby", "guy's":"guy", "president's":"president", "President's":"president", "parent's":"parent", "partner's":"partner", "dad's":"dad", "facebook's":"facebook", "Facebook's":"facebook", "doctor's":"doctor", "car's":"car", "others'":"others", "countries'":"countries", "school's":"school", "family's":"family", "nation's":"nation", "people's":"people", "People's":"people", "party's":"party", "Party's":"party", "jesus's":"jesus", "Jesus's":"jesus"}
    text = multireplace(text, posessive_mappings)

    meaning_mapping = {"quorans":"quora", "Quorans":"quora", "90's":"90s", "80's":"80s", "70's":"70s", "30's":"30s", "20's":"20s", "Brexit":"britain exit", "brexit":"britain exit", "master's":"masters", "Master's":"masters", "mcdonald's":"mcdonalds", "McDonald's":"mcdonalds", "one's":"someone", "cryptocurrencies":"bitcoin", "bachelor's":"bachelors", "Bachelor's":"bachelors", "demonetisation":"demonetization", "qoura":"quora", "Qoura":"quora", "Qur'an":"quran", "qur'an":"quran", "sjws":"sjw", "SJWs":"sjw", "valentine's":"valentines", "Valentine's":"valentines", "anysomeone":"someone", "Anysomeone":"someone"}
    text = multireplace(text, meaning_mapping)

    return text


def preprocess(df):
    print(get_time(), "Pre-processing started...")
    df["question_text"] = df["question_text"].apply(lambda x: preprocess_text(x))
    print(get_time(), "Pre-processing finished")


def init_tokenizer(filepath):
    global dictionary_size, train_nrows
    print(get_time(), "Training data load started...")
    if train_nrows is not None:
        df = pd.read_csv(filepath, nrows=train_nrows)
    else:
        df = pd.read_csv(filepath)
    print(get_time(), "Training data loaded:", df.shape[0])
    preprocess(df)

    print(get_time(), "Tokenizer start...")
    texts = df['question_text'].values
    tokenizer = Tokenizer(num_words=dictionary_size)
    tokenizer.fit_on_texts(texts)
    print(get_time(), "Tokenizer fit. Total words:", len(tokenizer.word_index))
    return tokenizer, df


def get_sequences(tokenizer, df, with_targets=False):
    global max_length
    texts = df['question_text'].values
    print(get_time(), "Converting text to sequences start...")
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    print(get_time(), "Converted text to sequences:", len(sequences))

    targets = None
    if with_targets:
        targets = df['target'].values

    return sequences, targets


def init(filepath_data_train, filepath_data_test):
    global test_nrows, dictionary_size
    tokenizer, df_train = init_tokenizer(filepath_data_train)
    sequences_train, targets_train = get_sequences(tokenizer, df_train, with_targets=True)
  
    if test_nrows is not None:
        df_test = pd.read_csv(filepath_data_test, nrows=test_nrows)
    else: 
        df_test = pd.read_csv(filepath_data_test)
    preprocess(df_test)
    sequences_test, _ = get_sequences(tokenizer, df_test, with_targets=False)

    # shuffle training data
    shuffled_indexes = np.random.permutation(len(sequences_train))
    sequences_train = sequences_train[shuffled_indexes]
    targets_train = targets_train[shuffled_indexes]

    print(get_time(), 'Data initialization done. Word_index:', len(tokenizer.word_index), 'Dictionary size:', dictionary_size)
    return sequences_train, targets_train, sequences_test, df_test, tokenizer.word_index


def load_embeddings_glove(filepath, word_index):
    print(get_time(), "Start load embeddings (glove)...")
    embedding_dict = {}
    for line in open(filepath, encoding='utf8'):
        split = line.split(' ')
        word = split[0]
        if len(split) == 301 and word in word_index:
            embedding_dict[word] = np.array(split[1:], dtype=np.float32)
    print(get_time(), "Finished load embeddings (glove):", len(embedding_dict))
    return embedding_dict


def load_embeddings_paragram(filepath, word_index):
    print(get_time(), "Start load embeddings (paragram)...")
    embedding_dict = {}
    for line in open(filepath, encoding="utf8", errors='ignore'):
        split = line.split(' ')
        word = split[0]
        if len(split) == 301 and word in word_index:
            embedding_dict[word] = np.array(split[1:], dtype=np.float32)
    print(get_time(), "Finished load embeddings (paragram):", len(embedding_dict))
    return embedding_dict


def load_embeddings_wiki(filepath, word_index):
    print(get_time(), "Start load embeddings (wikinews)...")
    embedding_dict = {}
    for line in open(filepath, encoding="utf8", errors='ignore'):
        split = line.split(' ')
        word = split[0]
        if len(split) == 301 and word in word_index:
            embedding_dict[word] = np.array(split[1:], dtype=np.float32)
    print(get_time(), "Finished load embeddings (wikinews):", len(embedding_dict))
    return embedding_dict
    

def load_embeddings_google(filepath, word_index):
    print(get_time(), "Start load embeddings (google)...")
    from gensim.models import KeyedVectors
    embedding_dict = {}
    word2vec = KeyedVectors.load_word2vec_format(filepath, binary=True) 
    for word, vector in zip(word2vec.vocab, word2vec.vectors):
        if word in word_index:
            embedding_dict[word] = np.array(vector, dtype=np.float32)
    print(get_time(), "Finished load embeddings (google):", len(embedding_dict))
    return embedding_dict    
    

def get_embeddings(embedding_dict, word_index):
    global dictionary_size
    print('Embedding dict values:', type(embedding_dict.values()))
    embedding_values = np.stack(list(embedding_dict.values()))
    print('Embedding values:', type(embedding_values), embedding_values.shape)
    emb_mean,emb_std = embedding_values.mean(), embedding_values.std()
    embedding_size = embedding_values.shape[1]

    # missing words: fill from random distribution, should be better than zeros
    embedding_matrix = np.random.normal(emb_mean, emb_std, (dictionary_size, embedding_size))
    for word, i in word_index.items():
        if i >= dictionary_size: continue
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
            
    print(get_time(), "Embedding metrix done. Shape:", embedding_matrix.shape)
    return embedding_matrix


def get_embeddings_matrices(filepath_glove, filepath_paragram, filepath_wiki, filepath_google, word_index):
    glove = load_embeddings_glove(filepath_glove, word_index)
    paragram = load_embeddings_paragram(filepath_paragram, word_index)
    #wiki = load_embeddings_wiki(filepath_wiki, word_index)
    #google = load_embeddings_google(filepath_google, word_index)
    
    matrix_glove = get_embeddings(glove, word_index)
    matrix_paragram = get_embeddings(paragram, word_index)
    #matrix_wiki = get_embeddings(wiki, word_index)
    #matrix_google = get_embeddings(google, word_index)
    
    return [matrix_glove, matrix_paragram]


def get_embeddings_average(embeddings_matrices):
    # average is almost as good as concatenation, sometimes better, and reduced dimension
    # http://aclweb.org/anthology/N18-2031
    # https://arxiv.org/pdf/1804.07983.pdf
    matrix_average = np.mean(embeddings_matrices, axis = 0)
    return matrix_average


def get_model_lstm(embedding_matrix):
    global vector_size, max_length, dictionary_size
    model = Sequential()
    model.add(Embedding(dictionary_size, embedding_size, 
                        weights=[embedding_matrix], 
                        trainable=False, 
                        input_length=max_length))
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
    model.add(Bidirectional(CuDNNLSTM(64)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_model_gru(embedding_matrix):
    global vector_size, max_length, dictionary_size
    model = Sequential()
    model.add(Embedding(dictionary_size, embedding_size, 
                        weights=[embedding_matrix], 
                        trainable=False, 
                        input_length=max_length))
    model.add(Bidirectional(CuDNNGRU(64, return_sequences=True)))
    model.add(Bidirectional(CuDNNGRU(64)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, train_X, train_y, val_X, val_y, test_X, epochs=3, callback=None):
    global threshold
    # manually loop through epochs to calculate validation F1 after each epoch
    # manually implement early stopping using F1
    #for e in range(epochs):
    e = 0
    no_improve_count = 0
    #best_model = model
    best_val_f1 = 0.
    while no_improve_count < 3:
        print(get_time(), 'Epoch ' + str(e+1) + ' started...')
        h = model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks = callback, verbose=0)
        predicted_val = model.predict([val_X], batch_size=1024, verbose=0)

        val_f1 = f1_score(val_y, (predicted_val > threshold).astype(int))
        print(get_time(), 'Epoch ' + str(e+1) + ' completed. ' + 
                          'Train: {:.4f}, '.format(h.history['acc'][0]) + 
                          'Valid: {:.4f}, '.format(h.history['val_acc'][0]) + 
                          'Valid F1: {:.4f}'.format(val_f1))
        e += 1
        if val_f1 < best_val_f1:
            no_improve_count += 1
            print(get_time(), 'No improvement: {:.4f}'.format(val_f1) + ' < {:.4f}'.format(best_val_f1))
        else:
            no_improve_count = 0
            #best_model = clone_model(model)
            #best_model.set_weights(model.get_weights())
            best_val_f1 = val_f1

    predicted_test = model.predict([test_X], batch_size=1024, verbose=0)
    return predicted_test, best_val_f1


def do_train(embedding_matrices, train_X, train_y, test_X):
    global num_models, num_epochs, num_models_best
    predicted_list = []
    predicted_test = np.zeros(test_X.shape[0])
    #splits = list(StratifiedKFold(n_splits=1, shuffle=True, random_state=7).split(train_X, train_y))
    splitter = StratifiedShuffleSplit(n_splits=num_models, test_size=0.05, train_size=None, random_state=7)
    splits = list(splitter.split(train_X, train_y))
    for idx, (train_idx, valid_idx) in enumerate(splits):
        print(get_time(), 'Model #' + str(idx+1) + ' of ' + str(num_models) + ' started...')
        X_train = train_X[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X[valid_idx]
        y_val = train_y[valid_idx]
        
        embedding_matrix = embedding_matrices[idx % len(embedding_matrices)]
        #model = get_model_lstm(embedding_matrix) if idx % 2 == 0 else get_model_gru(embedding_matrix)
        #embedding_matrix = get_embeddings_average(embedding_matrices)
        model = get_model_lstm(embedding_matrix)
        predicted, val_f1 = train_model(model,
                                        X_train, y_train, X_val, y_val, test_X, 
                                        epochs=num_epochs, 
                                        callback=None)
        if len(predicted_list) < num_models_best:
            predicted_list.append((predicted[:,0], val_f1))
        elif val_f1 > predicted_list[0][1]:
            predicted_list[0] = (predicted[:,0], val_f1)
        predicted_list = sorted(predicted_list, key=lambda x: x[1])

        print(get_time(), 'Model #' + str(idx+1) + ' of ' + str(num_models) + ' completed')

    print(get_time(), 'Best models:', [f1 for _,f1 in predicted_list])
    for predicted,_ in predicted_list:
        predicted_test += predicted
    predicted_test /= len(predicted_list)
    return predicted_test


def do_submission(df_test, predicted_test, filepath):
    global threshold
    print(get_time(), 'Start submission...')
    df_test['prediction'] = (predicted_test > threshold).astype(int)
    df_test.to_csv(filepath, columns=['qid','prediction'], index=False)
    print(get_time(), 'Completed submission')
    


dirpath = os.path.realpath('../input')
filepath_data_train = os.path.join(dirpath, 'train.csv')
filepath_data_test = os.path.join(dirpath, 'test.csv')
filepath_embeddings_glove = os.path.join(dirpath, 'embeddings/glove.840B.300d/glove.840B.300d.txt')
filepath_embeddings_paragram = os.path.join(dirpath, 'embeddings/paragram_300_sl999/paragram_300_sl999.txt')
filepath_embeddings_wiki = os.path.join(dirpath, 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
filepath_embeddings_google = os.path.join(dirpath, 'embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin')

sequences_train, targets_train, sequences_test, df_test, word_index = init(filepath_data_train, filepath_data_test)
embedding_matrices = get_embeddings_matrices(filepath_embeddings_glove, filepath_embeddings_paragram, filepath_embeddings_wiki, filepath_embeddings_google, word_index)
predicted_test = do_train(embedding_matrices, sequences_train, targets_train, sequences_test)
do_submission(df_test, predicted_test, 'submission.csv')
