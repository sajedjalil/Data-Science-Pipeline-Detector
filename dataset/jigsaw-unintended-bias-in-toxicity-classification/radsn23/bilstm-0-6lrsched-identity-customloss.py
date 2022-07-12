# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from keras.preprocessing import text, sequence
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
#from keras.losses import binary_crossentropy
#from keras import backend as K
# Any results you write to the current directory are saved as output.
EMBEDDING_FILES = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
    ]


NUM_MODELS = 2 #??
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4*LSTM_UNITS #??
EPOCHS = 4
MAX_LEN = 220
# IDENTITY_COLUMNS = [
#     'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
#     'muslim', 'black', 'psychiatric_or_mental_illness'
#     ]
# 
IDENTITY_COLUMNS = [
    'transgender', 'female', 'homosexual_gay_or_lesbian', 'muslim', 'hindu',
    'white', 'black', 'psychiatric_or_mental_illness', 'jewish'
    ]  
    #--- 0.9331
# IDENTITY_COLUMNS = [
#     'transgender', 'female', 'homosexual_gay_or_lesbian', 'muslim', 'hindu',
#     'white', 'black', 'psychiatric_or_mental_illness', 'jewish'
#     ]
AUX_COLUMNS = ['target', 'severe_toxicity','obscene','identity_attack','insult','threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

#Following swear words obtained from 
# https://github.com/RobertJGabriel/Google-profanity-words/blob/master/list.txt
swear_words = [
    ' 4r5e ',
    ' 5h1t ',
    ' 5hit ',
    ' a55 ',
    ' anal ',
    ' anus ',
    ' ar5e ',
    ' arrse ',
    ' arse ',
    ' ass ',
    ' ass-fucker ',
    ' asses ',
    ' assfucker ',
    ' assfukka ',
    ' asshole ',
    ' assholes ',
    ' asswhole ',
    ' a_s_s ',
    ' b!tch ',
    ' b00bs ',
    ' b17ch ',
    ' b1tch ',
    ' ballbag ',
    ' balls ',
    ' ballsack ',
    ' bastard ',
    ' beastial ',
    ' beastiality ',
    ' bellend ',
    ' bestial ',
    ' bestiality ',
    ' biatch ',
    ' bitch ',
    ' bitcher ',
    ' bitchers ',
    ' bitches ',
    ' bitchin ',
    ' bitching ',
    ' bloody ',
    ' blow job ',
    ' blowjob ',
    ' blowjobs ',
    ' boiolas ',
    ' bollock ',
    ' bollok ',
    ' boner ',
    ' boob ',
    ' boobs ',
    ' booobs ',
    ' boooobs ',
    ' booooobs ',
    ' booooooobs ',
    ' breasts ',
    ' buceta ',
    ' bugger ',
    ' bum ',
    ' bunny fucker ',
    ' butt ',
    ' butthole ',
    ' buttmuch ',
    ' buttplug ',
    ' c0ck ',
    ' c0cksucker ',
    ' carpet muncher ',
    ' cawk ',
    ' chink ',
    ' cipa ',
    ' cl1t ',
    ' clit ',
    ' clitoris ',
    ' clits ',
    ' cnut ',
    ' cock ',
    ' cock-sucker ',
    ' cockface ',
    ' cockhead ',
    ' cockmunch ',
    ' cockmuncher ',
    ' cocks ',
    ' cocksuck ',
    ' cocksucked ',
    ' cocksucker ',
    ' cocksucking ',
    ' cocksucks ',
    ' cocksuka ',
    ' cocksukka ',
    ' cok ',
    ' cokmuncher ',
    ' coksucka ',
    ' coon ',
    ' cox ',
    ' crap ',
    ' cum ',
    ' cummer ',
    ' cumming ',
    ' cums ',
    ' cumshot ',
    ' cunilingus ',
    ' cunillingus ',
    ' cunnilingus ',
    ' cunt ',
    ' cuntlick ',
    ' cuntlicker ',
    ' cuntlicking ',
    ' cunts ',
    ' cyalis ',
    ' cyberfuc ',
    ' cyberfuck ',
    ' cyberfucked ',
    ' cyberfucker ',
    ' cyberfuckers ',
    ' cyberfucking ',
    ' d1ck ',
    ' damn ',
    ' dick ',
    ' dickhead ',
    ' dildo ',
    ' dildos ',
    ' dink ',
    ' dinks ',
    ' dirsa ',
    ' dlck ',
    ' dog-fucker ',
    ' doggin ',
    ' dogging ',
    ' donkeyribber ',
    ' doosh ',
    ' duche ',
    ' dyke ',
    ' ejaculate ',
    ' ejaculated ',
    ' ejaculates ',
    ' ejaculating ',
    ' ejaculatings ',
    ' ejaculation ',
    ' ejakulate ',
    ' f u c k ',
    ' f u c k e r ',
    ' f4nny ',
    ' fag ',
    ' fagging ',
    ' faggitt ',
    ' faggot ',
    ' faggs ',
    ' fagot ',
    ' fagots ',
    ' fags ',
    ' fanny ',
    ' fannyflaps ',
    ' fannyfucker ',
    ' fanyy ',
    ' fatass ',
    ' fcuk ',
    ' fcuker ',
    ' fcuking ',
    ' feck ',
    ' fecker ',
    ' felching ',
    ' fellate ',
    ' fellatio ',
    ' fingerfuck ',
    ' fingerfucked ',
    ' fingerfucker ',
    ' fingerfuckers ',
    ' fingerfucking ',
    ' fingerfucks ',
    ' fistfuck ',
    ' fistfucked ',
    ' fistfucker ',
    ' fistfuckers ',
    ' fistfucking ',
    ' fistfuckings ',
    ' fistfucks ',
    ' flange ',
    ' fook ',
    ' fooker ',
    ' fuck ',
    ' fucka ',
    ' fucked ',
    ' fucker ',
    ' fuckers ',
    ' fuckhead ',
    ' fuckheads ',
    ' fuckin ',
    ' fucking ',
    ' fuckings ',
    ' fuckingshitmotherfucker ',
    ' fuckme ',
    ' fucks ',
    ' fuckwhit ',
    ' fuckwit ',
    ' fudge packer ',
    ' fudgepacker ',
    ' fuk ',
    ' fuker ',
    ' fukker ',
    ' fukkin ',
    ' fuks ',
    ' fukwhit ',
    ' fukwit ',
    ' fux ',
    ' fux0r ',
    ' f_u_c_k ',
    ' gangbang ',
    ' gangbanged ',
    ' gangbangs ',
    ' gaylord ',
    ' gaysex ',
    ' goatse ',
    ' God ',
    ' god-dam ',
    ' god-damned ',
    ' goddamn ',
    ' goddamned ',
    ' hardcoresex ',
    ' hell ',
    ' heshe ',
    ' hoar ',
    ' hoare ',
    ' hoer ',
    ' homo ',
    ' hore ',
    ' horniest ',
    ' horny ',
    ' hotsex ',
    ' jack-off ',
    ' jackoff ',
    ' jap ',
    ' jerk-off ',
    ' jism ',
    ' jiz ',
    ' jizm ',
    ' jizz ',
    ' kawk ',
    ' knob ',
    ' knobead ',
    ' knobed ',
    ' knobend ',
    ' knobhead ',
    ' knobjocky ',
    ' knobjokey ',
    ' kock ',
    ' kondum ',
    ' kondums ',
    ' kum ',
    ' kummer ',
    ' kumming ',
    ' kums ',
    ' kunilingus ',
    ' l3itch ',
    ' labia ',
    ' lmfao ',
    ' lust ',
    ' lusting ',
    ' m0f0 ',
    ' m0fo ',
    ' m45terbate ',
    ' ma5terb8 ',
    ' ma5terbate ',
    ' masochist ',
    ' master-bate ',
    ' masterb8 ',
    ' masterbat3 ',
    ' masterbate ',
    ' masterbation ',
    ' masterbations ',
    ' masturbate ',
    ' mo-fo ',
    ' mof0 ',
    ' mofo ',
    ' mothafuck ',
    ' mothafucka ',
    ' mothafuckas ',
    ' mothafuckaz ',
    ' mothafucked ',
    ' mothafucker ',
    ' mothafuckers ',
    ' mothafuckin ',
    ' mothafucking ',
    ' mothafuckings ',
    ' mothafucks ',
    ' mother fucker ',
    ' motherfuck ',
    ' motherfucked ',
    ' motherfucker ',
    ' motherfuckers ',
    ' motherfuckin ',
    ' motherfucking ',
    ' motherfuckings ',
    ' motherfuckka ',
    ' motherfucks ',
    ' muff ',
    ' mutha ',
    ' muthafecker ',
    ' muthafuckker ',
    ' muther ',
    ' mutherfucker ',
    ' n1gga ',
    ' n1gger ',
    ' nazi ',
    ' nigg3r ',
    ' nigg4h ',
    ' nigga ',
    ' niggah ',
    ' niggas ',
    ' niggaz ',
    ' nigger ',
    ' niggers ',
    ' nob ',
    ' nob jokey ',
    ' nobhead ',
    ' nobjocky ',
    ' nobjokey ',
    ' numbnuts ',
    ' nutsack ',
    ' orgasim ',
    ' orgasims ',
    ' orgasm ',
    ' orgasms ',
    ' p0rn ',
    ' pawn ',
    ' pecker ',
    ' penis ',
    ' penisfucker ',
    ' phonesex ',
    ' phuck ',
    ' phuk ',
    ' phuked ',
    ' phuking ',
    ' phukked ',
    ' phukking ',
    ' phuks ',
    ' phuq ',
    ' pigfucker ',
    ' pimpis ',
    ' piss ',
    ' pissed ',
    ' pisser ',
    ' pissers ',
    ' pisses ',
    ' pissflaps ',
    ' pissin ',
    ' pissing ',
    ' pissoff ',
    ' poop ',
    ' porn ',
    ' porno ',
    ' pornography ',
    ' pornos ',
    ' prick ',
    ' pricks ',
    ' pron ',
    ' pube ',
    ' pusse ',
    ' pussi ',
    ' pussies ',
    ' pussy ',
    ' pussys ',
    ' rectum ',
    ' retard ',
    ' rimjaw ',
    ' rimming ',
    ' s hit ',
    ' s.o.b. ',
    ' sadist ',
    ' schlong ',
    ' screwing ',
    ' scroat ',
    ' scrote ',
    ' scrotum ',
    ' semen ',
    ' sex ',
    ' sh!t ',
    ' sh1t ',
    ' shag ',
    ' shagger ',
    ' shaggin ',
    ' shagging ',
    ' shemale ',
    ' shit ',
    ' shitdick ',
    ' shite ',
    ' shited ',
    ' shitey ',
    ' shitfuck ',
    ' shitfull ',
    ' shithead ',
    ' shiting ',
    ' shitings ',
    ' shits ',
    ' shitted ',
    ' shitter ',
    ' shitters ',
    ' shitting ',
    ' shittings ',
    ' shitty ',
    ' skank ',
    ' slut ',
    ' sluts ',
    ' smegma ',
    ' smut ',
    ' snatch ',
    ' son-of-a-bitch ',
    ' spac ',
    ' spunk ',
    ' s_h_i_t ',
    ' t1tt1e5 ',
    ' t1tties ',
    ' teets ',
    ' teez ',
    ' testical ',
    ' testicle ',
    ' tit ',
    ' titfuck ',
    ' tits ',
    ' titt ',
    ' tittie5 ',
    ' tittiefucker ',
    ' titties ',
    ' tittyfuck ',
    ' tittywank ',
    ' titwank ',
    ' tosser ',
    ' turd ',
    ' tw4t ',
    ' twat ',
    ' twathead ',
    ' twatty ',
    ' twunt ',
    ' twunter ',
    ' v14gra ',
    ' v1gra ',
    ' vagina ',
    ' viagra ',
    ' vulva ',
    ' w00se ',
    ' wang ',
    ' wank ',
    ' wanker ',
    ' wanky ',
    ' whoar ',
    ' whore ',
    ' willies ',
    ' willy ',
    ' xrated ',
    ' xxx '    
]

replace_with_fuck = []

for swear in swear_words:
    #if swear[1:(len(swear)-1)] not in embeddings_index:
    replace_with_fuck.append(swear)
        
replace_with_fuck = '|'.join(replace_with_fuck)


# define functions here
#def custom_loss(y_true, y_pred):
#    return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]

def handle_swears(text):
    text = re.sub(replace_with_fuck, ' fuck ', text)
    return text

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
    
def get_coefs(word, *arr):
    """
    Get word, word_embedding
    """
    return word, np.asarray(arr,dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)
        
def build_matrix(word_index, path):
    """
    Here we take each word we've tokenized in our text corpus
    for each word we look up in the pre-trained embedding.
    Each row in this matrix is a corpus word's embedding.
    """
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index)+1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix
        


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
        
    
def build_model(embedding_matrix, num_aux_targets):#, loss_weight):
    """
    embedding layer
    droput layer
    2 * bidirectional LSTM layers
    2 * pooling layers
    2 dense layers
    1 softmax layer
    """
    words = Input(shape=(MAX_LEN,)) 
    #Embedding layer takes variable size input
    x = Embedding(*embedding_matrix.shape, weights = [embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    
    #att = Attention(MAX_LEN)(x)
    hidden = concatenate([ 
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x)
        ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result =Dense(num_aux_targets, activation='sigmoid')(hidden)

    model = Model(inputs =words, outputs =[result, aux_result])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model
    
    

train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

train_df['comment_text'] = train_df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
test_df['comment_text'] = test_df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
gc.collect()
# train_df['comment_text'] = train_df['comment_text'].apply(lambda x: handle_swears(x))
# test_df['comment_text'] = test_df['comment_text'].apply(lambda x: handle_swears(x))
gc.collect()

x_train = train_df[TEXT_COLUMN].astype(str)
y_train = train_df[TARGET_COLUMN].values
y_aux_train = train_df[AUX_COLUMNS].values
x_test = test_df[TEXT_COLUMN].astype(str)

for column in IDENTITY_COLUMNS + [TARGET_COLUMN]: #??
    train_df[column] = np.where(train_df[column] >=0.5, True, False)

#Return a tokenizer class
tokenizer = text.Tokenizer(filters = CHARS_TO_REMOVE)
tokenizer.fit_on_texts(list(x_train)+ list(x_test))
# Turn text to sequences of tokens
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
#Pad sequences to the same length
x_train = sequence.pad_sequences(x_train,maxlen=MAX_LEN)
x_test= sequence.pad_sequences(x_test, maxlen=MAX_LEN)


# Initialize weights
sample_weights = np.ones(len(x_train), dtype=np.float32)
# Add all the values of the identities along rows
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
#Add all values of targets*~identity
sample_weights += train_df[TARGET_COLUMN]*(~train_df[IDENTITY_COLUMNS]).sum(axis=1)
#Add all values ~targets*identity
sample_weights += (~train_df[TARGET_COLUMN])*train_df[IDENTITY_COLUMNS].sum(axis=1)
#Normalize them
sample_weights/=sample_weights.mean()

embedding_matrix = np.concatenate([build_matrix(tokenizer.word_index,f) for f in EMBEDDING_FILES], axis =-1)
del train_df, tokenizer
gc.collect()

checkpoint_predictions = []
weights = []

for model_idx in range(NUM_MODELS):
    #Passes embedding matrix and aux outputs shape
    model = build_model(embedding_matrix, y_aux_train.shape[-1]) #1/sample_weights.mean())
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            [y_train, y_aux_train],
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=2,
            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],
            callbacks = [
                LearningRateScheduler(lambda _: 1e-3*(0.55**global_epoch))
                ]
        )
    
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        weights.append(2 ** global_epoch)
    del model
    gc.collect()
        
    predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
    submission = pd.DataFrame.from_dict({
        'id': test_df.id,
        'prediction': predictions
    })
    submission.to_csv('submission.csv', index=False)