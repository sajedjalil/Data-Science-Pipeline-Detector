"""
Problem: How in the world am I supposed to do sufficent text cleaning in
Russian when I can barely do it in english? 

Solution: Character level embeddings!

I've tried to trained CNN based text analysis in the main model but
it seems to overfit a lot on LB

This model tries to predict the next character, although word2vec
and other word embedding models try to predict a missing middle word
"""
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Activation
from keras.layers.merge import concatenate
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Nadam, RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import TimeDistributed as TD
from keras.layers import CuDNNGRU as LSTM

data_dir = "../input"

print("Importing Data")
train_data = pd.read_csv(data_dir+"/train.csv", parse_dates=["activation_date"])

text_data = train_data.description.as_matrix()
text_data = [str(line) for line in text_data if len(str(line))>=50]#lets filter out low-character lines
#reduces train to ~1M
del train_data

print("Tokenizing")
n_char = 150 #Started at 200, but it included about 50 emoji-like characters
char_tk = Tokenizer(num_words=n_char, char_level=True, lower=False, filters=None)
char_tk.fit_on_texts(text_data)

max_len = 100 #This is pretty tunable
vocab_size = n_char+1
x_data = char_tk.texts_to_sequences(text_data)
x_data = pad_sequences(x_data, max_len, padding="post", truncating="post", value=0)
#Typically both padding and trunc are pre, but ehhhh

print("Every day I'm shufffff--falin")
shuffle = np.arange(len(x_data))
np.random.shuffle(shuffle)
x_data = x_data[shuffle]

print("Generating y-data")
y_data = np.zeros_like(x_data)
y_data[:,:-1] = x_data[:,1:]#next char shifting
y_data = np.expand_dims(y_data, axis=-1)
#we don't want to involve padded values in loss or accuracy
loss_weights = 1 - (x_data==0) #if x-data is 0, do not count that time-step

##================Begin Model Stuff
def build_model(text_len, n_chars, emb_size, opt):
    print("Building Model")
    inp_text = Input(shape=(text_len,))
    x0 = Embedding(vocab_size, emb_size, mask_zero=False)(inp_text)
    
    x = LSTM(32, return_sequences=True)(x0)
    x1 = Activation("relu")(x)
    x = LSTM(64, return_sequences=True)(x1)
    x2 = Activation("relu")(x)
    
    x = concatenate([x0, x1, x2])#"Skip connections"
    y = TD(Dense(vocab_size, activation="softmax"))(x)
    
    model = Model(inputs=[inp_text],outputs=y)

    if opt=="sgd" :optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    if opt=="rms" :optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    if opt=="adam" :optimizer= Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if opt=="nadam" :optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
    #note: there seems to be updates to how keras deals with sample weighting
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
              sample_weight_mode="temporal",weighted_metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, x_train, y_train, batch_size,model_name):
    earlystop = EarlyStopping(monitor="val_loss",mode="auto",patience=4,verbose=0)
    checkpt = ModelCheckpoint(monitor="val_loss",mode="auto",filepath=model_name,verbose=0,save_best_only=True)
    hist = model.fit(x_train, y_train,batch_size=batch_size,validation_split=0.15,
              epochs=1000, verbose=1,callbacks =[checkpt, earlystop],sample_weight=loss_weights)
    return hist

model = build_model(max_len, vocab_size, 16, "rms")
train_model(model, x_data, y_data, 128, "next_char_prediction_lstm.hdf5")