# %% [code]
!pip install transformers==3.0.2
!pip install nlp

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nlp import load_dataset
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

np.random.seed(1234) 
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
mnli = load_dataset(path='glue', name='mnli') # loading more data from the Huggin face dataset
#snli   =  load_dataset("snli") # loading more data from the Huggin face dataset

# %% [code]
train_df = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')
print('Traning Data, the size of the dataset is: {} \n'.format(train_df.shape))

test_df = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')

# %% [code]
train_df = pd.concat([train_df]) #appending the original dataset to the additional datasets
train_df = train_df[train_df['label'] != -1] #cleaning values with the wrong label


print('the shape of the whole DF to be used is: ' + str(train_df.shape))

# %% [code]
# searching for duplicates

train_df = train_df[train_df.duplicated() == False]
print('the shape of the whole DF to be used is: ' + str(train_df.shape))

# %% [code]
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (15,5))

plt.subplot(1,2,1)
plt.title('Traning data language distribution')
sns.countplot(data = train_df, x = 'lang_abv', order = train_df['lang_abv'].value_counts().index)

plt.subplot(1,2,2)
plt.title('Test data laguage distribution')
sns.countplot(data = test_df, x = 'lang_abv', order = test_df['lang_abv'].value_counts().index)

# %% [markdown]
# The language column is clearly unbalanced being English the most frequent language

# %% [code]
# word count

def word_count(dataset, column):
    len_vector = []
    for text in dataset[column]:
        len_vector.append(len(text.split()))
    
    return len_vector

train_premise = word_count(train_df, 'premise')
train_hypothesis = word_count(train_df, 'hypothesis')

test_premise = word_count(test_df, 'premise')
test_hypothesis = word_count(test_df, 'hypothesis')

fig = plt.figure(figsize = (15,10))

plt.subplot(2,2,1)
plt.title('word count for train dataset premise')
sns.distplot(train_premise)

plt.subplot(2,2,2)
plt.title('word count for train dataset hypothesis')
sns.distplot(train_hypothesis)

plt.subplot(2,2,3)
plt.title('word count for test dataset premise')
sns.distplot(test_premise)

plt.subplot(2,2,4)
plt.title('word count for test dataset hypothesis')
sns.distplot(test_hypothesis)        

# %% [markdown]
# premises are observed to be longer than the hypothesis

# %% [code]
# looking at the countplot of the labels of the traning data set

plt.title('Label column countplot')
sns.countplot(data = train_df, x = 'label')

# %% [markdown]
# The following code is used to tokenize and preprocess the data for the Hugginface model, creating an array of ids, maks and type_id

# %% [code]
from transformers import BertTokenizer, TFAutoModel, AutoTokenizer
import tensorflow as tf
import keras
from tensorflow.math import softplus, tanh
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Embedding, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers


np.random.seed(123)
max_len = 50

# this is the model used BERT huggin face

Bert_model = "bert-large-uncased"

# tokenizer

Bert_tokenizer = BertTokenizer.from_pretrained(Bert_model)

def tokeniZer(dataset,tokenizer):
    encoded_list = [] # word id array
    type_id_list = np.zeros((dataset.shape[0], max_len)) #type id array
    mask_list = np.zeros((dataset.shape[0], max_len)) #masks array
    
    for i in range(dataset.shape[0]):
        datapoint = '[CLS] ' + dataset['premise'][i] + ' [SEP]' + dataset['hypothesis'][i] + ' [SEP]' # putting the two sentences together along with special characters
        datapoint = tokenizer.tokenize(datapoint)
        datapoint = tokenizer.convert_tokens_to_ids(datapoint)
        encoded_list.append(datapoint) 
    
    encoded_list = pad_sequences(encoded_list, maxlen = max_len, padding = 'post')
    
    for i in range(encoded_list.shape[0]):
        flag = 0
        a = encoded_list[i]
        for j in range(len(a)):
            
            #building the type_id matrix
            
            if flag == 0:
                type_id_list[i,j] = 0
            else:
                type_id_list[i,j] = 1
                
            #flag for the type_id matrix
            
            if encoded_list[i,j] == 102:
                flag = 1
            
    
            #building the mask matrix 
            
            if encoded_list[i,j] == 0:
                mask_list[i,j] = 0
            else:
                mask_list[i,j] = 1
                
    return encoded_list,mask_list,type_id_list
        
        
        

# %% [code]
# softplus - log(exp(x)+1), function that can be used for extra layers in the models
def mish(x):
    return x*tanh(softplus(x))
get_custom_objects()["mish"] = Activation(mish)

# %% [code]
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# %% [markdown]
# This is the function to create a single BERT model that takes as single input  random seed which is used in this case to create an esemble model using several BERT models with different seeds and shuffling the data

# %% [code]
# model creator

def create_BERT(random_seed):
    
    tf.random.set_seed(random_seed)
    
    with tpu_strategy.scope():
    
        transformer_encoder = TFAutoModel.from_pretrained(Bert_model)

        input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_layer")
        input_masks = Input(shape = (max_len,), dtype = tf.int32, name = 'input_mask')
        input_type_id = Input(shape = (max_len,), dtype = tf.int32, name = 'input_type_id')

        sequence_output = transformer_encoder([input_ids, input_masks, input_type_id])[0]

        cls_token = sequence_output[:, 0, :]

        output_layer = Dense(3, activation='softmax')(cls_token)


        model = Model(inputs=[input_ids, input_masks, input_type_id], outputs = output_layer)

        model.summary()

        model.compile(Adam(lr=1e-5), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy']
            )
    return model

# %% [code]
#ensemble creation and prediction

from sklearn.utils import shuffle # shuffle dataframes

callbacks = [tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss', \
                                           restore_best_weights = True, mode = 'min')]

shuffled_data = shuffle(train_df).reset_index(drop = True)#shuffle the data to add more variance

train_df = None #clearing more memory

batch_size = 128

# %% [markdown]
# The following code will preprocess and create a XLM-RoBERTa Model 

# %% [code]
XLM_model = "jplu/tf-xlm-roberta-large"
xlm_tokenizer = AutoTokenizer.from_pretrained(XLM_model) #Xlm tokenizer


X_train_ids, X_train_masks, _ = tokeniZer(shuffled_data,xlm_tokenizer) #encoding input

# %% [code]
# creating the XLM model 

def create_xlm(transformer_layer,  random_seed, learning_rate = 1e-5):
    
    tf.keras.backend.clear_session()

    tf.random.set_seed(random_seed)
    
    with tpu_strategy.scope():
    
        input_ids = Input(shape = (max_len,), dtype = tf.int32)
        input_masks = Input(shape = (max_len,), dtype = tf.int32)

            #insert roberta layer
        roberta = TFAutoModel.from_pretrained(transformer_layer)
        roberta = roberta([input_ids, input_masks])[0]
        
        out = GlobalAveragePooling1D()(roberta)
                

                #add our softmax layer
        out = Dense(3, activation = 'softmax')(out)

        #assemble model and compile


        model = Model(inputs = [input_ids, input_masks], outputs = out)
        model.compile(
                                optimizer = Adam(lr = learning_rate), 
                                loss = 'sparse_categorical_crossentropy', 
                                metrics = ['accuracy'])
    model.summary()
        
    return model  

Xlm = create_xlm(XLM_model ,123443334, 1e-5)

# %% [code]
#STEPS_PER_EPOCH = int(train_df.shape[0] // batch_size)

history_xlm = Xlm.fit([X_train_ids, X_train_masks], shuffled_data['label'],
          batch_size = batch_size,
        validation_split = 0.2,
         epochs = 39, callbacks = callbacks)

# %% [code]
# preprocessing test data

input_ids_test_xml, input_masks_test_xml, _ = tokeniZer(test_df, xlm_tokenizer)

#model predictions

predictions_xlm = Xlm.predict([input_ids_test_xml, input_masks_test_xml])

predictions = predictions_xlm

final = np.argmax(predictions, axis = 1)    

submission = pd.DataFrame()    

submission['id'] = test_df['id']
submission['prediction'] = final.astype(np.int32)

submission.to_csv('submission.csv', index = False)