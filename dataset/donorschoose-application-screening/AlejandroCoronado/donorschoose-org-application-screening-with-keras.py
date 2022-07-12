import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

import math

train_data  = pd.read_csv( "../input/train.csv")
test_data = pd.read_csv( "../input/test.csv")


def toLower(dataset):
    """
    Transform all string variables to lower case
    """
    for i in dataset.columns:
        try:
            dataset[i] = dataset[i].str.lower()
        except Exception as e:
            continue
    return dataset


def handle_missing(dataset):
    """
    Replace missing values
    """
    dataset.project_grade_category.fillna(value="missing", inplace=True)
    dataset.project_subject_categories.fillna(value="missing", inplace=True)
    dataset.project_subject_subcategories.fillna(value="missing", inplace=True)
    return (dataset)

train_data  = toLower(train_data)
test_data  = toLower(test_data)

train_data  = handle_missing(train_data)
test_data  = handle_missing(test_data)



#PROCESS CATEGORICAL DATA
"""
In this section we use an encoder in order to transform each of the categorical variables to
numeric id.

"""
print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train_data.project_subject_categories, test_data.project_subject_categories]))
train_data.project_subject_categories = le.transform(train_data.project_subject_categories)
test_data.project_subject_categories = le.transform(test_data.project_subject_categories)

le.fit(np.hstack([train_data.project_subject_subcategories, test_data.project_subject_subcategories]))
train_data.project_subject_subcategories = le.transform(train_data.project_subject_subcategories)
test_data.project_subject_subcategories = le.transform(test_data.project_subject_subcategories)

le.fit(np.hstack([train_data.project_grade_category, test_data.project_grade_category]))
train_data.project_grade_category = le.transform(train_data.project_grade_category)
test_data.project_grade_category = le.transform(test_data.project_grade_category)


train_data.head(3)



"""
For the test we can't just simply use a LabelEnconder because we have combinations
we need to take advantaje of the structure of the text in order to generate more 
significative variables.
Supouse for example that there are a lot of descriptions with word fun. 
we may calssify all the descrptions as afun label but this is not entirely true.
Imagine the descrption actually was This project is NOT fun, well it changes everything
right?

We use tokenizer in order to provide some sense to the words, just as a programming languaje 
parser. I'm just going to use the first project description but you can use it for the second
and third and I also sugest to use it for the title and project_subject_categories.
"""

#PROCESS TEXT: RAW
print("Text to seq process...")
from keras.preprocessing.text import Tokenizer
raw_text = np.hstack([train_data.project_essay_1.str.lower()])

#raw_text = np.hstack([train_data.project_essay_1.str.lower(), train_data.project_essay_2.str.lower(), train_data.project_essay_3.str.lower()])

print("   Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")

train_data["project_essay_1"] = tok_raw.texts_to_sequences(train_data.project_essay_1.str.lower())
test_data["project_essay_1"] = tok_raw.texts_to_sequences(test_data.project_essay_1.str.lower())
#train_data["project_essay_2"] = tok_raw.texts_to_sequences(train_data.project_essay_2.str.lower())
#test_data["project_essay_2"] = tok_raw.texts_to_sequences(test_data.project_essay_2.str.lower())


train_data.head(3)




#SEQUENCES VARIABLES ANALYSIS
max_name_seq = np.max([np.max(train_data.project_essay_1.apply(lambda x: len(x))), np.max(test_data.project_essay_1.apply(lambda x: len(x)))])
#max_name_seq = np.max([np.max(train_data.project_essay_2.apply(lambda x: len(x))), np.max(test_data.project_essay_2.apply(lambda x: len(x)))])
#max_name_seq = np.max([np.max(train_data.project_essay_3.apply(lambda x: len(x))), np.max(test_data.project_essay_3.apply(lambda x: len(x)))])

print("max essay seq "+str(max_name_seq))

train_data.project_essay_1.apply(lambda x: len(x)).max()


#EMBEDDINGS MAX VALUE
#Base on the histograms, we select the next lengths
MAX_TEXT = np.max([np.max(train_data.project_essay_1.max())
                   , np.max(test_data.project_essay_1.max())])+2
MAX_CATEGORY = np.max([train_data.project_subject_categories.max(), test_data.project_subject_categories.max()])+1
#MAX_ESSAY1 = np.max([train_data.project_essay_1.max(), test_data.project_essay_1.max()])+1

#MAX_BRAND = np.max([train_data.brand_name.max(), test_data.brand_name.max()])+1
#MAX_CONDITION = np.max([train_data.item_condition_id.max(), test_data.item_condition_id.max()])+1

#KERAS DATA DEFINITION
from keras.preprocessing.sequence import pad_sequences

def get_keras_data(dataset):
    X = {
        'essay1': pad_sequences(dataset.project_essay_1, maxlen=MAX_ESSAY1), 
        'category_name': np.array(dataset.project_subject_categories)
    }
    return X

#EXTRACT DEVELOPTMENT TEST
dtrain, dvalid = train_test_split(train_data, random_state=123, train_size=0.80)
print(dtrain.shape)
print(dvalid.shape)


X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test_data = get_keras_data(test_data)



#KERAS MODEL DEFINITION
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def get_model():
    #params
    dr_r = 0.1
    
    #Inputs
    essay1 = Input(shape=[X_train["essay1"].shape[1]], name="essay1")
    category_name = Input(shape=[1], name="category_name")
    
    #Embeddings layers
    emb_essay1 = Embedding(MAX_TEXT, 50)(essay1)
    emb_category_name = Embedding(MAX_TEXT, 50)(category_name)
    
    #rnn layer
    rnn_layer2 = GRU(8) (emb_essay1)
    
    #main layer
    main_l = concatenate([Flatten() (emb_category_name), rnn_layer2])
    main_l = Dropout(dr_r) (Dense(128) (main_l))
    main_l = Dropout(dr_r) (Dense(64) (main_l))
    
    #output
    output = Dense(1, activation="linear") (main_l)
    
    #model
    model = Model([essay1, category_name], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])
    
    return model


#FITTING THE MODEL  Change the values so ican run in the kernel but you could try with different values
BATCH_SIZE = 200 #20000
epochs = 1 #5

model = get_model()
model.fit(X_train, dtrain.project_is_approved, epochs=epochs, batch_size=BATCH_SIZE
          , validation_data=(X_valid, dvalid.project_is_approved)
          , verbose=1)

#EVLUEATE THE MODEL ON DEV TEST: What is it doing?
val_preds = model.predict(X_valid)
val_preds = np.exp(val_preds)+1

