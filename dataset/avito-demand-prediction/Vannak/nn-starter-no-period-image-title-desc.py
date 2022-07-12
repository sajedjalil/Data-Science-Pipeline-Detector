"""
This model is designed to use embeddings to deal with the categorical variables, while ignoring 
some of the data like the periods, images, titles and descriptions. 
Most of the remaining data is categorical, except for the price. 
A simple FFNN model seems to do pretty good work. 

"""


import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Flatten
from keras.layers.merge import concatenate, dot, multiply, add
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Nadam, RMSprop, adam
from keras.layers.noise import AlphaDropout, GaussianNoise
from keras import backend as K

##================Import Data, Replace NaNs with -1
data_dir = "../input/"
train_data = pd.read_csv(data_dir+"/train.csv", parse_dates=["activation_date"]) #we will eventually turn the date column into day of week [0,6]
test_data  = pd.read_csv(data_dir+"/test.csv", parse_dates=["activation_date"])
train_data = train_data.replace(np.nan,-1,regex=True) #nan and other missing values are mapped to -1
test_data  = test_data.replace(np.nan,-1,regex=True)

##================Remove unwanted columns
del train_data['image'], test_data['image'],train_data['user_id'],
test_data['user_id'],train_data['item_id'],test_data['item_id']

##================Replace Full Dates with Day-of-Week
train_data['activation_date'] = train_data["activation_date"].dt.weekday
test_data['activation_date'] = test_data["activation_date"].dt.weekday

##================split into x_train/x_val. No stratification requried probably
val_split = 0.15
train_data = train_data.sample(frac=1).reset_index(drop=True)
val_ix = int(np.rint(len(train_data)*(1.-val_split)))
#data frame formats with y-values packed in
train_df = train_data[:val_ix]
val_df = train_data[val_ix:]
test_df = test_data

##================Create the Tokenizers
region_tk = {x:i+1 for i, x in enumerate(train_df.region.unique())}#+1 because we want to reserve 0 for new but not missing values
city_tk =  {x:i+1 for i, x in enumerate(train_df.city.unique())}
cat1_tk =  {x:i+1 for i, x in enumerate(train_df.parent_category_name.unique())}
cat2_tk =  {x:i+1 for i, x in enumerate(train_df.category_name.unique())}
param1_tk =  {x:i+1 for i, x in enumerate(train_df.param_1.unique())}
param2_tk =  {x:i+1 for i, x in enumerate(train_df.param_2.unique())}
param3_tk =  {x:i+1 for i, x in enumerate(train_df.param_3.unique())}
seqnum_tk =  {x:i+1 for i, x in enumerate(train_df.item_seq_number.unique())}
usertype_tk = {x:i+1 for i, x in enumerate(train_df.user_type.unique())}
imgtype_tk = {x:i+1 for i, x in enumerate(train_df.image_top_1.unique())}
tokenizers = [region_tk, city_tk, cat1_tk, cat2_tk, param1_tk, param2_tk, param3_tk, seqnum_tk, usertype_tk, imgtype_tk]

##================These functions are going to get repeated on train, val, and test data
def tokenize_data(data, tokenizers):
    region_tk, city_tk, cat1_tk, cat2_tk, param1_tk, param2_tk, param3_tk, seqnum_tk, usertype_tk, imgtype_tk = tokenizers
    x_reg = np.asarray([region_tk.get(key, 0) for key in data.region], dtype=int)
    x_city   = np.asarray([city_tk.get(key, 0) for key in data.city], dtype=int)
    x_cat1   = np.asarray([cat1_tk.get(key, 0) for key in data.parent_category_name], dtype=int)
    x_cat2   = np.asarray([cat2_tk.get(key, 0) for key in data.category_name], dtype=int)
    x_prm1 = np.asarray([param1_tk.get(key, 0) for key in data.param_1], dtype=int)
    x_prm2 = np.asarray([param2_tk.get(key, 0) for key in data.param_2], dtype=int)
    x_prm3 = np.asarray([param3_tk.get(key, 0) for key in data.param_3], dtype=int)
    x_sqnm = np.asarray([seqnum_tk.get(key, 0) for key in data.item_seq_number], dtype=int)
    x_usr = np.asarray([usertype_tk.get(key, 0) for key in data.user_type], dtype=int)
    x_itype = np.asarray([imgtype_tk.get(key, 0) for key in data.image_top_1], dtype=int)
    return [x_reg, x_city, x_cat1, x_cat2, x_prm1, x_prm2, x_prm3, x_sqnm, x_usr, x_itype]

def log_prices(data):
    prices = data.price.as_matrix()
    prices = np.log1p(prices)
    prices[prices==-np.inf] = -1
    return prices

##================Final Processing on x, y train, val, test data
x_train = tokenize_data(train_df, tokenizers)
x_train.append(train_df.activation_date.as_matrix())
x_train.append(log_prices(train_df))
y_train = train_df.deal_probability.as_matrix()

x_val = tokenize_data(val_df, tokenizers)
x_val.append(val_df.activation_date.as_matrix())
x_val.append(log_prices(val_df))
y_val = val_df.deal_probability.as_matrix()

x_test = tokenize_data(test_df, tokenizers)
x_test.append(test_df.activation_date.as_matrix())
x_test.append(log_prices(test_df))

##================Beginning of the NN Model Outline. 
def build_model():
    inp_reg = Input(shape=(1,))
    inp_city = Input(shape=(1,))
    inp_cat1 = Input(shape=(1,))
    inp_cat2 = Input(shape=(1,))
    inp_prm1 = Input(shape=(1,))
    inp_prm2 = Input(shape=(1,))
    inp_prm3 = Input(shape=(1,))
    inp_sqnm = Input(shape=(1,))
    inp_usr = Input(shape=(1,))
    inp_itype = Input(shape=(1,))
    inp_weekday = Input(shape=(1,))
    inp_price = Input(shape=(1,))
    nsy_price = GaussianNoise(0.1)(inp_price)
    
    emb_size = 8
    emb_reg  = Embedding(len(region_tk)+1, emb_size)(inp_reg)
    emb_city = Embedding(len(city_tk)+1, emb_size)(inp_city)
    emb_cat1 = Embedding(len(cat1_tk)+1, emb_size)(inp_cat1)
    emb_cat2 = Embedding(len(cat2_tk)+1, emb_size)(inp_cat2)
    emb_prm1 = Embedding(len(param1_tk)+1, emb_size)(inp_prm1)
    emb_prm2 = Embedding(len(param2_tk)+1, emb_size)(inp_prm2)
    emb_prm3 = Embedding(len(param3_tk)+1, emb_size)(inp_prm3)
    emb_sqnm = Embedding(len(seqnum_tk)+1, emb_size)(inp_sqnm)
    emb_usr  = Embedding(len(usertype_tk)+1, emb_size)(inp_usr)
    emb_itype= Embedding(len(imgtype_tk)+1, emb_size)(inp_itype)
    x = concatenate([emb_reg,emb_city,emb_cat1,emb_cat2,emb_prm1,emb_prm2,emb_prm3,
                     emb_sqnm,emb_usr,emb_itype])
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = concatenate([x, nsy_price])#Do not want to dropout price, its noised up instead. 
    
    x = Dense(64, activation="selu", kernel_initializer="lecun_normal")(x)
    x = AlphaDropout(0.05)(x)
    x = Dense(32, activation="selu", kernel_initializer="lecun_normal")(x)
    x = AlphaDropout(0.05)(x)
    x = Dense(8, activation="selu", kernel_initializer="lecun_normal")(x)
    y = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=[inp_reg, inp_city, inp_cat1, inp_cat2, inp_prm1, inp_prm2, 
                          inp_prm3, inp_sqnm, inp_usr, inp_itype, inp_weekday, inp_price],
                  outputs=y)
    model.compile(optimizer="Nadam", loss=["MSE"], metrics=[root_mean_squared_error])
    model.summary()
    
    return model

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model = build_model()

earlystop = EarlyStopping(monitor="val_loss",mode="auto",patience=5,verbose=0)
checkpt = ModelCheckpoint(monitor="val_loss",mode="auto",filepath='model_baseline_weights.hdf5',verbose=0,save_best_only=True)
rlrop = ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=2,verbose=1,factor=0.1,cooldown=0,min_lr=1e-6)
batch_size = 2048
model.fit(x_train, y_train,batch_size=batch_size,validation_data=(x_val, y_val),
          epochs=100,verbose=2,callbacks =[checkpt, earlystop, rlrop])

model.load_weights('model_baseline_weights.hdf5')
preds = model.predict(x_test, batch_size=batch_size)
submission = pd.read_csv(data_dir+"/sample_submission.csv")
submission['deal_probability'] = preds
submission.to_csv("submission.csv", index=False)