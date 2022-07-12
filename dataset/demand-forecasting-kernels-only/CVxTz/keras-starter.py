import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# https://www.kaggle.com/hammadkhan/xgboost-with-timeseriesfeatures
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv('../input/sample_submission.csv')
print("Train rows and columns : ", train.shape)
print("Test rows and columns : ", test.shape)

df = pd.concat([train, test])
print(df.shape)

df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['week_of_year'] = df.date.dt.weekofyear

df["median-store_item"] = df.groupby(["item", "store"])["sales"].transform("median")
df["mean-store_item"] = df.groupby(["item", "store"])["sales"].transform("mean")
df["mean-month_item"] = df.groupby(["month", "item"])["sales"].transform("mean")
df["median-month_item"] = df.groupby(["month", "item"])["sales"].transform("median")
df["median-month_store"] = df.groupby(["month", "store"])["sales"].transform("median")
df["median-item"] = df.groupby(["item"])["sales"].transform("median")
df["median-store"] = df.groupby(["store"])["sales"].transform("median")
df["mean-item"] = df.groupby(["item"])["sales"].transform("mean")
df["mean-store"] = df.groupby(["store"])["sales"].transform("mean")

df["median-store_item-month"] = df.groupby(['month', "item", "store"])["sales"].transform("median")
df["mean-store_item-week"] = df.groupby(['week_of_year', "item", "store"])["sales"].transform("mean")
df["item-month-mean"] = df.groupby(['month', "item"])["sales"].transform(
    "mean")  # mean sales of that item  for all stores scaled

df["store-month-mean"] = df.groupby(['month', "store"])["sales"].transform(
    "mean")  # mean sales of that store  for all items scaled

df['store_item_shifted-365'] = df.groupby(["item", "store"])['sales'].transform(
    lambda x: x.shift(365))  # sales for that 1 year  ago
df["item-week_shifted-90"] = df.groupby(['week_of_year', "item"])["sales"].transform(
    lambda x: x.shift(12).mean())  # shifted total sales for that item 12 weeks (3 months) ago

df['store_item_shifted-365'].fillna(df['store_item_shifted-365'].mode()[0], inplace=True)
df["item-week_shifted-90"].fillna(df["item-week_shifted-90"].mode()[0], inplace=True)

numeric_variables = ["median-store_item-month", "mean-store_item-week", "item-month-mean", "store-month-mean",
                     'store_item_shifted-365', "item-week_shifted-90", "median-store_item", "mean-store_item",
                     "mean-month_item", "median-month_item", "median-month_store", "median-item", "median-store",
                     "mean-item", "mean-store"]

cat_variables = ["month", "week_of_year"]#, "item", "store"
cat_variables_ = [c + "_" for c in cat_variables]

for cat in cat_variables:
    df[cat] = df[cat].apply(lambda x: "%s_%s" % (str(x), cat))

set_cat_variables = set()

for cat in tqdm(cat_variables):
    set_cat_variables.update(set(df[cat]))

print("mapping : ")
cat_map = {value: i for i, value in enumerate(set_cat_variables)}

train = df[df.sales.notnull()]
print("new train", train.shape)
test = df[df.id.notnull()]
print("new test", test.shape)

for df in tqdm([train, test]):
    for cat in tqdm(cat_variables):
        df[cat + "_"] = df[cat].apply(lambda x: cat_map.get(x, 0))

target = "sales"

X_cat_train = np.array(train[cat_variables_])
X_cat_test = np.array(test[cat_variables_])
X_num_train = np.array(train[numeric_variables])
X_num_test = np.array(test[numeric_variables])

scaler = StandardScaler()
X_num_train_ = scaler.fit_transform(X_num_train)
X_num_test_ = scaler.transform(X_num_test)

Y_train = np.array(train[target].values)
# mean, std = float(np.mean(Y_train)), float(np.mean(Y_train))
# Y_train = (Y_train-mean)/std
#

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import concatenate, Flatten, Dropout, \
    GlobalAvgPool1D, SpatialDropout1D, dot
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.losses import mae
import tensorflow as tf


def smape(y_true, y_pred):
    # return tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y_pred,y_true)*std),
    #                                 (tf.abs(y_true*std+mean)+tf.abs(y_pred*std+mean)+ 1e-5)))
    return tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y_pred,y_true)),
                                    (tf.abs(y_true)+tf.abs(y_pred)+ 1e-5)))

def get_model():
    input_cat = Input((len(cat_variables),))
    input_num_ = Input((len(numeric_variables),))
    input_num = Input((len(numeric_variables),))

    x_cat = SpatialDropout1D(0.1)(Embedding(len(cat_map), 10)(input_cat))
    x_cat_1 = Flatten()(x_cat)
    x_cat_1 = Dense(200, activation="relu")(x_cat_1)
    x_cat_2 = GlobalAvgPool1D()(x_cat)

    x_num = Dense(200, activation="relu")(input_num_)

    x = concatenate([x_cat_1, x_cat_2, x_num])
    x = Dropout(0.1)(x)

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(len(numeric_variables), activation="softmax")(x) #, activation="softmax"
    #x = Dense(1)(x)
    x = dot([x, input_num], axes=-1)

    model = Model(inputs=[input_cat, input_num_, input_num], outputs=x)
    model.compile(loss=smape, optimizer=Adam(0.001))

    model.summary()

    return model


n_bag = 5
predictions = 0
for i in range(n_bag):
    model = get_model()

    early = EarlyStopping(patience=5)
    reduce_on = ReduceLROnPlateau(patience=2)
    filepath = "baseline.h5"
    check = ModelCheckpoint(filepath)
    model.fit([X_cat_train, X_num_train_, X_num_train], Y_train, validation_split=0.1, epochs=5,
              callbacks=[early, check, reduce_on], batch_size=32)
    model.load_weights(filepath)
    predictions += model.predict([X_cat_test, X_num_test_, X_num_test]) / n_bag

# In[ ]:


test[target] = np.array(predictions).ravel()
test["id"] = np.array(test["id"]).astype(np.int32)
test[["id", target]].to_csv("baseline.csv", index=False)
