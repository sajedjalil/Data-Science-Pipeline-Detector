__author__ = "Vang https://www.kaggle.com/vangaa"


import time

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout
from keras.regularizers import l2

# Preprocess functions get from
# https://www.kaggle.com/jeffd23/predicting-red-hat-business-value/single-unified-table-0-94-sklearn

def preprocess_acts(data, train_set=True):

    # Getting rid of data feature for now
    data = data.drop(['activity_id'], axis=1)
    data.date = pd.to_datetime(data.date)

    if(train_set):
        data = data.drop(['outcome'], axis=1)

    ## Split off _ from people_id
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)

    columns = list(data.columns)
    columns.remove("date")

    # Convert strings to ints
    for col in columns[1:]:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col] = pd.to_numeric(data[col]).astype(int)

    # Add features from date
    data["year"] = data.date.apply(lambda x: x.year)
    data["month"] = data.date.apply(lambda x: x.month)
    data["day"] = data.date.apply(lambda x: x.day)
    data = data.drop(["date"], axis = 1)
    return data

def preprocess_people(data):

    # TODO refactor this duplication
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)

    data.date = pd.to_datetime(data.date)
    #  Values in the people df is Booleans and Strings
    columns = list(data.columns)
    columns.remove("date")
    bools = columns[11:]
    strings = columns[1:11]

    for col in bools:
        data[col] = pd.to_numeric(data[col]).astype(int)
    for col in strings:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col] = pd.to_numeric(data[col]).astype(int)

    # Add features from date
    data["year_p"] = data.date.apply(lambda x: x.year)
    data["month_p"] = data.date.apply(lambda x: x.month)
    data["day_p"] = data.date.apply(lambda x: x.day)

    data = data.drop(['date'], axis=1)
    return data

def get_model(input_shape, layers, dropout = 0.4, regularization = 1e-3):
    input = Input(shape=(input_shape,))

    layer = input
    for layer_dim in layers:
        layer = Dense(layer_dim, activation = 'relu',
                      W_regularizer=l2(regularization),
                      b_regularizer=l2(regularization))(layer)
    
    layer = Dropout(dropout)(layer)
    result = Dense(1, activation = 'sigmoid')(layer)


    model = Model(input=input, output=result)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model

def make_submission(model, test_labels, X_test,
                    submission_file_template = "submission_{}.csv"):
    submission = pd.DataFrame()
    submission["activity_id"] = test_labels
    submission["outcome"] = model.predict(X_test)
    filename = submission_file_template.format(time.strftime("%d-%m-%Y_%H-%M"))
    submission.to_csv(filename, index=None)

def main():
    test_size = 0.20
    
    ## Neural network layers configuration
    nn_layers = [100, 50, 25]
    train_epoches = 10
    batch_size = 100

    train_df = pd.read_csv("../input/act_train.csv")
    test_df = pd.read_csv("../input/act_test.csv")
    people = pd.read_csv("../input/people.csv")

    print("Train size = {0}, test size = {1}".format(train_df.shape[0], test_df.shape[0]))
    print("Start preprocessing data...")
    peeps = preprocess_people(people)
    actions_train = preprocess_acts(train_df)
    actions_test = preprocess_acts(test_df, train_set=False)

    features_train = actions_train.merge(peeps, how='left', on='people_id')
    labels = train_df['outcome']
    features_test = actions_test.merge(peeps, how='left', on='people_id')

    # Scale all features for Neural Network. You can also try min-max scaler
    scaler = StandardScaler().fit(features_train)

    X_train, X_test, y_train, y_test = train_test_split(
        scaler.transform(features_train), labels, test_size = test_size,
        random_state=2345, stratify=labels)

    # Try use another layers configuration
    print("Builing keras model with layers configuration {0}...".format(str(nn_layers)))
    model = get_model(X_train.shape[1], nn_layers)

    print("Train model with batch_size = {0} on {1} epoches..."\
        .format(batch_size, train_epoches))
    
    callback = EarlyStopping("val_loss", patience=1, verbose=0, mode='auto')
    model.fit(
        X_train, y_train,
        nb_epoch=train_epoches, batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[callback], verbose = 0)

    # Calculate total roc auc score
    score = roc_auc_score(y_test, model.predict(X_test))
    print("Total roc auc score = {0:0.4f}".format(score))

    print("Making submission file")
    make_submission(model, test_df.activity_id, scaler.transform(features_test))
    print("Submission ready to upload. Good luck :)")


main()