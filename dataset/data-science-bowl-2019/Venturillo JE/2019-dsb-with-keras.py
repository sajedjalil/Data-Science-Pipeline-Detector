import os
import numpy as np
import pandas as pd
import random as rd

seed = 11

np.random.seed(seed)
rd.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1_l2 as ll
from keras.optimizers import RMSprop,Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau as RLRP
from keras.callbacks import EarlyStopping as ES

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split as TTS

## credits: 
# feature engineering, model: https://www.kaggle.com/ragnar123/simple-exploratory-data-analysis-and-model
# metric calculation: https://www.kaggle.com/cpmpml/ultra-fast-qwk-calc-method

def create_model(size):
    model = Sequential()
    model.add(Dense(256, input_dim=size, activation='hard_sigmoid'))
    model.add(Dropout(0.625))
    #model.add(Dense(32, activation='hard_sigmoid'))
    #model.add(Dropout(0.1))
    model.add(Dense(16, activation='hard_sigmoid'))
    model.add(Dropout(.25))
    #model.add(Dense(4, activation='sigmoid'))
    model.add(Dense(4, activation='softmax'))
    
    optimizer = RMSprop(lr=0.007, epsilon=3e-7, decay=3e-7)
    
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['acc'])
    return model

def graph(history):
    # summarize history for accuracy
    import matplotlib.pyplot as plt
    print(f'Accuracy:')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    #plt.ylim(top=.5, bottom=-.5)
    plt.savefig('history_acc.png')
    plt.show()
    
    #summarize history for loss
    print('Loss:')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    #plt.ylim(top=.5, bottom=-.5)
    plt.savefig('history_loss.png')
    plt.show()
    
def read_data():
    print(f'Read data')
    train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')
    test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')
    train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
    specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
    sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
    
    return train_df, test_df, train_labels_df, specs_df, sample_submission_df


def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekofyear'] = df['timestamp'].dt.weekofyear
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_month_start'] = df['timestamp'].dt.is_month_start    
    
    return df
    
def get_object_columns(df, columns):
    df = df.groupby(['installation_id', columns])['event_id'].count().reset_index()
    df = df.pivot_table(index = 'installation_id', columns = [columns], values = 'event_id')
    df.columns = list(df.columns)
    df.fillna(0, inplace = True)
    return df

def get_numeric_columns(df, column):
    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']})
    df[column].fillna(df[column].mean(), inplace = True)
    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_min', f'{column}_max', f'{column}_std', f'{column}_skew']
    return df

def get_numeric_columns_add(df, agg_column, column):
    df = df.groupby(['installation_id', agg_column]).agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']}).reset_index()
    df = df.pivot_table(index = 'installation_id', columns = [agg_column], values = [col for col in df.columns if col not in ['installation_id', 'type']])
    df[column].fillna(df[column].mean(), inplace = True)
    df.columns = list(df.columns)
    return df

def perform_features_engineering(train_df, test_df, train_labels_df):
    print(f'Perform features engineering')
    numerical_columns = ['game_time']
    categorical_columns = ['type', 'world']

    comp_train_df = pd.DataFrame({'installation_id': train_df['installation_id'].unique()})
    comp_train_df.set_index('installation_id', inplace = True)
    comp_test_df = pd.DataFrame({'installation_id': test_df['installation_id'].unique()})
    comp_test_df.set_index('installation_id', inplace = True)

    test_df = extract_time_features(test_df)
    train_df = extract_time_features(train_df)

    for i in numerical_columns:
        comp_train_df = comp_train_df.merge(get_numeric_columns(train_df, i), left_index = True, right_index = True)
        comp_test_df = comp_test_df.merge(get_numeric_columns(test_df, i), left_index = True, right_index = True)
    
    for i in categorical_columns:
        comp_train_df = comp_train_df.merge(get_object_columns(train_df, i), left_index = True, right_index = True)
        comp_test_df = comp_test_df.merge(get_object_columns(test_df, i), left_index = True, right_index = True)
    
    for i in categorical_columns:
        for j in numerical_columns:
            comp_train_df = comp_train_df.merge(get_numeric_columns_add(train_df, i, j), left_index = True, right_index = True)
            comp_test_df = comp_test_df.merge(get_numeric_columns_add(test_df, i, j), left_index = True, right_index = True)
    
    
    comp_train_df.reset_index(inplace = True)
    comp_test_df.reset_index(inplace = True)
    
    print('Our training set have {} rows and {} columns'.format(comp_train_df.shape[0], comp_train_df.shape[1]))

    # get the mode of the title
    labels_map = dict(train_labels_df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))
    # merge target
    labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]
    # replace title with the mode
    labels['title'] = labels['title'].map(labels_map)
    # get title from the test set
    comp_test_df['title'] = test_df.groupby('installation_id').last()['title'].map(labels_map).reset_index(drop = True)
    # join train with labels
    comp_train_df = labels.merge(comp_train_df, on = 'installation_id', how = 'left')
    print('We have {} training rows'.format(comp_train_df.shape[0]))
    
    return comp_train_df, comp_test_df


def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)
    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))
    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)
    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)
    e = e / a1.shape[0]
    return 1 - o / e

def run_model(comp_train_df, comp_test_df):
    print(f'Run model')
    scale = SS(with_mean=False)
    x_col = [col for col in comp_train_df.columns if col not in ['installation_id', 'accuracy_group']]
    y = to_categorical(comp_train_df['accuracy_group'], num_classes=4)
    
    x_train, x_val, y_train, y_val = TTS(scale.fit_transform(comp_train_df[x_col]), y, random_state=seed)
    
    model = create_model(x_train.shape[1])
    
    lrr = RLRP(monitor='val_acc',
               patience=5,
               verbose=1,
               factor=0.95,
               min_lr=1e-5)
    es = ES(monitor='val_loss',
           patience=20,
           mode='min',
           verbose=1)
               
    print(f'Model Fitting...')
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=1000, batch_size=64, verbose=2,
             callbacks=[lrr, es])
    graph(history)
    pred = model.predict(x_val)
    
    res = qwk3(y_val.argmax(axis=1), pred.argmax(axis=1))
    print(f'Quadratic weighted score: {np.round(res,4)}')
    #print(history.history['val_acc'][-1])
    y_pred = model.predict(scale.fit_transform(comp_test_df[x_col]))
    return y_pred

def prepare_submission(comp_test_df, sample_submission_df, y_pred):
    comp_test_df = comp_test_df.reset_index()
    comp_test_df = comp_test_df[['installation_id']]
    comp_test_df['accuracy_group'] = y_pred.argmax(axis = 1)
    sample_submission_df.drop('accuracy_group', inplace = True, axis = 1)
    sample_submission_df = sample_submission_df.merge(comp_test_df, on = 'installation_id')
    sample_submission_df.to_csv('submission.csv', index = False)

    
train_df, test_df, train_labels_df, specs_df, sample_submission_df = read_data()
comp_train_df, comp_test_df = perform_features_engineering(train_df, test_df, train_labels_df)
y_pred = run_model(comp_train_df, comp_test_df)
prepare_submission(comp_test_df, sample_submission_df, y_pred)
