# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# coding=utf-8
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical


def sklear_mlp(x_train, y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(1024, 512), tol=1e-6,
                        alpha=1e-2, max_iter=5000, verbose=True)
    mlp.fit(x_train, y_train)
    return mlp


def keras_mlp(x_train, y_train):
    mlp = Sequential()
    mlp.add(Dense(1000, input_dim=192, init="normal", activation='relu'))
    mlp.add(Dropout(0.3))
    mlp.add(Dense(500, activation='sigmoid'))
    mlp.add(Dropout(0.3))
    mlp.add(Dense(99, activation='softmax'))
    mlp.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    estop = EarlyStopping(monitor='val_loss', patience=300)
    mlp.fit(x_train, y_train, batch_size=150,
            nb_epoch=1000, validation_split=0.1,
            callbacks=[estop])
    return mlp


if __name__ == "__main__":
    train_file = "../input/train.csv"
    test_file = "../input/test.csv"
    # train dataset
    train_df = pd.read_csv(train_file, index_col=0)
    train_data = train_df.ix[:, 'margin1':]
    train_target_label = train_df['species'].values
    # species
    species = list(set(train_target_label))
    species.sort()

    # test dataset
    test_df = pd.read_csv(test_file)
    test_data = test_df.ix[:, 'margin1':]

    # label encoder
    le = LabelEncoder()
    le.fit(train_target_label)
    train_target = le.transform(train_target_label)
    # train test split
    # spl = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
    X = train_data.values
    x_train = X
    y_train = train_target
    # for train_index, test_index in spl.split(X, train_target):
    #    x_train, x_test = X[train_index], X[test_index]
    #    y_train, y_test = train_target[train_index], train_target[test_index]
    scale = StandardScaler()
    # scale = Normalizer()
    scale.fit(x_train)
    x_train = scale.transform(x_train)
    # x_test = scale.transform(x_test)
    # x_test = StandardScaler().fit(x_test).transform(x_test)
    # pca
    # pca = PCA(svd_solver='full')
    # pca.fit(x_train)
    # x_train = pca.transform(x_train)
    # x_test = pca.transform(x_test)
    # only needed for keras
    y_train_keras = to_categorical(y_train)
    kmlp = keras_mlp(x_train, y_train_keras)
    kmlp.save("keras_model.h5")
    # kmlp = sklear_mlp(x_train, y_train)
    try:
        print("train_score:", accuracy_score(y_train, kmlp.predict_classes(x_train)))
        # print("test_score:", accuracy_score(y_test, kmlp.predict_classes(x_test)))
    except ValueError as err:
        print(err)
    print("\ntrain_loss:", log_loss(y_train, kmlp.predict_proba(x_train)))
    # print("\ntest_loss:", log_loss(y_test, kmlp.predict_proba(x_test)))

    test_data = test_data.values
    # test_data = scale.transform(test_data.values)
    test_data = StandardScaler().fit(test_data).transform(test_data)
    # test_data = pca.transform(test_data)
    test_target = kmlp.predict_proba(test_data)
    df = pd.DataFrame(data=test_target, index=test_df['id'].values,
                      columns=species)
    df.index.name = 'id'
    df.to_csv("submission_keras_nn2.csv")
    gc.collect()