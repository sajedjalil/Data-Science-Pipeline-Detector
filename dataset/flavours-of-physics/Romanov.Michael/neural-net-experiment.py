# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas
import numpy
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.regularizers import l2

valid_size = 0.25

train_df = pandas.read_csv('../input/training.csv')
test_df = pandas.read_csv('../input/test.csv')

X_train = train_df.values[:, 1:-4]
y_train = train_df.values[:, -3]

valid_num = int(valid_size * float(X_train.shape[0]))

numpy.random.seed(100)
numpy.random.shuffle(X_train)
numpy.random.seed(100)
numpy.random.shuffle(y_train)

X_valid = X_train[:valid_num, :]
y_valid = y_train[:valid_num]

X_train = X_train[valid_num:, :]
y_train = y_train[valid_num:]

print(X_train.shape, '<- X_train shape')
print(X_valid.shape, '<- X_valid shape')
print(y_train.shape, '<- y_train shape')
print(y_valid.shape, '<- y_valid shape')

network = Sequential()

num_features = len(train_df.columns) - 1 - 4

first_layer = 100
second_layer = 50
third_layer = 25
fourth_layer = 10
fifth_layer = 5

reg_parameter = 0.00

network.add(Dense(num_features, first_layer, W_regularizer=l2(reg_parameter)))
network.add(Activation('tanh'))
network.add(Dropout(0.1))
network.add(Dense(first_layer, second_layer))
network.add(Activation('tanh'))
network.add(Dense(second_layer, third_layer))
network.add(Activation('tanh'))
network.add(Dense(third_layer, 1))
network.add(Activation('softmax'))

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
network.compile(loss='binary_crossentropy', optimizer=sgd)
network.fit(X_train, y_train, batch_size=128, verbose=2, nb_epoch=100, validation_data=[X_valid, y_valid], show_accuracy=True)
#selected_columns = []

#print(train_df.describe())
#print(test_df.describe())

print(train_df.columns)
#print(test_df.columns)

#plot()