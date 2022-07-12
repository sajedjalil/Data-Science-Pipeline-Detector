# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


def updateDataset(data):
   data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
   data['inc_angle'] = data['inc_angle'].fillna(method='pad')


def normalize(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x

# todo: implement logistic regression for gradient descent
#  y^ = sigmoid ( W^T * x + b)

def sigmoid(z):
    """
       calculates sigmoid (z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))


def initialize_parameters_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b


def loss_function(Y, A):
    """
        calculates loss function
        L( y^, y) = - ( y * log(y^) + (1-y) * log(1- y^))
    """
    return - (Y * np.log(A) + (1 - Y) * np.log(1 - A))


def forward_propagate(w, b, X, Y):
    m = X.shape[1]
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    cost = (1 / m) * np.sum(loss_function(Y, A))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return A, cost


def backward_propagate(w, A, X, Y):
    m = X.shape[1]
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    return dw, db


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        A, cost = forward_propagate(w, b, X, Y)
        dw, db = backward_propagate(w, A, X, Y)
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return w, b, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m), dtype=np.int)
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    assert(Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_parameters_with_zeros(X_train.shape[0])
    w, b, costs = optimize(
       w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    values = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "train_accuracy": train_accuracy,
         "test_accuracy": test_accuracy,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return values


train_filename = '../input/train.json'
test_filename = '../input/test.json'

# Data fields
# train.json, test.json
#
#     id - the id of the image
#     band_1, band_2 - the flattened image data. Each band has 75x75 pixel values in the list
#     inc_angle - the incidence angle of which the image was taken.
#     is_iceberg - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship.


train = pd.read_json(train_filename)
updateDataset(train)
m_train = train.shape[0]

test = pd.read_json(test_filename)
updateDataset(test)

to_arr = lambda x: np.asarray([np.asarray(item) for item in x])

split_value = 0.92
train_number = int(m_train * split_value)

train_set_x, dev_set_x = np.split(
    to_arr(train['band_1'].values), [train_number])
train_set_y, dev_set_y = np.split(
    train['is_iceberg'].values, [train_number])

print("train_set_y iceberg count is {}".format(np.count_nonzero(train_set_y)))
print("dev_set_y iceberg count is {}".format(np.count_nonzero(dev_set_y)))

train_set_x = normalize(train_set_x.T)
train_set_y = normalize(train_set_y.reshape(1, train_number))

dev_set_x = normalize(dev_set_x.T)
dev_set_y = normalize(dev_set_y.reshape(1, m_train - train_number))

# run band_1 model 
model_band_1 = model(train_set_x, train_set_y, dev_set_x, dev_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=True)

print("band_1 train accuracy: {} %".format(model_band_1["train_accuracy"]))
print("band_1 test accuracy: {} %".format(model_band_1["test_accuracy"]))

# band_2
train_set_x, dev_set_x = np.split(
    to_arr(train['band_2'].values), [train_number])
train_set_x = normalize(train_set_x.T)
dev_set_x = normalize(dev_set_x.T)

# run band_2 model 
model_band_2 = model(train_set_x, train_set_y, dev_set_x, dev_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=True)

print("band_2 train accuracy: {} %".format(model_band_2["train_accuracy"]))
print("band_2 test accuracy: {} %".format(model_band_2["test_accuracy"]))

# estimate

X_test_band_1 = to_arr(test['band_1'].values)
X_test_band_1 = normalize(X_test_band_1.T)

X_test_band_2 = to_arr(test['band_2'].values)
X_test_band_2 = normalize(X_test_band_2.T)

Y_test_band_1 = predict(model_band_1["w"], model_band_1["b"], X_test_band_1)
Y_test_band_2 = predict(model_band_2["w"], model_band_2["b"], X_test_band_2)

print(np.count_nonzero(Y_test_band_1))
print(np.count_nonzero(Y_test_band_2))
print(np.count_nonzero((Y_test_band_1==Y_test_band_2).all()))

# save results
subm = pd.DataFrame()
subm['id'] = test['id']
subm['is_iceberg'] = np.squeeze(Y_test_band_1 * Y_test_band_2)
subm.to_csv("submission.csv", index=False)