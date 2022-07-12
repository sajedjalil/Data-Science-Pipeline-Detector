# coding=utf-8
"""

@author: André V Lopes
@I changed the keras-script-code from  'ZFTurbo: https://kaggle.com/zfturbo' , and implemented it in lasagne.
The load methods are from ZFTurbo and the KFold idea too.
I separated LasagneClassifier into another file, but since kaggle just allows to add 1 file script.. there it is.

Hopefully i can get some feedback from ZFTurbo and others :)

#Thanks ZFTurbo.

#
Change the model and batch_size and input_dim.
Changing input_dim, automatically resize the image based on it. 

"""

import datetime
import glob
import os
import os.path
import time
import warnings

import cv2
import lasagne
import nolearn
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from keras.utils import np_utils
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def get_im_cv2(path, rx=32, ry=32):
    resized = cv2.resize(cv2.imread(path), dsize=(rx, ry), interpolation=cv2.INTER_LINEAR)

    return resized


def load_train(rx=32, ry=32):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..','input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl, rx, ry)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def read_and_normalize_train_data(rx=32, ry=32):
    train_data, train_target, train_id = load_train(rx, ry)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    train_data = train_data.transpose((0, 3, 1, 2))

    train_data = train_data.astype('float32')
    train_data = train_data / 255

    train_target = np_utils.to_categorical(train_target, 8)
    train_target = np.array(train_target, dtype=np.uint8)

    # return train_data, train_target, train_id
    return train_data, train_target, train_id


def load_test(rx=32, ry=32):
    path = os.path.join('..','input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, rx, ry)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def read_and_normalize_test_data(rx=32, ry=32):
    start_time = time.time()
    test_data, test_id = load_test(rx, ry)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print ('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return test_data, test_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


class LasagneClassifier():
    network = None
    batch_size = None
    verbose = None
    colour_channels = None

    def __init__(self, *a, **kw):
        self.batch_size = kw['batch_size']
        self.verbose = kw['verbose']
        self.network = kw['network']
        self.input_var = kw['input_var']
        self.target_var = kw['target_var']
        self.colour_channels = kw.get('channels', 1)

    def setWeights(self, network, param_values):
        try:
            lasagne.layers.set_all_param_values(network, param_values)
        except Exception as ex:
            print ('Failure to load Weights , Reason :' + str(ex.message))

    def getNetworkWeights(self, network):
        try:
            return lasagne.layers.get_all_param_values(network)
        except Exception as ex:
            print ('Failure to get Weights , Reason :' + str(ex.message))

    def minibatch_iterator(self, inputs, targets, batch_size):

        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]
            yield inputs[excerpt], targets[excerpt]

    def minibatch_iterator_predictor(self, inputs, batch_size):

        assert len(inputs) > 0

        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt]

    def split_validation_data(self, x, y, validation_percentage=0, random_state=1):
        x, x_val, y, y_val = train_test_split(x, y, test_size=validation_percentage / 100.00, random_state=random_state)

        return x, y, x_val, y_val

    def decayLearningRate(self, current_learning_rate, decay):
        current_learning_rate = current_learning_rate - decay
        # print "LR ", np.float32(current_learning_rate)

        return np.float32(current_learning_rate)

    def train(self, dataset, kw):
        max_epochs = kw.get('epochs', 10)
        max_patience = kw.get('max_patience', int((10 * int(max_epochs)) / 100))
        test_loss = kw.get('test_loss')
        test_acc = kw.get('test_acc')
        updates = kw.get('update')
        loss_or_grads = kw.get('loss_or_grads')
        X, Y = dataset
        X_val, Y_val = kw.get('validation_set', (None, None))
        X_test, Y_test = kw.get('testset', (None, None))
        decay = kw.get('decay', 1e-6)
        update_learning_rate = kw.get('update_learning_rate')

        # Reshape them to fit the Convnet
        X = X.reshape((-1, self.colour_channels, lasagne.layers.get_all_layers(self.network)[0].shape[2], lasagne.layers.get_all_layers(self.network)[0].shape[3]))
        Y = Y.reshape(-1, 8)

        if X_test != None and Y_test != None:
            # Reshape the test set
            X_test = X_test.reshape((-1, self.colour_channels, lasagne.layers.get_all_layers(self.network)[0].shape[2], lasagne.layers.get_all_layers(self.network)[0].shape[3]))
            Y_test = Y_test.reshape(-1, 8)

        # Reshape the validation set
        X_val = X_val.reshape((-1, self.colour_channels, lasagne.layers.get_all_layers(self.network)[0].shape[2], lasagne.layers.get_all_layers(self.network)[0].shape[3]))
        Y_val = Y_val.reshape(-1, 8)

        # Create the Theano functions that will run on the GPU/CPU
        train_fn = theano.function([self.input_var, self.target_var], loss_or_grads, updates=updates)
        val_fn = theano.function([self.input_var, self.target_var], [test_loss, test_acc])

        # Save epochs in which nan's occurred.
        nan_occurrences = []

        # Prepare the patience variables
        if max_patience != -1:

            # Do not allow max_patience to be 0. It can be -1 to disable patience, but not 0.
            if max_patience == 0:
                max_patience = 1

            current_patience = int(max_patience)
            best_validation_loss = float('inf')

            best_acc = -1
            best_weights = None

        # Finally, launch the training loop.
        print("Starting training...")

        for epoch in range(max_epochs):

            # In each epoch, do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.minibatch_iterator(X, Y, self.batch_size):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.minibatch_iterator(X_val, Y_val, self.batch_size):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Calculate Results
            training_loss = (train_err / train_batches)
            validation_loss = (val_err / val_batches)
            validation_accuracy = (val_acc / val_batches * 100)

            # Then print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, max_epochs, time.time() - start_time))
            print("  Training Loss:\t\t{:.6f}".format(training_loss))
            print("  Validation Loss:\t\t{:.6f}".format(validation_loss))
            print("  Validation Accuracy:\t\t{:.4f} %".format(validation_accuracy))

            ##Check for NAN and append NAN warnings to HTML LOG.
            if np.isnan(training_loss):
                nan_occurrences.append("Training Loss is NaN at epoch :" + str(epoch))

            if np.isnan(validation_loss):
                nan_occurrences.append("Validation Loss is NaN at epoch :" + str(epoch))

            if np.isnan(validation_accuracy):
                nan_occurrences.append("Validation Accuracy is NaN at epoch :" + str(epoch))

            # Patience

            # If patience Is ENABLED then go through the patience-logic
            if max_patience != -1:

                if epoch == 1 or best_validation_loss == -1 or best_validation_loss is None or best_acc == -1 or best_acc is None:
                    best_validation_loss = validation_loss
                    best_acc = validation_accuracy
                    best_weights = self.getNetworkWeights(self.network)

                ##First verify when to decrease patience count. Decreasing means losing patience.
                if epoch > 1:

                    # If validation loss is WORSE than best loss, LOSE patience
                    if validation_loss > best_validation_loss:
                        current_patience = current_patience - 1

                    if validation_loss <= best_validation_loss:
                        best_validation_loss = validation_loss
                        best_acc = validation_accuracy
                        current_patience = int(max_patience)
                        best_weights = self.getNetworkWeights(self.network)

                    # Check if theres no more patience and if its time to stop.
                    if current_patience <= 0:
                        print ("\nNo more patience. Stopping training...\n")
                        if best_weights != None:
                            self.setWeights(network=self.network, param_values=best_weights)
                        break

                print ("Current Patience : " + str(current_patience) + " | Max Patience : " + str(max_patience))
                print ("Best Validation Loss : " + str(best_validation_loss) + "\n")

            ##Update learning rate
            if decay != 0:
                update_learning_rate.set_value(self.decayLearningRate(current_learning_rate=update_learning_rate.get_value(), decay=decay))

        if X_test != None and Y_test != None:

            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in self.minibatch_iterator(X_test, Y_test, self.batch_size):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1

            print("Final results:")
            print("  Test Loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  Test Accuracy:\t\t{:.4f} %".format(test_acc / test_batches * 100))

            # Print nan_occurences
            for occurrence in nan_occurrences:
                print(str(occurrence))

            score = (float(test_acc) / float(test_batches))

            return score

    ########################

    def apply(self, x, batch_size=1):

        # Reshape so it fits the network input
        x = x.reshape((-1, self.colour_channels, lasagne.layers.get_all_layers(self.network)[0].shape[2], lasagne.layers.get_all_layers(self.network)[0].shape[3]))

        # Predict
        predict_function = theano.function([self.input_var], lasagne.layers.get_output(self.network, deterministic=True))

        prediction = np.empty((x.shape[0], 8), dtype=np.float32)

        index = 0
        for batch in self.minibatch_iterator_predictor(inputs=x, batch_size=batch_size):
            inputs = batch

            y = predict_function(inputs)

            prediction[index * batch_size:batch_size * (index + 1), :] = y[:]
            index += 1
        return prediction

    def write_state(self, obj_dict={}):

        save_to = str(obj_dict['save_weights_as'])

        np.savez(save_to, *lasagne.layers.get_all_param_values(self.network))

    def set_state(self, obj_dict={}):

        try:
            load_from = str(obj_dict['load_weights_as'])

            with np.load(load_from) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]

            lasagne.layers.set_all_param_values(self.network, param_values)
        except Exception as ex:
            print('Error when attempting to load Weights , Reason :', ex.message)

        return self.network

    def getNetworkInfo(self, network):

        if network == None:
            print ("Failure to Print Network , reason : Parameter == none")
            raise TypeError("Parameter 'network' cannot be None")

        nolearnNetwork = NeuralNet(layers=network, update=lasagne.updates.adam)
        nolearnNetwork.initialize()

        layer_info = nolearn.lasagne.PrintLayerInfo()
        firstInfo = layer_info._get_greeting(nolearnNetwork)

        layer_info, legend = layer_info._get_layer_info_conv(nolearnNetwork)

        # Split it
        splitted_layer_info = layer_info.splitlines()

        # Add The Num Filters
        justify_size = 14
        splitted_layer_info[0] = splitted_layer_info[0] + "FilterSize".rjust(justify_size)
        splitted_layer_info[1] = splitted_layer_info[1] + "------------".rjust(justify_size)

        all_layers = lasagne.layers.get_all_layers(network)
        for x in xrange(0, len(all_layers)):
            layer = all_layers[x]

            if isinstance(layer, lasagne.layers.InputLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "----------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.Conv2DLayer):
                fs = str(layer.filter_size).replace(",", "x")
                fs = fs.replace("(", "")
                fs = fs.replace(")", "")
                fs = fs.replace(" ", "")
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + fs.rjust(justify_size)

            if isinstance(layer, lasagne.layers.DilatedConv2DLayer):
                fs = str(layer.filter_size).replace(",", "x")
                fs = fs.replace("(", "")
                fs = fs.replace(")", "")
                fs = fs.replace(" ", "")
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + fs.rjust(justify_size)

            if isinstance(layer, lasagne.layers.TransposedConv2DLayer):
                fs = str(layer.filter_size).replace(",", "x")
                fs = fs.replace("(", "")
                fs = fs.replace(")", "")
                fs = fs.replace(" ", "")
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + fs.rjust(justify_size)

            if isinstance(layer, lasagne.layers.MaxPool2DLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "----------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.DenseLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "----------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.ConcatLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "----------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.DropoutLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "----------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.NonlinearityLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "----------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.BatchNormLayer) and not isinstance(layer.input_layer, lasagne.layers.InputLayer):
                # network.input_layer.input_layer.filter_size
                fs = str(layer.input_layer.filter_size).replace(",", "x")
                fs = fs.replace("(", "")
                fs = fs.replace(")", "")
                # strip wont work here. Probably due to unicode str type
                fs = fs.replace(" ", "")
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + fs.rjust(justify_size)

        # After added the num filters , now add the pad info
        # Add The pad info
        justify_size = 14
        splitted_layer_info[0] = splitted_layer_info[0] + "Padding".rjust(justify_size)
        splitted_layer_info[1] = splitted_layer_info[1] + "------------".rjust(justify_size)

        all_layers = lasagne.layers.get_all_layers(network)
        for x in xrange(0, len(all_layers)):
            layer = all_layers[x]

            if isinstance(layer, lasagne.layers.InputLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.Conv2DLayer):
                layerpad = str(layer.pad)
                layerpad = layerpad.replace(",", "x")
                layerpad = layerpad.replace("(", "")
                layerpad = layerpad.replace(")", "")
                layerpad = layerpad.replace(" ", "")
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + layerpad.rjust(justify_size)

            if isinstance(layer, lasagne.layers.DilatedConv2DLayer):
                layerpad = str(layer.pad)
                layerpad = layerpad.replace(",", "x")
                layerpad = layerpad.replace("(", "")
                layerpad = layerpad.replace(")", "")
                layerpad = layerpad.replace(" ", "")
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + layerpad.rjust(justify_size)

            if isinstance(layer, lasagne.layers.TransposedConv2DLayer):
                layerpad = str(layer.pad)
                layerpad = layerpad.replace(",", "x")
                layerpad = layerpad.replace("(", "")
                layerpad = layerpad.replace(")", "")
                layerpad = layerpad.replace(" ", "")
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + layerpad.rjust(justify_size)

            if isinstance(layer, lasagne.layers.BatchNormLayer) and not isinstance(layer.input_layer, lasagne.layers.InputLayer):
                layerpad = str(layer.input_layer.pad)
                layerpad = layerpad.replace(",", "x")
                layerpad = layerpad.replace("(", "")
                layerpad = layerpad.replace(")", "")
                layerpad = layerpad.replace(" ", "")
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + layerpad.rjust(justify_size)

            if isinstance(layer, lasagne.layers.MaxPool2DLayer):
                layerpad = str(layer.pad)
                layerpad = layerpad.replace(",", "x")
                layerpad = layerpad.replace("(", "")
                layerpad = layerpad.replace(")", "")
                layerpad = layerpad.replace(" ", "")
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + layerpad.rjust(justify_size)

            if isinstance(layer, lasagne.layers.DenseLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.ConcatLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.DropoutLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.NonlinearityLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

        # After added the pad size , now add the dropout info
        # Add The pad info
        justify_size = 14
        splitted_layer_info[0] = splitted_layer_info[0] + "Dropout".rjust(justify_size)
        splitted_layer_info[1] = splitted_layer_info[1] + "------------".rjust(justify_size)

        all_layers = lasagne.layers.get_all_layers(network)
        for x in xrange(0, len(all_layers)):
            layer = all_layers[x]

            if isinstance(layer, lasagne.layers.InputLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.Conv2DLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.TransposedConv2DLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.DilatedConv2DLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.BatchNormLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.MaxPool2DLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.DenseLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.DropoutLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + str(layer.p).rjust(justify_size)

            if isinstance(layer, lasagne.layers.NonlinearityLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.ConcatLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

        # After added the dropout info , now add the neurons info
        # Add The pad info
        justify_size = 16
        splitted_layer_info[0] = splitted_layer_info[0] + "NonLinearity".rjust(justify_size)
        splitted_layer_info[1] = splitted_layer_info[1] + "--------------".rjust(justify_size)

        all_layers = lasagne.layers.get_all_layers(network)
        for x in xrange(0, len(all_layers)):
            layer = all_layers[x]

            if isinstance(layer, lasagne.layers.InputLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.Conv2DLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + str(str(layer.nonlinearity).split()[1]).rjust(justify_size)

            if isinstance(layer, lasagne.layers.TransposedConv2DLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + str(str(layer.nonlinearity).split()[1]).rjust(justify_size)

            if isinstance(layer, lasagne.layers.DilatedConv2DLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + str(str(layer.nonlinearity).split()[1]).rjust(justify_size)

            if isinstance(layer, lasagne.layers.NonlinearityLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + str(str(layer.nonlinearity).split()[1]).rjust(justify_size)

            if isinstance(layer, lasagne.layers.BatchNormLayer) and not isinstance(layer.input_layer, lasagne.layers.InputLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + str(str(layer.input_layer.nonlinearity).split()[1]).rjust(justify_size)

            if isinstance(layer, lasagne.layers.MaxPool2DLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.DenseLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + str(str(layer.nonlinearity).split()[1]).rjust(justify_size)

            if isinstance(layer, lasagne.layers.ConcatLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

            if isinstance(layer, lasagne.layers.DropoutLayer):
                splitted_layer_info[x + 2] = splitted_layer_info[x + 2] + "-------".rjust(justify_size)

        # join the lines
        layer_info = ""
        for line in splitted_layer_info:
            layer_info = layer_info + "\n" + line

        return firstInfo, layer_info, legend


def split_validation_data(x, y, validation_percentage=0, random_state=1):
    x, x_val, y, y_val = train_test_split(x, y, test_size=validation_percentage / 100.00, random_state=random_state)

    return x, y, x_val, y_val


def script(input_dim, X, Y, X_val, Y_val, test_data, test_id, batch_size=16, max_epochs=30, colour_channels=3, max_patience=-1):
    def getModel(input_var=None, batch_size=None, input_dim=None, output_dim=None, colour_channels=1):
        # Input Layer
        network = lasagne.layers.InputLayer(shape=(batch_size, colour_channels, input_dim.shape[0], input_dim.shape[1]), input_var=input_var)

        # Model
        network = lasagne.layers.Conv2DLayer(network, num_filters=4, filter_size=(3, 3), stride=1, pad=1)
        network = lasagne.layers.Conv2DLayer(network, num_filters=4, filter_size=(3, 3), stride=1, pad=1)
        network = lasagne.layers.MaxPool2DLayer(incoming=network, pool_size=(2, 2), stride=2)

        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=32, nonlinearity=lasagne.nonlinearities.rectify)

        # Output
        network = lasagne.layers.DenseLayer(network, num_units=output_dim, nonlinearity=lasagne.nonlinearities.softmax)

        return network

    ##############
    ##PARAMETERS##



    update_learning_rate = theano.shared(np.float32(1e-2))

    ##Prepare theano variables########
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')
    ##################################

    network = getModel(input_var=input_var, batch_size=None, input_dim=input_dim, output_dim=8, colour_channels=colour_channels)

    loss_or_grads = lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(network), target_var).mean()

    updates = lasagne.updates.nesterov_momentum(loss_or_grads=loss_or_grads, params=lasagne.layers.get_all_params(network, trainable=True), learning_rate=update_learning_rate, momentum=0.90)

    test_loss = lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(network, deterministic=True), target_var).mean()
    test_acc = lasagne.objectives.categorical_accuracy(lasagne.layers.get_output(network, deterministic=True), target_var).mean()

    # Initiate LasagneClassifier
    classifier = LasagneClassifier(verbose=1, input_var=input_var, target_var=target_var, network=network, batch_size=batch_size, channels=3)

    # Start Training
    model_params = {'epochs': max_epochs,
                    'max_patience': max_patience,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'update': updates,
                    'loss_or_grads': loss_or_grads,
                    'testset': (X_val, Y_val),
                    'input_var': input_var,
                    'target_var': target_var,
                    'colour_channels': colour_channels,
                    'update_learning_rate': update_learning_rate,
                    'decay': 1e-6,
                    'validation_set': (X_val, Y_val)}

    dataset = X, Y
    score = classifier.train(dataset=dataset, kw=model_params)

    preds = classifier.apply(x=test_data, batch_size=1)

    return preds, score


##################################################################################################################################

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def run_cross_validation(nfolds, train_data, train_target, train_id, test_data, test_id, seed, batch_size=16, max_epochs=30, colour_channels=3, max_patience=-1):
    yfull_test = []
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=seed)
    num_fold = 0
    fullScore = 0

    for train_index, test_index in kf:
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        test_prediction, score = script(input_dim=input_dim,
                                        X=X_train, Y=Y_train,
                                        X_val=X_valid, Y_val=Y_valid,
                                        test_data=test_data, test_id=test_id,
                                        batch_size=batch_size,
                                        max_epochs=max_epochs,
                                        colour_channels=colour_channels, max_patience=max_patience)
        yfull_test.append(test_prediction)
        fullScore = fullScore + score

    test_res = merge_several_folds_mean(yfull_test, nfolds)

    fullScore = fullScore / num_fold

    return test_res, test_id, fullScore


##################################################################################################################################
#
##################################################################################################################################

batch_size = 32
max_epochs = 2
colour_channels = 3
#max_patience = int((10 * int(max_epochs)) / 100)
max_patience = 1

seed = 8000
lasagne.random.set_rng(np.random.RandomState(seed))
input_dim = np.ones(shape=(16, 16), dtype=np.uint8)

test_data, test_id = read_and_normalize_test_data(rx=input_dim.shape[0], ry=input_dim.shape[1])

train_data, train_target, train_id = read_and_normalize_train_data(rx=input_dim.shape[0], ry=input_dim.shape[1])

##################################################################################################################################
##################################################################################################################################

test_res, test_id, total_score = run_cross_validation(nfolds=10,
                                                      train_data=train_data, train_target=train_target, train_id=train_id,
                                                      test_data=test_data, test_id=test_id,
                                                      seed=seed,
                                                      batch_size=batch_size, max_epochs=max_epochs, colour_channels=colour_channels, max_patience=max_patience)

# Make subm file
create_submission(test_res, test_id, "score_" + str(total_score) + "_")
