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

#from __future__ import division

import six
import numpy as np
import pandas as pd
import cv2
import glob
import random

np.random.seed(2016)
random.seed(2016)

from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

#%%javascript
#IPython.OutputArea.prototype._should_scroll = function(lines) {
#    return false;
#}

conf = dict()

# How many patients will be in train and validation set during training. Range: (0; 1)
conf['train_valid_fraction'] = 0.75

# Batch size for CNN [Depends on GPU and memory available]
conf['batch_size'] = 1

# Number of epochs for CNN training
#conf['nb_epoch'] = 200
conf['nb_epoch'] = 1

# Early stopping. Stop training after epochs without improving on validation
conf['patience'] = 3

# Shape of image for CNN (Larger the better, but you need to increase CNN as well)
#conf['image_shape'] = (4160,4128)
#conf['image_shape'] = (2080,2064)
#conf['image_shape'] = (1024,1024)
conf['image_shape'] = (64,64)

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)
def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(input)
        return _bn_relu(conv)

    return f

def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(activation)

    return f
    
def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual_shape[CHANNEL_AXIS],
                                 nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid",
                                 W_regularizer=l2(0.0001))(input)

    return merge([shortcut, residual], mode="sum")

def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filter=nb_filter, init_subsample=init_subsample,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def basic_block(nb_filter, init_subsample=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Convolution2D(nb_filter=nb_filter,
                                 nb_row=3, nb_col=3,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3,
                                  subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3)(conv1)
        return _shortcut(input, residual)

    return f

def bottleneck(nb_filter, init_subsample=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of nb_filter * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Convolution2D(nb_filter=nb_filter,
                                 nb_row=1, nb_col=1,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=1, nb_col=1,
                                     subsample=init_subsample)(input)

        conv_3_3 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3)(conv_1_1)
        residual = _bn_relu_conv(nb_filter=nb_filter * 4, nb_row=1, nb_col=1)(conv_3_3)
        return _shortcut(input, residual)

    return f

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3
def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

        block = pool1
        nb_filter = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block)
            nb_filter *= 2

        # Last activation
        block = _bn_relu(block)

        block_norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(block)
        block_output = Activation("relu")(block_norm)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block_output)
        flatten1 = Flatten()(pool2)
        dense = Dense(output_dim=num_outputs, init="he_normal", activation="softmax")(flatten1)
        #dense = Dense(output_dim=num_outputs, W_regularizer=l2(0.01), init="he_normal", activation="linear")(flatten1)

        model = Model(input=input, output=dense)
        return model

    @staticmethod
    def build_resnet_test(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [1, 1, 1, 1])

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])

def batch_generator_train(files, batch_size):
    number_of_batches = np.ceil(len(files)/batch_size)
    counter = 0
    random.shuffle(files)
    while True:
        batch_files = files[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []
        for f in batch_files:
            image = cv2.imread(f)
            image = cv2.resize(image, conf['image_shape'])

            cancer_type = f[20:21] # relies on path lengths that is hard coded below
            if cancer_type == '1':
                mask = [1, 0, 0]
            elif cancer_type == '2':
                mask = [0, 1, 0]
            else:
                mask = [0, 0, 1]

            image_list.append(image)
            mask_list.append(mask)
        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)

        yield image_list, mask_list

        if counter == number_of_batches:
            random.shuffle(files)
            counter = 0

# file paths to training and additional samples
filepaths = []
filepaths.append('../input/train/Type_1/')
filepaths.append('../input/train/Type_2/')
filepaths.append('../input/train/Type_3/')

allFiles = []

for i, filepath in enumerate(filepaths):
    files = glob.glob(filepath + '*.jpg')
    allFiles = allFiles + files

split_point = int(round(conf['train_valid_fraction']*len(allFiles)))

random.shuffle(allFiles)

train_list = allFiles[:split_point]
valid_list = allFiles[split_point:]
print('Train patients: {}'.format(len(train_list)))
print('Valid patients: {}'.format(len(valid_list)))

print('Create and compile model...')

nb_classes = 3
img_rows, img_cols = conf['image_shape'][1], conf['image_shape'][0]
img_channels = 3

model = ResnetBuilder.build_resnet_34((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='hinge',optimizer='adadelta',metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=conf['patience'], verbose=0),
    ModelCheckpoint('cervical_best.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
]

print('Fit model...')
fit = model.fit_generator(generator=batch_generator_train(train_list, conf['batch_size']),
                      nb_epoch=conf['nb_epoch'],
                      #samples_per_epoch=len(train_list),
                      samples_per_epoch=3,
                      validation_data=batch_generator_train(valid_list, conf['batch_size']),
                      #nb_val_samples=len(valid_list),
                      nb_val_samples=1,
                      verbose=1,
                      callbacks=callbacks)

#from keras.models import load_model
#model = load_model('cervical_best.hdf5')

sample_subm = pd.read_csv("../input/sample_submission.csv")
ids = sample_subm['image_name'].values

for id in ids:
    print('Predict for image {}'.format(id))
    files = glob.glob("../input/test/" + id)
    image_list = []
    for f in files:
        image = cv2.imread(f)
        image = cv2.resize(image, conf['image_shape'])
        image_list.append(image)
        
    image_list = np.array(image_list)

    predictions = model.predict(image_list, verbose=1, batch_size=1)

    sample_subm.loc[sample_subm['image_name'] == id, 'Type_1'] = predictions[0,0]
    sample_subm.loc[sample_subm['image_name'] == id, 'Type_2'] = predictions[0,1]
    sample_subm.loc[sample_subm['image_name'] == id, 'Type_3'] = predictions[0,2]
    
sample_subm.to_csv("subm.csv", index=False)




