# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.engine import Layer, InputSpec
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
#from keras.utils.multi_gpu_utils import multi_gpu_model
from sklearn.model_selection import train_test_split
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
import keras.backend as K


Batch_size = 4
Epochs = 15
USE_MultiGPU = False
MODEL = "DenseNet"
#resnet parameters
img_channels = 1
img_cols = 512
img_rows = 512
nb_classes = 5
train_image_path = "/kaggle/input/aptos2019-blindness-detection/train_images/"
train_label_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
predict_image_path = "/kaggle/input/aptos2019-blindness-detection/test_images/"
predict_label_path = "/kaggle/input/aptos2019-blindness-detection/test.csv"
predict_output_path = "/Kaggle/input/aptos2019-blindness-detection/output_test.csv"

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
#train_generator = aug.flow_from_dataframe(train_label_path, train_image_path, x_col="id_code", y_col="diagnosis", has_ext=False)

def generator(x_list, y_list,image_path):
    i = 0
    while True:
        images = []
        labels = []
        for m in range(Batch_size):
            if i == len(x_list):
                i = 0
            img_path = image_path + x_list[i]
            image_data = cv2.imread(img_path)
            #image_data = convert_image(image_data)
            image_data = crop_image_from_gray(image_data)
            image_data = preprocess_image(image_data)
            image_data = image_data.astype(np.float32)
            images.append(image_data)
            labels.append(y_list[i])
            i = i + 1
        images = np.array(images)
        labels = np.array(labels)
        #数据增强
        (images, labels) = next(aug.flow(np.array(images), labels, batch_size=Batch_size))
        yield images, labels


def convert_image(img):
    #裁剪
    #if img.shape[0] > img.shape[1]:
        #img = img[(img.shape[0]-img.shape[1])//2:(img.shape[0]-img.shape[1])//2 + img.shape[1], :]
    #elif img.shape[1] > img.shape[0]:
        #img = img[:, (img.shape[1]-img.shape[0])//2: (img.shape[1]-img.shape[0])//2 + img.shape[0]]
    #填充
    if img.shape[0] > img.shape[1]:
        img = cv2.copyMakeBorder(img, 0, 0, (img.shape[0]-img.shape[1])//2, (img.shape[0]-img.shape[1])//2,cv2.BORDER_CONSTANT, value=0)
    elif img.shape[1] > img.shape[0]:
        img = cv2.copyMakeBorder(img, (img.shape[1]-img.shape[0])//2, (img.shape[1]-img.shape[0])//2, 0, 0, cv2.BORDER_CONSTANT, value=0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray, gray)
    gray = cv2.resize(gray, (512, 512), cv2.INTER_CUBIC)
    gray = gray.reshape(512, 512, 1)
    return gray


def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and
    returns the a preprocessed image with
    3 channels
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def preprocess_image(image, sigmaX=10):
    """
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (img_cols, img_rows))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


class Scale(Layer):
    '''Custom Layer for DenseNet used for BatchNormalization.

    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        # Tensorflow >= 1.0.0 compatibility
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        # self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        # self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def DenseNet(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.8, weight_decay=1e-4,
             classes=2, weights_path=None):
    '''Instantiate the DenseNet 121 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks. 1-theta
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    #if K.image_dim_ordering() == 'tf':
    concat_axis = 3  # 通道数在第三维
    img_input = Input(shape=(512, 512, 3), name='data')
    #else:
    #    concat_axis = 1
     #   img_input = Input(shape=(3, 224, 224), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6, 12, 24, 16]  # For DenseNet-121

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)  # (?,112,112,64)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)  # (?,56,56,64)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)
    x = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)

    x = Dense(classes, name='fc6')(x)
    x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet')

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base + '_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base + '_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)

        concat_feat = concatenate([concat_feat, x], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def train_predict():
    #load data
    label = pd.read_csv(train_label_path, index_col="id_code")
    imglist = []
    labellist = []
    for m in os.listdir(train_image_path):
        imglist.append(m)
        labels = label.loc[m[:-4], "diagnosis"]
        labels = to_categorical(labels, 5)
        labellist.append(labels)
    train_imglist, test_imglist, train_label, test_label = train_test_split(imglist, labellist, test_size=0.2)
    #begin train
    if MODEL == "ResNet":
        pass
    elif MODEL == "DenseNet":
        model = DenseNet(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.8, weight_decay=1e-4, classes=5, weights_path=None)
    optimizer = Adam(lr=1e-4)
    get_train_generator = generator(train_imglist, train_label, train_image_path)
    get_val_generator = generator(test_imglist, test_label, train_image_path)

    callbacks_lists = [
        keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=1,mode='min'),
        keras.callbacks.ReduceLROnPlateau(monitor='var_loss',factor=0.1,patience=10)]

    if USE_MultiGPU:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        parallel_model.fit_generator(generator=get_train_generator,
                            steps_per_epoch=(len(train_imglist) // Batch_size),
                            epochs=Epochs,
                            validation_data=get_val_generator,
                            validation_steps=(len(test_imglist) // Batch_size))
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        model.fit_generator(generator=get_train_generator,
                            steps_per_epoch=(len(train_imglist) // Batch_size),
                            epochs=Epochs,
                            callbacks=callbacks_lists,
                            validation_data=get_val_generator,
                            validation_steps=(len(test_imglist) // Batch_size))
    #model.save("/kaggle/input/aptos2019-blindness-detection/model.h5")
    predict_img_list = []
    for img_name in os.listdir(predict_image_path):
        predict_img_list.append(img_name)
    #load label list
    labeldata = pd.read_csv(predict_label_path, index_col="id_code")
    labeldata['diagnosis'] = None
    for i in predict_img_list:
        img = cv2.imread(predict_image_path + i)
        #img = convert_image(img)
        img = crop_image_from_gray(img)
        img = preprocess_image(img)
        img = np.expand_dims(img, axis=0)
        predict = model.predict(img)
        label = np.argmax(predict)
        labeldata.loc[i[:-4],'diagnosis'] = str(label)
    labeldata.to_csv('submission.csv',index=False)


if __name__ == "__main__":
    train_predict()


