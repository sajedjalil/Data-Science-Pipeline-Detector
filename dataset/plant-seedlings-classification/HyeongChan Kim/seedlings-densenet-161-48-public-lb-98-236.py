import pandas as pd

import numpy as np

import random

import shutil

import os



from keras.applications import xception

from keras.optimizers import adam, SGD, rmsprop

from keras.models import Model

from keras.layers import Input, merge, ZeroPadding2D

from keras.layers.core import Layer, Dense, Dropout, Activation

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2



from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler



from keras.engine import InputSpec

from keras import initializers as initializations



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator



import keras.backend as K





model_name = "DenseNet-161"

# model_name = "Xception"



# data_set_path = "./plant_seedlings/"
data_set_path = "../input/"


categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

num_categories = len(categories)



lr = 2e-4  # 8e-4

batch_size = 8

num_epochs = 200

pre_train_epochs = 20

num_steps_per_epoch = 300

img_rows, img_cols, img_channel = 224, 224, 3



num_snapshots = 6  # for ensemble





class SnapshotModelCheckpoint(Callback):

    """

    Callback that saves the snapshot weights of the model.

    Saves the model weights on certain epochs (which can be considered the

    snapshot of the model at that epoch).

    Should be used with the cosine annealing learning rate schedule to save

    the weight just before learning rate is sharply increased.

    # Arguments:

        nb_epochs: total number of epochs that the model will be trained for.

        nb_snapshots: number of times the weights of the model will be saved.

        fn_prefix: prefix for the filename of the weights.

    """



    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):

        super(SnapshotModelCheckpoint, self).__init__()



        self.check = nb_epochs // nb_snapshots

        self.fn_prefix = fn_prefix



    def on_epoch_end(self, epoch, logs={}):

        if not epoch == 0 and (epoch + 1) % self.check == 0:

            file_path = data_set_path + "model/" + self.fn_prefix + "-%d.h5" % ((epoch + 1) // self.check)

            # self.model.save_weights(file_path, overwrite=True)  # disable this code



            print("[*] Saved snapshot at model/%s_%d.h5" % (self.fn_prefix, (epoch + 1) // self.check))





class SnapshotCallbackBuilder:

    """

    Callback builder for snapshot ensemble training of a model.

    Creates a list of callbacks, which are provided when training a model

    so as to save the model weights at certain epochs, and then sharply

    increase the learning rate.

    """



    def __init__(self, nb_epochs, nb_snapshots, init_lr=1e-3):

        """

        Initialize a snapshot callback builder.

        # Arguments:

            nb_epochs: total number of epochs that the model will be trained for.

            nb_snapshots: number of times the weights of the model will be saved.

            init_lr: initial learning rate

        """

        self.T = nb_epochs

        self.M = nb_snapshots

        self.alpha_zero = init_lr



    def get_callbacks(self, model_prefix='Model'):

        """

        Creates a list of callbacks that can be used during training to create a

        snapshot ensemble of the model.

        Args:

            model_prefix: prefix for the filename of the weights.

        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,

                 SnapshotModelCheckpoint] which can be provided to the 'fit' function

        """



        callback_list = [ModelCheckpoint(filepath=data_set_path + "model/%s_acc-best.h5" % model_prefix,

                                         monitor="val_acc", save_best_only=True, save_weights_only=True, verbose=1),

                         LearningRateScheduler(schedule=self._cosine_anneal_schedule),

                         # LearningRateScheduler(schedule=lr_schedule),

                         # ReduceLROnPlateau(monitor='val_loss',

                         #                   patience=12, factor=0.75, cooldown=10, min_lr=1e-5, verbose=1),

                         EarlyStopping('val_loss', patience=30, verbose=1),

                         SnapshotModelCheckpoint(self.T, self.M, fn_prefix='%s' % model_prefix)]



        return callback_list



    def _cosine_anneal_schedule(self, t):

        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.

        cos_inner /= self.T // self.M

        cos_out = np.cos(cos_inner) + 1



        return float(self.alpha_zero / 2 * cos_out)





class Scale(Layer):

    """

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

    """



    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):

        self.momentum = momentum

        self.axis = axis

        self.beta_init = initializations.get(beta_init)

        self.gamma_init = initializations.get(gamma_init)

        self.initial_weights = weights



        super(Scale, self).__init__(**kwargs)



    def build(self, input_shape):

        self.input_spec = [InputSpec(shape=input_shape)]

        shape = (int(input_shape[self.axis]),)



        # Compatibility with TensorFlow >= 1.0.0

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





def densenet161_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.5,

                      dropout_rate=0.0, weight_decay=1e-4, num_classes=None, lr=8e-4):

    """

    DenseNet 161 Model for Keras

    Model Schema is based on

    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights

    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfVnlCMlBGTDR3RGs

    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfUDZwVjU2cFNidTA

    # Arguments

        nb_dense_block: number of dense blocks to add to end

        growth_rate: number of filters to add per dense block

        nb_filter: initial number of filters

        reduction: reduction factor of transition blocks.

        dropout_rate: dropout rate

        weight_decay: weight decay factor

        classes: optional number of classes to classify images

        weights_path: path to pre-trained weights

    # Returns

        A Keras model instance.

    """



    eps = 1.1e-5



    # compute compression factor

    compression = 1.0 - reduction



    # Handle Dimension Ordering for different backends

    global concat_axis



    concat_axis = 3

    img_input = Input(shape=(img_rows, img_cols, 3), name='data')



    # From architecture for ImageNet (Table 1 in the paper)

    nb_filter = 96

    nb_layers = [6, 12, 36, 24]  # For DenseNet-161



    # Initial convolution

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)

    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)

    x = Scale(axis=concat_axis, name='conv1_scale')(x)

    x = Activation('relu', name='relu1')(x)

    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)



    # Add dense blocks

    for block_idx in range(nb_dense_block - 1):

        stage = block_idx+2

        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,

                                   weight_decay=weight_decay)



        # Add transition_block

        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,

                             weight_decay=weight_decay)

        nb_filter = int(nb_filter * compression)



    final_stage = stage + 1

    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,

                               weight_decay=weight_decay)



    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)

    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)

    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)



    # x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    # x_fc = Dense(1000, name='fc6')(x_fc)

    # x_fc = Activation('softmax', name='prob')(x_fc)



    model = Model(img_input, x, name='densenet')



    # weight path

    weights_path = './plant_seedlings/model/densenet161_weights_tf.h5'



    model.load_weights(weights_path, by_name=True)



    # Truncate and replace softmax layer for transfer learning

    # Cannot use model.layers.pop() since model is not of Sequential() type

    # The method below works since pre-trained weights are stored in layers but not in the model

    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    x_newfc = Dense(num_classes, name='fc6')(x_newfc)

    x_newfc = Activation('softmax', name='prob')(x_newfc)



    model = Model(img_input, x_newfc)



    # Learning rate is changed to 0.001

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])



    return model





def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):

    """

    Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout

        # Arguments

            x: input tensor

            stage: index for dense block

            branch: layer index within each dense block

            nb_filter: number of filters

            dropout_rate: dropout rate

            weight_decay: weight decay factor

    """



    eps = 1.1e-5

    conv_name_base = 'conv' + str(stage) + '_' + str(branch)

    relu_name_base = 'relu' + str(stage) + '_' + str(branch)



    # 1x1 Convolution (Bottleneck layer)

    inter_channel = nb_filter * 4

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)

    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)

    x = Activation('relu', name=relu_name_base+'_x1')(x)

    x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)



    if dropout_rate:

        x = Dropout(dropout_rate)(x)



    # 3x3 Convolution

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)

    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)

    x = Activation('relu', name=relu_name_base+'_x2')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)

    x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)



    if dropout_rate:

        x = Dropout(dropout_rate)(x)



    return x





def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1e-4):

    """

    Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout

        # Arguments

            x: input tensor

            stage: index for dense block

            nb_filter: number of filters

            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.

            dropout_rate: dropout rate

            weight_decay: weight decay factor

    """



    eps = 1.1e-5

    conv_name_base = 'conv' + str(stage) + '_blk'

    relu_name_base = 'relu' + str(stage) + '_blk'

    pool_name_base = 'pool' + str(stage)



    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)

    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)

    x = Activation('relu', name=relu_name_base)(x)

    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)



    if dropout_rate:

        x = Dropout(dropout_rate)(x)



    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)



    return x





def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,

                grow_nb_filters=True):

    """

    Build a dense_block where the output of each conv_block is fed to subsequent ones

        # Arguments

            x: input tensor

            stage: index for dense block

            nb_layers: the number of layers of conv_block to append to the model.

            nb_filter: number of filters

            growth_rate: growth rate

            dropout_rate: dropout rate

            weight_decay: weight decay factor

            grow_nb_filters: flag to decide to allow number of filters to grow

    """



    eps = 1.1e-5

    concat_feat = x



    for i in range(nb_layers):

        branch = i + 1

        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)

        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis,

                            name='concat_' + str(stage) + '_' + str(branch))



        if grow_nb_filters:

            nb_filter += growth_rate



    return concat_feat, nb_filter





def split_val_data(split_rate=0.2):

    val_path = data_set_path + "val/"

    train_path = data_set_path + "train/"



    if os.path.exists(val_path):

        return False

    else:

        # make validation data set path

        os.mkdir(val_path)



        for cat in categories:

            os.mkdir(val_path + cat)



            train_path_name = os.listdir(train_path + cat)

            # shuffling files

            random.shuffle(train_path_name)



            # split data set

            to_val = train_path_name[:int(len(train_path_name) * split_rate)]



            for f in to_val:

                shutil.move(os.path.join(train_path, cat, f), os.path.join(val_path, cat))



        return True





def read_img(file_path, size=(img_rows, img_cols), data_dir=data_set_path):

    img = image.load_img(os.path.join(data_dir, file_path), target_size=size)

    img = image.img_to_array(img)



    return img





# split train/val data

split_data = False  # just done with once at initial time

if split_data:

    split_val_data()



# Callbacks

snapshot = SnapshotCallbackBuilder(num_epochs, num_snapshots, lr)



train_data_gen = ImageDataGenerator(

    rescale=1. / 255,

    rotation_range=60,

    width_shift_range=0.30,

    height_shift_range=0.30,

    shear_range=0.25,

    zoom_range=0.25,

    horizontal_flip=True,

    vertical_flip=True,

    fill_mode='nearest')

train_generator = train_data_gen.flow_from_directory(

    directory=data_set_path + "train/",

    target_size=(img_cols, img_rows),

    batch_size=batch_size,

    class_mode='categorical',

    shuffle=True

)



val_data_gen = ImageDataGenerator(

    rescale=1. / 255,

)

val_generator = val_data_gen.flow_from_directory(

    directory=data_set_path + "val/",

    target_size=(img_cols, img_rows),

    batch_size=batch_size,

    class_mode='categorical',

    shuffle=True

)



# building model

model = densenet161_model(img_rows=img_rows, img_cols=img_cols, color_type=img_channel, num_classes=num_categories,

                          lr=lr)

# base_model = xception.Xception(include_top=False, weights='imagenet',

#                                input_shape=(img_rows, img_cols, img_channel), pooling='avg')



"""

x = base_model.output

x = Dropout(.5)(x)

x = Dense(num_categories, activation='softmax')(x)



model = Model(inputs=base_model.input, outputs=x)

"""



# print model summary

model.summary()



for layer in model.layers:

    layer.W_regularizer = l2(1e-2)

    layer.trainable = True



opt = SGD(lr=lr, decay=1e-8, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



# load for re-training

# model.load_weights(data_set_path + 'model/%s_acc-best.h5' % model_name)

"""

model.fit_generator(

    generator=train_generator,

    validation_data=val_generator,

    steps_per_epoch=num_steps_per_epoch,

    callbacks=snapshot.get_callbacks(model_prefix=model_name),

    # initial_epoch=pre_train_epochs,

    epochs=num_epochs,

    shuffle=True,

    verbose=1)



# save trained weights

train_end_weights = data_set_path + 'model/%s_train-end.h5' % model_name



model.save_weights(train_end_weights, overwrite=True)

"""

# load test images

test_set = pd.read_csv(data_set_path + "sample_submission.csv")



test_fn = []

test_img = np.zeros((794, img_rows, img_cols, img_channel), dtype=np.float32)

for i, f in enumerate(test_set['file']):

    test_fn.append(f)



    img = read_img("test/" + f) / 255.

    img = np.expand_dims(img[:], axis=0)



    test_img[i] = img



print("[*] test image shape : {} size : {:,}".format(test_img.shape, test_img.size))



model_file_name = [data_set_path + "model/%s_acc-best.h5" % model_name]

for i in range(1, num_snapshots + 1):

    model_file_name.append(data_set_path + "model/%s-%d.h5" % (model_name, i))



# Ensemble

test_preds = []

saved_total = 0

for weight in model_file_name:

    try:

        model.load_weights(weight, by_name=True)

    except OSError:

        continue



    print("[*] predict with %s" % weight.split('/')[-1])



    test_pred = model.predict(test_img, batch_size=batch_size)

    test_preds.append(test_pred)



    saved_total += 1



pred_weights = [1. / saved_total] * saved_total



weighted_predictions = np.zeros((794, num_categories), dtype='float32')

for weight, prediction in zip(pred_weights, test_preds):

    weighted_predictions += weight * prediction



preds = np.argmax(weighted_predictions, axis=1)

print(preds, preds.shape)



test_set['species'] = [categories[c] for c in preds]

test_set.to_csv(data_set_path + "submit.csv", index=None)



# This is how i actually did.

# 1. Running this script, many times. 

# 2. Getting best-acc.h5 (model file) from each runnings.

# 2-1. (After first running, use beforehand best-acc.h5 to initialize model weights instead of orignal DenseNet-161-48 pre-trained model.)

# 3. Ensembling all of it! (maybe 4~5 best-acc models), (except SnapShoted model file generated by SnapShot Ensemble Callback)

# 4. Then, submit!



# Hyper-parameters

# 1. Batch Size : Because of my GPU spec..., i just use 8 batch_size instead of 16.

# 2. Weight Regularizer : I just test all of cases from 1e-2 to 5e-4.

# 3. Opzimier : SGD with LR 2e-4, nestorv



# Callbacks

# 1. Learning Rate Scheduler : I used this one. https://github.com/titu1994/Snapshot-Ensembles

# 2. Eaerly Stop : patience 30 by 'val_loss'

# 3. ...



# Augumentation

# 1. shift range : maybe 0.2 to 0.3 is proper.

# 2. rotate range : there's no problem setting rot range to 360 on this DataSet (plant seedlings). But i just try some cases (~60 and 360) with same model, then ~60 got higher acc than 360.

# 3. flip : horizontal & vertical_flip could be find.



# Model

# I used many models like Xception, Inception-v3, VGG16, DenseNet-161-48, ....

# But, on my machine, DensenNet-161-48 got highest test acc with same hyper-parameters. So i just used it... :)

# 

# Here is DenseNet-161-48 pre-trained file link : https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-161-48-no-top.h5

# Which got from here : https://github.com/titu1994/DenseNet



# Train/Validation/Test

# Train : Val = 0.85 : 0.15



# Maybe, on average, about 100 mins are taken on GPU 1060 6GB (because of EealyStop Callback).