# full code can be found here: https://github.com/evagian/Human-Protein-Atlas-Image-Classification-SENet/

import os
from matplotlib import pyplot as plt
import numpy as np
import glob
import pandas as pd
import random
import cv2
from sklearn.utils import shuffle

#create custom color maps
cdict1 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

plt.register_cmap(name='greens', data=cdict1)

# number of images in batch
batch_size = 8
# models are saved here
ckpt_dir = './checkpoint'
# samples are saved here
sample_dir = './sample'


train_set = 'train'

#import training data
train_labels = pd.read_csv("../input/human-protein-atlas-image-classification-dataset/kaggle_protein_classes_augmented_one_hot.csv")
train_labels = train_labels.sort_values(by=['Id']).reset_index()
print(train_labels.head())
print(len(train_labels))

seed = 100
train_labels = shuffle(train_labels, random_state=seed).reset_index(drop=True)
print(train_labels.head())
print(len(train_labels))

datay = train_labels.loc[101:200]

datay = datay.iloc[:, 4:]
# get image id
data_im_id = train_labels.loc[101:200, "Id"]


# read train data files
data_files = []
for im_id in data_im_id:
    data_files.append(glob.glob('./input/train/{}_green.png'.format(im_id)))
#data_files.sort()
filepaths = [''.join(x) for x in data_files]

print("testing code on few images... change lines 46 and 50 to include more training images")
print("number of training data x", len(filepaths))

# data augmentation options
def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 3:
        # flip left and right
        return np.flipud(image)

# data augmentation xDATA_AUG_TIMES times
DATA_AUG_TIMES = 1
count = 0

# calculate the number of patches
step = 0
pat_size = 300
stride = 300

count = len(filepaths)

origin_patch_num = count * DATA_AUG_TIMES
print('origin_patch_num', origin_patch_num)
numClasses = 28
bat_size = 8
# use power of 2 bat_size ex 128

# calculate number of batches and patches
# must be whole number
if origin_patch_num % bat_size != 0:
    numPatches = (divmod(origin_patch_num, bat_size)[0]+1) * bat_size

else:
    numPatches = origin_patch_num
print("total patches =", numPatches, ", batch size =", bat_size,
      ", total batches =", numPatches / bat_size)

# data matrix 4-D
numPatches = int(numPatches)
inputs = np.zeros((numPatches, pat_size, pat_size, 1), dtype="uint8")
inputsy = np.zeros((numPatches, numClasses), dtype="uint8")

count = 0

# generate patches
for i in range(len(filepaths)):
    # get image id
    im_id = data_im_id.iloc[i]

    # open x
    img_s = cv2.imread('../input/human-protein-atlas-image-classification/train/{}_green.png'.format(im_id), 0)


    # open y
    imgy =  datay.iloc[i]


    img_sy = np.array(imgy, dtype="uint8")

    # data augmentation
    for j in range(DATA_AUG_TIMES):
        im_h, im_w= img_s.shape



        z = random.randint(0, 212)
        inputs[count, :, :, 0] = data_augmentation(
            img_s[z:z + pat_size, z:z + pat_size], random.randint(0, 3))

        inputsy[count, :] = img_sy

        count += 1

#imgplot = plt.imshow(inputs[count-9, :, :,0] )
#plt.show()

# pad training examples into the empty patches of the last batch
if count < numPatches:
    to_pad = numPatches - count
    inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
    inputsy[-to_pad:, :] = inputsy[:to_pad, :]



#imgplot = plt.imshow(inputs[count-9, :, :,0] )
#plt.show()


# directory of patches
save_dir='../Human Protein Atlas Image Classification Dataset/'


# save x
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
np.save(os.path.join(save_dir,
                     "protein_image_patches_pat_size_300_bat_size_8_101_200"),
        inputs)
print("size of x inputs tensor = ", str(inputs.shape))

# save y

np.save(os.path.join(save_dir,
                     "protein_image_classes_pat_size_300_bat_size_8_101_200"),
        inputsy)
print("size of y inputs tensor = ", str(inputsy.shape))

### up to here it generates patches


# train the model (SENet architecture)

import gc
from sklearn.utils import shuffle
from PIL import Image
import time
import random
from glob import glob
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
import tensorflow as tf
import sklearn
import pandas as pd
from PIL import Image
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from keras.losses import binary_crossentropy, categorical_crossentropy
import keras.backend as K
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



class train_datatrain_da():
    def __init__(self, filepath):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)

        # np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


# data augmentation options# data a
def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 3:
        # flip left and right
        return np.flipud(image)


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = cv2.imread(filelist, 0)

        #im = Image.open(filelist)#.convert('L')
        return(im) #np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:

        im = cv2.imread(file, 0)

        #im = Image.open(file)#.convert('L')
        data.append(im) #(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data


def multi_label_hot(prediction, threshold=0.5):
    prediction = tf.cast(prediction, tf.float32)
    threshold = float(threshold)
    return tf.cast(tf.greater(prediction, threshold), tf.float32)


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return per_entry_cross_ent


def f1_loss(y_true, y_pred):
    tp = K.sum(y_true * y_pred, axis=0)
    tn = K.sum((1 - y_true) * (1 - y_pred), axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    # 1 - K.mean(f1)
    return 1 - f1


momentum = 0.9
cardinality = 2 #original  8 # how many split ?
blocks = 2 # original 3 # 3 res_block ! (split + transition)
depth = 64  # out channel

reduction_ratio = 4
batch_size = 8  # 128

img_channels = 1
class_num = 28
weight_decay = 0.0005


def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding=padding)
        return network


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Relu(x):
    return tf.nn.relu(x)


def Sigmoid(x):
    return tf.nn.sigmoid(x)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Fully_connected(x, units=class_num, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=False, units=units)


def first_layer(x, scope, training):
    with tf.name_scope(scope):
        x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope + '_conv1')
        x = Batch_Normalization(x, training=training, scope=scope + '_batch1')
        x = Relu(x)

        return x


def transform_layer(x, stride, scope, training=True):
    with tf.name_scope(scope):
        x = conv_layer(x, filter=depth, kernel=[1, 1], stride=1, layer_name=scope + '_conv1')
        x = Batch_Normalization(x, training=training, scope=scope + '_batch1')
        x = Relu(x)

        x = conv_layer(x, filter=depth, kernel=[3, 3], stride=stride, layer_name=scope + '_conv2')
        x = Batch_Normalization(x, training=training, scope=scope + '_batch2')
        x = Relu(x)
        return x


def transition_layer(x, out_dim, scope, training=True):
    with tf.name_scope(scope):
        x = conv_layer(x, filter=out_dim, kernel=[1, 1], stride=1, layer_name=scope + '_conv1')
        x = Batch_Normalization(x, training=training, scope=scope + '_batch1')
        # x = Relu(x)

        return x


def split_layer(input_x, stride, layer_name, training=True):
    with tf.name_scope(layer_name):
        layers_split = list()
        for i in range(cardinality):
            splits = transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i), training=training)
            layers_split.append(splits)

        return Concatenation(layers_split)


def squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input_x * excitation

        return scale


def residual_layer(input_x, out_dim, layer_num, res_block=blocks, training=True):
    # split + transform(bottleneck) + transition + merge
    # input_dim = input_x.get_shape().as_list()[-1]

    for i in range(res_block):
        input_dim = int(np.shape(input_x)[-1])

        if input_dim * 2 == out_dim:
            flag = True
            stride = 2
            channel = input_dim // 2
        else:
            flag = False
            stride = 1

        x = split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i), training = training )
        x = transition_layer(x, out_dim=out_dim, scope='trans_layer_' + layer_num + '_' + str(i),
                             training=training)
        x = squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio,
                                     layer_name='squeeze_layer_' + layer_num + '_' + str(i))

        if flag is True:
            pad_input_x = Average_pooling(input_x)
            pad_input_x = tf.pad(pad_input_x,
                                 [[0, 0], [0, 0], [0, 0], [channel, channel]])  # [?, height, width, channel]
        else:
            pad_input_x = input_x

        input_x = Relu(x + pad_input_x)

    return input_x


# the model
def SE_ResNeXt(input, is_training=True, output_channels=1):
    training = is_training

    input = first_layer(input, scope='first_layer', training=training)

    x = residual_layer(input, out_dim=64, layer_num='1', training=training)  # out_dim=64
    
    # activate layer_num='2' and layer_num='3'
    # x = residual_layer(x, out_dim=128, layer_num='2', training = training)
    # x = residual_layer(x, out_dim=256, layer_num='3' , training = training)

    x = Global_Average_Pooling(x)
    x = flatten(x)

    x = Fully_connected(x, layer_name='final_fully_connected')
    return x


def roc_auc_score(y_true, y_pred):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):
        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))
        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)
        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p = 3
        difference = tf.zeros_like(pos * neg) + pos - neg - gamma
        masked = tf.boolean_mask(difference, difference < 0.0)
        return tf.reduce_sum(tf.pow(-masked, p))


class denoiser(object):
    def __init__(self, sess, input_c_dim=1, batch_size=8):
        self.sess = sess
        input_c_dim = input_c_dim

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # input images
        #self.image_size = 512
        self.img_channels = 1
        self.weight_decay = 0.0005

        self.class_num = 28
        self.X = tf.placeholder(tf.float32, shape=[None, None, None , self.img_channels],
                                name='input_image')

        self.p = tf.placeholder(tf.float32, shape=[None, self.class_num], name='actual_classes')
        # actual classes
        self.one_hot_prediction = tf.placeholder(tf.float32, shape=[None, self.class_num], name='one_hot_prediction')

        # predicted classes
        self.logit_q = tf.placeholder(tf.float32, shape=[None, self.class_num], name='input_classes_logit')
        # self.q = tf.placeholder(tf.float32, shape=[None, self.class_num],  name='input_classes')

        # predicted probabilities
        self.logit_q = SE_ResNeXt(self.X, is_training=self.is_training)

        # learning rate
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        # prediction after softmax
        self.one_hot_prediction = multi_label_hot(self.logit_q)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p, logits=self.logit_q))



        # f1_score

        # self.eva_f1_score = sklearn.metrics.f1_score(self.p, self.q, labels=None, pos_label=1, average='micro') # or average='weighted' and 'samples'

        self.eva_f1_score = f1(self.p, self.one_hot_prediction)
        # adam optimizer
        # default variables
        # beta one 0.9
        # beta two 0.999
        # Epsilon 10^-8
        # beta1=0.9, beta2=0.999, epsilon=1e-08
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           name='AdamOptimizer')

        # returns the list of values in the collection with the given name
        # UPDATE_OPS is a collection of ops (operations performed when the
        # graph runs, like multiplication, ReLU, etc.), not variables.
        # Specifically, this collection maintains a list of ops which
        # need to run after every training step.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # min loss
            self.train_op = optimizer.minimize(self.loss)
        # initialize variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, eval_files, eval_datay, sample_dir,
                 summary_merged, summary_writer):
        # assert test_data value range is 0-255
        outstreval = "[*] Evaluating...\n"

        print("[*] Evaluating...")
        f1_score_sum = 0

        input_image = np.zeros((1, 512, 512, 1), dtype="uint8")

        for idx in range(len(eval_files)):

            in_image = cv2.imread(eval_files[idx], 0)


            input_image[0,:,:,0] = in_image


            actual_classes = eval_datay.iloc[[idx]].values.reshape((1, 28))



            output_classes, input_image, one_hot_prediction, f1_score_summary = self.sess.run(
                [self.logit_q, self.X, self.one_hot_prediction, summary_merged],

                feed_dict={self.p: actual_classes, self.X: input_image,
                           self.is_training: False})

            summary_writer.add_summary(f1_score_summary, iter_num)
            # feed_dict={self.Y_: actual_classes,
            # self.is_training: False})

            # np.clip
            # Given an interval, values outside the interval are clipped to
            # the interval edges.
            # For example, if an interval of [0, 1] is specified,
            # values smaller than 0 become 0, and values larger than 1 become 1.

            one_hot_prediction = one_hot_prediction.astype('uint8')

            groundtruth_classes = actual_classes.astype('uint8')

            # calculate f1_score
            groundtruth_classes = np.ndarray.transpose(groundtruth_classes)
            one_hot_prediction = np.ndarray.transpose(one_hot_prediction)

            f1_score = sklearn.metrics.f1_score(groundtruth_classes, one_hot_prediction, labels=None, average='macro')

            print('img ', str(idx + 1), ' f1_score:', str(f1_score))

            outstreval = outstreval + ' img ' + str(idx + 1) + ' f1_score: ' + str(
                f1_score) + '\n'

            f1_score_sum += f1_score


        #print('evaluate CNN')
        #imgplot = plt.imshow(input_image[0,:,:,0])
        #plt.show()

        avg_f1_score = f1_score_sum / len(eval_files)

        print('--- Test ---- Average f1_score %.2f ---- ' % (avg_f1_score))

        # add average f1_score to tensorboard

        outstreval = outstreval + '--- Test ---- Average f1_score ' + str(
            avg_f1_score) + ' ---\n'

        filename = 'outstreval' + str(iter_num) + '.txt'
        file = open(filename, 'w')
        file.write(outstreval)
        file.close()

    def train(self, data, datay, eval_files, eval_datay, batch_size, ckpt_dir,
              epoch, lr, sample_dir, eval_every_epoch=1):


        # assert data range is between 0 and 1
        numBatch = int((len(data) + 1) / batch_size)


        print('numBatch', (len(data)) / batch_size, len(data), batch_size)

        # if pretrained model exists - load pretrained model
        # else train new model
        # ckpt_dir=checkpoint
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch


            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Did not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_f1_score = tf.summary.scalar('eva_f1_score', self.eva_f1_score)

        outstrtr = "Save output here.\n"

        outstrtr = outstrtr + "[*] Start training, with start epoch " + str(start_epoch) + " start iter " + str(
            iter_num) + "\n"

        print("[*] Start training, with start epoch %d start iter %d : " % (
            start_epoch, iter_num))
        print('[*][*][*] Go to line 328 and 329 and set cardinality and blocks to their original values')
        print('[*][*][*] Uncomment lines 482 and 483')

        start_time = time.time()


        self.evaluate(iter_num,  eval_files, eval_datay, sample_dir=sample_dir,
                      summary_merged=summary_f1_score,
                      summary_writer=writer)




        for epoch in range(start_epoch, epoch):


            for batch_id in range(start_step, numBatch):

                batch_images = data[batch_id * batch_size:(
                                                                  batch_id + 1) * batch_size,
                               :, :, :]




                batch_images = batch_images.astype(
                    np.int8)


                batch_imagesy = datay[batch_id * batch_size:(
                                                                    batch_id + 1) * batch_size,
                                :]
                batch_imagesy = batch_imagesy.astype(
                    np.int8)




                _, loss, summary = self.sess.run(
                    [self.train_op, self.loss, merged],
                    feed_dict={self.p: batch_imagesy, self.X: batch_images,
                               self.lr: lr[epoch],
                               self.is_training: True})

                # add f1_score as well
                outstrtr = outstrtr + "Epoch: " + str(epoch + 1) + " [" + str(
                    batch_id + 1) + "/" + str(numBatch) + "] " + "time: " + str(
                    time.time() - start_time) + "loss: " + str(loss) + "\n"

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"

                      % (epoch + 1, batch_id + 1, numBatch,
                         time.time() - start_time, loss))

                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_files, eval_datay,
                              sample_dir=sample_dir,
                              summary_merged=summary_f1_score,
                              summary_writer=writer)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)



        print("[*] Finish training.")
        filename = 'outstrtr' + str(epoch + 1) + '.txt'
        file = open(filename, 'w')
        file.write(outstrtr)
        file.close()

    def save(self, iter_num, ckpt_dir, model_name='CNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):

        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0


    def test(self,test_files,  test_datay, ckpt_dir, save_dir):

        """Test CNN"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data x!'
        assert len(test_datay) != 0, 'No testing data y!'

        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        f1_score_sum = 0
        print("[*] " + " start testing...")

        input_image = np.zeros((1, 512, 512, 1), dtype="uint8")


        for idx in range(len(test_files)):

            # open x
            in_image = cv2.imread(test_files[idx], 0)

            input_image[0,:,:,0] = in_image


            actual_classes = test_datay.iloc[[idx]].values.reshape((1, 28))


            output_classes, one_hot_prediction = self.sess.run([self.logit_q, self.one_hot_prediction],
                                                               feed_dict={
                                                                   self.p: actual_classes,
                                                                   self.X: input_image,
                                                                   self.is_training: False})

            groundtruth_classes = actual_classes.astype('uint8')
            one_hot_prediction = one_hot_prediction.astype('uint8')


            # calculate f1_score
            groundtruth_classes = np.ndarray.transpose(groundtruth_classes)
            one_hot_prediction = np.ndarray.transpose(one_hot_prediction)

            print( groundtruth_classes, one_hot_prediction)
            f1_score = sklearn.metrics.f1_score(groundtruth_classes, one_hot_prediction, labels=None, average='macro')
            print(np.argwhere(output_classes))
            print("img%d f1_score: %.2f" % (idx, f1_score))
            f1_score_sum += f1_score
        #print('test CNN')
        #imgplot = plt.imshow(input_image[0,:,:,0])
        #plt.show()

        avg_f1_score = f1_score_sum / len(test_files)

        print("--- Average f1_score %.2f ---" % avg_f1_score)


def make_image_row(image):
    image = np.reshape(np.array(image, dtype="uint8"),
                       (image.size[0], image.size[1], 1))

    return image


class train_data():
    def __init__(self, filepath):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)

        # np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath):
    return train_data(filepath=filepath)


# denoiser_train
def denoiser_train(denoiser, lr):


    with load_data(filepath='../Human Protein Atlas Image Classification Dataset/protein_image_patches_pat_size_300_bat_size_8_101_200.npy') as data:
        with load_data(filepath='../Human Protein Atlas Image Classification Dataset/protein_image_classes_pat_size_300_bat_size_8_101_200.npy') as datay:


            train_labels = pd.read_csv(
                '../input/human-protein-atlas-image-classification-dataset/kaggle_protein_classes_augmented_one_hot.csv')
            train_labels = train_labels.sort_values(by=['Id']).reset_index()
            seed = 100
            train_labels = shuffle(train_labels, random_state=seed).reset_index(drop=True)



            eval_datay = train_labels.loc[:10]

            eval_datay = eval_datay.iloc[:, 4:]

            # print(train_labels.head())

            # get image id
            eval_data_im_id = train_labels.loc[:10, "Id"]

            # get each image channel as a greyscale image (second argument 0 in imread)

            eval_files = []
            for eval_im_id in eval_data_im_id:
                eval_files.append(
                    glob('../input/human-protein-atlas-image-classification/train/{}_green.png'.format(eval_im_id)))
            eval_files = [''.join(x) for x in eval_files]


            # if there is a small memory, please comment this line and uncomment the line99 in model.py
            #data = data.astype(np.int8) / 255.0  # normalize the data to 0-1


            # number of images in batch
            batch_size = 8
            # models are saved here
            ckpt_dir = './checkpoint'
            epoch = 2
            # samples are saved here
            sample_dir = './sample'

            lr = 0.1
            lr = lr * np.ones([epoch])

            lr[30:] = lr[0] / 10.0
            lr[60:] = lr[0] / 100.0
            lr[90:] = lr[0] / 1000.0

            train_set = 'train'


            #eval_data = load_images(eval_files)  # list of array of different size, 4-D, pixel value range is 0-255

            denoiser.train(data, datay, eval_files, eval_datay,
                           batch_size=batch_size, ckpt_dir=ckpt_dir,
                           epoch=epoch, lr=lr,
                           sample_dir=sample_dir)



# denoiser_test
# dataset for testing

# this is ok
def denoiser_test(denoiser):
    # models are saved here
    ckpt_dir = './checkpoint'
    # test sample are saved here
    test_dir = './test'

    test_set = 'test_set'

    # load test images
    train_set = 'train'

    train_labels = pd.read_csv('../input/human-protein-atlas-image-classification-dataset/kaggle_protein_classes_augmented_one_hot.csv')

    train_labels = train_labels.sort_values(by=['Id']).reset_index()
    seed = 100
    train_labels = shuffle(train_labels, random_state=seed).reset_index(drop=True)

    test_filesy = train_labels.loc[11:20]
    test_datay = test_filesy.iloc[:, 4:]

    #test_datay = test_filesy

    # get image id
    test_data_im_id = train_labels.loc[11:20, "Id"]

    # get each image channel as a greyscale image (second argument 0 in imread)

    test_files = []
    for eval_im_id in test_data_im_id:
        test_files.append(glob('../input/human-protein-atlas-image-classification/train/{}_green.png'.format(eval_im_id)))
    # data_files.sort()
    test_files = [''.join(x) for x in test_files]

    denoiser.test(test_files,  test_datay, ckpt_dir=ckpt_dir, save_dir=test_dir)


######## Start training here

# initial learning rate for adam
lr = 0.1

# number of epochs

epoch = 2
# use_gpu=0: use tensorflow cpu
# use_gpu=1: use tensorflow gpu
use_gpu = 0

# train or test
phase = 'train'
ckpt_dir = './checkpoint'
sample_dir = './sample'
test_dir = './test'

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

lr = lr * np.ones([epoch])

lr[30:] = lr[0] / 10.0
lr[60:] = lr[0] / 100.0
lr[90:] = lr[0] / 1000.0

tf.reset_default_graph()
if use_gpu:
    # added to control the gpu memory
    print("GPU\n")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = denoiser(sess)
        if phase == 'train':
            denoiser_train(model, lr=lr)
        elif phase == 'test':
            denoiser_test(model)
        else:
            print('[!]Unknown phase')
            exit(0)
else:
    print("CPU\n")
    with tf.Session() as sess:
        model = denoiser(sess)
        if phase == 'train':
            denoiser_train(model, lr=lr)
        elif phase == 'test':
            denoiser_test(model)
        else:
            print('[!]Unknown phase')
            exit(0)




