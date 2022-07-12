""" Utility script for Kaggle notebooks using tf.keras  """
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

import os
import random
import numpy as np
import matplotlib.pyplot as plt



##############################################################################################################################
### Contents:
# OneCycle LR Scheduler
# Focal loss functions
# Generalized Mean Pooling Layers
# Batch CutMix + QuarterMix augmentation



##############################################################################################################################
### General purpose:


def is_interactive():
    ''' Return True if inside a notebook/kernel in Edit mode, or False if committed '''
    try:
        from IPython import get_ipython
        return 'runtime' in get_ipython().config.IPKernelApp.connection_file
    except:
        return False


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def setup_tpu(tpu_id=None):
    """ resolve a tpu cluster """
    ## Detect hardware, return appropriate distribution strategy
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_id)
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return strategy




##############################################################################################################################
### LR Scheduling



class OneCycleScheduler(Callback):
    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25., final_div_factor=1e4):
        super().__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / final_div_factor
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps

        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)], 
                       [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]

        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1

        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass # ignore

    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]

    def mom_schedule(self):
        return self.phases[self.phase][1]

    def plot(self):
        plt.figure(figsize=(15, 6))
        ax = plt.subplot(1, 3, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate')

        ax = plt.subplot(1, 3, 2)
        ax.plot(self.lrs)
        ax.set_yscale('log')
        ax.set_title('Learning Rate - Log')

        ax = plt.subplot(1, 3, 3)
        ax.plot(self.moms)
        ax.set_title('Momentum')


class CosineAnnealer:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0
        
    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos



##############################################################################################################################
### Loss



def binary_focal_loss(gamma=2., pos_weight=1, label_smoothing=0.05):
    """ binary focal loss with label_smoothing """
    def focal_loss(labels, p):
        """ bfl clojure """
        labels = tf.dtypes.cast(labels, dtype=p.dtype)
        if label_smoothing is not None:
            labels = (1 - label_smoothing) * labels + label_smoothing * 0.5

        # Predicted probabilities for the negative class
        q = 1 - p

        # For numerical stability (so we don't inadvertently take the log of 0)
        p = tf.math.maximum(p, K.epsilon())
        q = tf.math.maximum(q, K.epsilon())

        # Loss for the positive examples
        pos_loss = -(q ** gamma) * tf.math.log(p) * pos_weight

        # Loss for the negative examples
        neg_loss = -(p ** gamma) * tf.math.log(q)

        # Combine loss terms
        loss = labels * pos_loss + (1 - labels) * neg_loss

        return loss

    return focal_loss


def categorical_focal_loss(num_classes, gamma=2., alpha=.25, smooth_alpha=0.05):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        if smooth_alpha > 0:
            y_true = y_true * (1 - smooth_alpha) + smooth_alpha / num_classes

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


##############################################################################################################################
### Layers



class GeneralizedMeanPooling1D(tf.keras.layers.Layer):
    def __init__(self, p=3, epsilon=1e-6, shape=1, name='', **kwargs):
        super().__init__(name, **kwargs)
        self.init_p = p
        self.epsilon = epsilon
        self.shape = shape

    def build(self, input_shape):
        if isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError(f'`GeM` pooling layer only allow 1 input with 3 dimensions (b, s, c): {input_shape}')
        self.build_shape = input_shape
        self.p = self.add_weight(
              name='p',
              shape=[self.shape,],
              initializer=tf.keras.initializers.Constant(value=self.init_p),
              regularizer=None,
              trainable=True,
              dtype=tf.float32
              )
        self.built=True

    def call(self, inputs):
        input_shape = inputs.get_shape()
        if isinstance(inputs, list) or len(input_shape) != 3:
            raise ValueError(f'`GeM` pooling layer only allow 1 input with 3 dimensions (b, s, c): {input_shape}')
        return (tf.reduce_mean(tf.abs(inputs**self.p), axis=1, keepdims=False) + self.epsilon)**(1.0/self.p)


class GeneralizedMeanPooling2D(tf.keras.layers.Layer):
    def __init__(self, p=3, epsilon=1e-6, shape=1, **kwargs):
        super().__init__(name, **kwargs)
        self.init_p = p
        self.epsilon = epsilon
        self.shape = shape

    def build(self, input_shape):
        if isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError('`GeM` pooling layer only allow 1 input with 4 dimensions (b, h, w, c)')
        self.build_shape = input_shape
        self.p = self.add_weight(
              name='p',
              shape=[self.shape,],
              initializer=tf.keras.initializers.Constant(value=self.init_p),
              regularizer=None,
              trainable=True,
              dtype=tf.float32
              )
        self.built=True

    def call(self, inputs):
        input_shape = inputs.get_shape()
        if isinstance(inputs, list) or len(input_shape) != 4:
            raise ValueError('`GeM` pooling layer only allow 1 input with 4 dimensions (b, h, w, c)')
        return (tf.reduce_mean(tf.abs(inputs**self.p), axis=[1,2], keepdims=False) + self.epsilon)**(1.0/self.p)




##############################################################################################################################
### CutMix / QuarterMix augmentation

def batch_cutmix(images, labels, PROBABILITY=1.0):
    ''' adapted from kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu '''
    batch_size, h, w, c = images.shape

    # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
    # This is a tensor containing 0 or 1 -- 0: no cutmix.
    # shape = [batch_size]
    do_cutmix = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)

    # Choose random images in the batch for cutmix
    # shape = [batch_size]
    new_image_indices = tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int32)

    # Choose random location in the original image to put the new images
    # shape = [batch_size]
    new_x = tf.random.uniform([batch_size], 0, w, dtype=tf.int32)
    new_y = tf.random.uniform([batch_size], 0, h, dtype=tf.int32)

    # Random width for new images, shape = [batch_size]
    b = tf.random.uniform([batch_size], 0, 1) # this is beta dist with alpha=1.0
    new_height = tf.cast(h * tf.math.sqrt(1-b), tf.int32) * do_cutmix
    new_width  = tf.cast(w * tf.math.sqrt(1-b), tf.int32) * do_cutmix

    # shape = [batch_size]
    ya = tf.math.maximum(0, new_y - new_height // 2)
    yb = tf.math.minimum(h, new_y + new_height // 2)
    xa = tf.math.maximum(0, new_x - new_width // 2)
    xb = tf.math.minimum(w, new_x + new_width // 2)

    # shape = [batch_size, h]
    target = tf.broadcast_to(tf.range(h), shape=(batch_size, h))
    mask_y = tf.math.logical_and(ya[:, tf.newaxis] <= target, target <= yb[:, tf.newaxis])

    # shape = [batch_size, w]
    target = tf.broadcast_to(tf.range(w), shape=(batch_size, w))
    mask_x = tf.math.logical_and(xa[:, tf.newaxis] <= target, target <= xb[:, tf.newaxis])    

    # shape = [batch_size, h, w]
    mask = tf.cast(tf.math.logical_and(mask_y[:, :, tf.newaxis], mask_x[:, tf.newaxis, :]), tf.float32)

    # All components are of shape [batch_size, h, w, 3]
    # also flips one of the images to avoid repeating pixels
    fliped_images = tf.image.flip_left_right(images)
    new_images = (tf.gather(images, new_image_indices) * tf.broadcast_to(mask[:, :, :, tf.newaxis],
                                                                        [batch_size, h, w, 3]) + 
                                         fliped_images * tf.broadcast_to(1 - mask[:, :, :, tf.newaxis],
                                                                        [batch_size, h, w, 3]))

    # Average binary labels
    a = tf.math.reduce_mean(mask, axis=(1,2))
    new_labels = (1-a) * labels + a * tf.gather(labels, new_image_indices)

    return new_images, new_labels



def batch_quartermix(images, labels, PROBABILITY=1.0):
    ''' mix images with a quarter or half (horizontal split) of another image '''
    batch_size, h, w, c = images.shape

    # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
    # This is a tensor containing 0 or 1 -- 0: no cutmix.
    # shape = [batch_size]
    do_cutmix = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)

    # Choose random images in the batch for cutmix
    # shape = [batch_size]
#         new_image_indices = tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int32)
    # (choose neighbour from the right to avoid network traffic)
    new_image_indices = np.arange(batch_size).tolist()
    new_image_indices = tf.cast([new_image_indices.pop()] + new_image_indices, dtype=tf.int32)

    # Choose random location in the original image to put the new images
    # shape = [batch_size]
    new_x = tf.random.uniform([batch_size], 0, 2, dtype=tf.int32) * (w)
    new_y = tf.random.uniform([batch_size], 0, 3, dtype=tf.int32) * (h // 2)

#     # Random width for new images, shape = [batch_size]
    new_width = w * do_cutmix
    new_height = h * do_cutmix

    # shape = [batch_size]
    ya = tf.math.maximum(0, new_y - new_height // 2)
    yb = tf.math.minimum(h, new_y + new_height // 2)
    xa = tf.math.maximum(0, new_x - new_width // 2)
    xb = tf.math.minimum(w, new_x + new_width // 2)

    # shape = [batch_size, h]
    target = tf.broadcast_to(tf.range(h), shape=(batch_size, h))
    mask_y = tf.math.logical_and(ya[:, tf.newaxis] <= target, target <= yb[:, tf.newaxis])

    # shape = [batch_size, w]
    target = tf.broadcast_to(tf.range(w), shape=(batch_size, w))
    mask_x = tf.math.logical_and(xa[:, tf.newaxis] <= target, target <= xb[:, tf.newaxis])    

    # shape = [batch_size, h, w]
    mask = tf.cast(tf.math.logical_and(mask_y[:, :, tf.newaxis], mask_x[:, tf.newaxis, :]), tf.float32)

    # All components are of shape [batch_size, h, w, 3]
    # also flips one of the images to avoid repeating pixels
    fliped_images = tf.image.flip_left_right(images)
    new_images = (tf.gather(images, new_image_indices) * tf.broadcast_to(mask[:, :, :, tf.newaxis],
                                                                        [batch_size, h, w, 3]) + 
                                         fliped_images * tf.broadcast_to(1 - mask[:, :, :, tf.newaxis],
                                                                        [batch_size, h, w, 3]))

    # Average binary labels
    a = tf.math.reduce_mean(mask, axis=(1,2))
    new_labels = (1-a) * labels + a * tf.gather(labels, new_image_indices)    

    return new_images, new_labels



if __name__ == "__main__":
    seed_everything(0)
    is_interactive()
    OneCycleScheduler(lr_max=1e-2, steps=10000).plot(); plt.show()
    
    # smoke test batch_cutmix
    batch, labels = tf.stack([tf.zeros([32, 32, 3]), tf.ones([32, 32, 3])]), tf.cast([0, 1], tf.float32)
    batch, labels = batch_cutmix(batch, labels, 1.0)
    print(labels)
    assert np.array_equal( labels,  batch.numpy().mean(axis=(1,2,3)) )
    plt.imshow(batch[0].numpy().astype(int)*255); plt.show()
    plt.imshow(batch[1].numpy().astype(int)*255); plt.show()
    # quartermix
    batch, labels = tf.stack([tf.zeros([32, 32, 3]), tf.ones([32, 32, 3])]), tf.cast([0, 1], tf.float32)
    batch, labels = batch_quartermix(batch, labels, 1.0)
    print(labels)
    assert np.array_equal( labels,  batch.numpy().mean(axis=(1,2,3)) )
    plt.imshow(batch[0].numpy().astype(int)*255); plt.show()
    plt.imshow(batch[1].numpy().astype(int)*255); plt.show()