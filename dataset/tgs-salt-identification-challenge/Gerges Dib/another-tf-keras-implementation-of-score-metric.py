import numpy as np
import keras.backend as K
import tensorflow as tf


def iou_precision(y_true, y_pred):
    """
    Computes the mean precision at different iou threshold levels.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = tf.to_int32(y_true)
    y_pred = tf.to_int32(tf.round(y_pred))

    n_batch = tf.shape(y_true)[0]

    y_true = tf.reshape(y_true, shape=[n_batch , -1])
    y_pred = tf.reshape(y_pred, shape=[n_batch, -1])

    intersection = K.sum(tf.bitwise.bitwise_and(y_true, y_pred), -1)
    union = K.sum(tf.bitwise.bitwise_or(y_true, y_pred), -1)
    #iou = tf.where(union == 0, tf.ones(n_batch), tf.to_float(intersection/union))
    SMOOTH = tf.constant(1e-6)
    iou = tf.add(tf.to_float(intersection), SMOOTH)/tf.add(tf.to_float(union), SMOOTH)

    precision = tf.zeros(n_batch)
    thresholds = np.arange(0.5, 1.0, 0.05)
    for thresh in thresholds:
        precision = precision + tf.to_float(iou > thresh)
    precision = precision/len(thresholds)

    return K.mean(precision)


if __name__ == '__main__':
    y_true = tf.constant([[[0, 0, 0], [0, 1, 1], [1, 0, 1]]])
    y_pred = tf.constant([[0.2, 0.4, 0.3], [0.7, 0.6, 0.7], [0.9, 0.2, 1]])
    mean_precision = iou_precision(y_true, y_pred)
    sess = tf.Session()
    assert np.isclose(sess.run(mean_precision), 0.6)
    print('Finished...')