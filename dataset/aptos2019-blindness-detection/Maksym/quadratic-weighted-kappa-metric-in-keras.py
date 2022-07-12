import numpy as np
from sklearn.metrics import cohen_kappa_score

import keras.backend as K
import tensorflow as tf


def kappa_keras(y_true, y_pred):

    y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
    y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype='int32')

    # Figure out normalized expected values
    min_rating = K.minimum(K.min(y_true), K.min(y_pred))
    max_rating = K.maximum(K.max(y_true), K.max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = K.map_fn(lambda y: y - min_rating, y_true, dtype='int32')
    y_pred = K.map_fn(lambda y: y - min_rating, y_pred, dtype='int32')

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = tf.confusion_matrix(y_true, y_pred,
                                num_classes=num_ratings)
    num_scored_items = K.shape(y_true)[0]

    weights = K.expand_dims(K.arange(num_ratings), axis=-1) - K.expand_dims(K.arange(num_ratings), axis=0)
    weights = K.cast(K.pow(weights, 2), dtype='float64')

    hist_true = tf.math.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[:num_ratings] / num_scored_items
    hist_pred = tf.math.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[:num_ratings] / num_scored_items
    expected = K.dot(K.expand_dims(hist_true, axis=-1), K.expand_dims(hist_pred, axis=0))

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    score = tf.where(K.any(K.not_equal(weights, 0)), 
                     K.sum(weights * observed) / K.sum(weights * expected), 
                     0)
    
    return 1. - score

if __name__ == '__main__':
    # Testing Keras implementation of QWK
    y_true = np.array([2, 0, 2, 2, 0, 1])
    y_pred = np.array([0, 0, 2, 2, 0, 2])
    
    # Calculating QWK score with scikit-learn
    skl_score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Keras implementation of QWK work with one hot encoding labels and predictions (also it works with softmax probabilities)
    # Converting arrays to one hot encoded representation
    shape = (y_true.shape[0], np.maximum(y_true.max(), y_pred.max()) + 1)

    y_true_ohe = np.zeros(shape)
    y_true_ohe[np.arange(shape[0]), y_true] = 1

    y_pred_ohe = np.zeros(shape)
    y_pred_ohe[np.arange(shape[0]), y_pred] = 1
    
    # Calculating QWK score with Keras
    with tf.Session() as sess:
        keras_score = kappa_keras(y_true_ohe, y_pred_ohe).eval()
    
    print('Scikit-learn score: {:.03}, Keras score: {:.03}'.format(skl_score, keras_score))
    