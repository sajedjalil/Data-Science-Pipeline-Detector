import numpy as np
import tensorflow as tf
from sklearn.metrics import label_ranking_average_precision_score

tf.enable_v2_behavior()


def _lwlrap_sklearn(truth, scores):
    """Reference implementation from https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8"""
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0, 
        scores[nonzero_weight_sample_indices, :], 
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


def _one_sample_positive_class_precisions(example):
    y_true, y_pred = example

    retrieved_classes = tf.argsort(y_pred, direction='DESCENDING')
    class_rankings = tf.argsort(retrieved_classes)
    retrieved_class_true = tf.gather(y_true, retrieved_classes)
    retrieved_cumulative_hits = tf.math.cumsum(tf.cast(retrieved_class_true, tf.float32))

    idx = tf.where(y_true)[:, 0]
    i = tf.boolean_mask(class_rankings, y_true)
    r = tf.gather(retrieved_cumulative_hits, i)
    c = 1 + tf.cast(i, tf.float32)
    precisions = r / c

    dense = tf.scatter_nd(idx[:, None], precisions, [y_pred.shape[0]])
    return dense


class LWLRAP(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='lwlrap'):
        super().__init__(name=name)

        self._precisions = self.add_weight(
            name='per_class_cumulative_precision',
            shape=[num_classes],
            initializer='zeros',
        )

        self._counts = self.add_weight(
            name='per_class_cumulative_count',
            shape=[num_classes],
            initializer='zeros',
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        precisions = tf.map_fn(
            fn=_one_sample_positive_class_precisions,
            elems=(y_true, y_pred),
            dtype=(tf.float32),
        )

        increments = tf.cast(precisions > 0, tf.float32)
        total_increments = tf.reduce_sum(increments, axis=0)
        total_precisions = tf.reduce_sum(precisions, axis=0)

        self._precisions.assign_add(total_precisions)
        self._counts.assign_add(total_increments)        

    def result(self):
        per_class_lwlrap = self._precisions / tf.maximum(self._counts, 1.0)
        per_class_weight = self._counts / tf.reduce_sum(self._counts)
        overall_lwlrap = tf.reduce_sum(per_class_lwlrap * per_class_weight)
        return overall_lwlrap

    def reset_states(self):
        self._precisions.assign(self._precisions * 0)
        self._counts.assign(self._counts * 0)


def test_match_sklearn():

    # Generate dummy data like https://colab.research.google.com/drive/1YwL7ewUE6vSLZRoSf3Oi9efyupLyfcdb
    num_samples = 100
    num_labels = 20
    scores = np.random.rand(num_samples, num_labels)
    truth = np.random.rand(num_samples, num_labels) > 0.5
    truth[0:1, :] = False

    # Compute expected overall LWLRAP.
    desired = _lwlrap_sklearn(truth, scores)

    # Accumulate LWLRAP per minibatch with tf.metrics.Metric.
    metric = LWLRAP(num_labels)
    batch_size = 1
    for i in range(0, num_samples, batch_size):
        y_true = truth[i:i+batch_size]
        y_pred = scores[i:i+batch_size]
        x = metric.update_state(y_true, y_pred)
    actual = metric.result()

    # Make sure both methods get similar averages.
    np.testing.assert_allclose(actual, desired)


if __name__ == '__main__':
    test_match_sklearn()
