import tensorflow as tf

def get_metric(true_labels, predicted_labels):
    """
    Tensorflow implementation of scoring metric
    """
    with tf.name_scope('custom_metric'):      
        mask1 = tf.cast(true_labels, dtype=tf.bool, name='bool_mask1')
        mask2 = tf.cast(predicted_labels, dtype=tf.bool, name='bool_mask2')
        intersection = tf.logical_and(mask1, mask2, name='intersection')
        union = tf.logical_or(mask1, mask2, name='intersection')
        intersection_len = tf.count_nonzero(intersection, axis=[1, 2])
        union_len = tf.count_nonzero(union, axis=[1, 2])
        iou = tf.divide(intersection_len, union_len)
        iou = tf.where(tf.is_nan(iou), tf.ones_like(iou), iou)
        iou = tf.floor(iou*20)
        return iou/20.0