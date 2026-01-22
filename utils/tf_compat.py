"""
TensorFlow 2 compatibility: use compat.v1 and disable eager/v2 behavior
so existing TF1-style code (Session, placeholder, etc.) runs on Python 3.8+.

Usage:  from utils.tf_compat import tf
"""
import tensorflow as _tf

_tf.compat.v1.disable_v2_behavior()
tf = _tf.compat.v1


def dense_to_sparse(dense, eos_token=-1):
    """Replace tf.contrib.layers.dense_to_sparse (removed in TF2).
    Converts dense tensor with eos_token padding to tf.SparseTensor.
    """
    indices = tf.where(tf.not_equal(dense, eos_token))
    values = tf.cast(tf.gather_nd(dense, indices), tf.int32)
    dense_shape = tf.cast(tf.shape(dense), tf.int64)
    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
