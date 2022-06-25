"""A Baseline model
Source: https://www.tensorflow.org/tutorials/structured_data/time_series
"""
import tensorflow as tf


# pylint: disable=too-few-public-methods
class Baseline(tf.keras.Model):
    def __init__(self, label_index: int = None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
