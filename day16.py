#=========================使用tf.Module来封装tf.Variables，来结构化Autograph
import tensorflow as tf
import numpy as np


class AddModel(tf.Module):
    def __init__(self):
        with tf.name_scope('AddModel'):
            self.result = tf.Variable(0,dtype=tf.float32)
        pass

    @tf.function(
        input_signature=([tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)]))
    def add(self, a, b):
        self.result.assign(a + b + self.result)
        return self.result


model = AddModel()
model.add(tf.constant(1, dtype=tf.float32), tf.constant(2, dtype=tf.float32))
model.add(tf.constant(1, dtype=tf.float32), tf.constant(2, dtype=tf.float32))
tf.print(model.add(tf.constant(1, dtype=tf.float32), tf.constant(2, dtype=tf.float32)))
tf.print(model.variables)
tf.print(model.trainable_variables)
