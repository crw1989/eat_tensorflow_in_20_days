# ===========================tensorflow 张量
import tensorflow as tf
import numpy_aa as np

value = tf.constant(12, dtype=tf.int32)
value = tf.constant([1, 2, 3], dtype=tf.int32)
value = tf.range(1, 10, 2)
value = tf.zeros(shape=value.shape)
value = tf.zeros_like(value, dtype=tf.int32)
value = tf.fill([3, 2], value=5)
value = tf.random.uniform(shape=[2, 3], minval=-10, maxval=10, dtype=tf.int32)
value = tf.random.normal([3, 2], dtype=tf.float32)
# tf.print(value)
# =========================切片索引访问=================
value = tf.random.uniform([5, 5], minval=-10, maxval=10)
# tf.print(value)
# tf.print(value[-1])
# tf.print(value[:,1])
# tf.print(value[:3,:3])
# tf.print(value[::2,::2])
# =========================修改部分元素值
value = tf.Variable([1, 2, 3], dtype=tf.int32)
value[:2].assign([4, 4])
# tf.print(value)
# ==========================支持省略号检索
value = tf.random.uniform(shape=[5, 5, 5, 3], minval=-10, maxval=10)
# tf.print(value)
# tf.print('=====================')
# tf.print(value[...,:1].shape)
# tf.print(value[...,0].shape)
# ===========================tf.where
value = tf.random.uniform([5, 5], minval=-10, maxval=10, dtype=tf.int32)
# tf.print(value)
value = tf.where(value < 0, 0, value)
# tf.print(value)
# ===========================tensorflow维度变换
value = tf.random.uniform([1, 6, 5, 5], minval=-10, maxval=10, dtype=tf.int32)
# tf.print(value.shape)
# tf.print(tf.reshape(value,[5,6,5]).shape)
# tf.print(tf.squeeze(value).shape)
# tf.print(tf.expand_dims(value,-2).shape)
# ===========================合并与分割，concat不增加新的维度,stack会
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
# tf.print(tf.concat([a, b], axis=0))
# tf.print(tf.concat([a, b], axis=1))
# tf.print(tf.stack([a,b],axis=0))
# tf.print(tf.stack([a,b],axis=1))
