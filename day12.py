import tensorflow as tf
import numpy as np

# ================== 标量运算
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
# tf.print(a+b)
# tf.print(a * b)
# tf.print(b/a)
# tf.print(a-b)
# tf.print(a**2)

# flag = tf.cast(a[...,1] == 2 , dtype=tf.bool)
# tf.print(a[flag])

# flag = tf.cast((a[..., 0] >= 1) & (a[..., 1] % 2 == 0), dtype=tf.bool)
# tf.print(a[flag])

# tf.print((a == 5).dtype)

# tf.print(tf.maximum(a,b)) # 两个相同维度的张量对应位置大小进行比较
# tf.print(tf.minimum(a,b))
# ================== 向量运算，函数一般以reduce开头
a = tf.reshape(tf.range(100), shape=[10, 10])
# tf.print(a)
# tf.print(tf.reduce_sum(a,axis=-1))
# tf.print(tf.reduce_mean(a,axis=-1))
# tf.print(tf.reduce_all(a % 2 == 0,axis=-1))
# tf.print(tf.reduce_any(a % 2 == 0))

a = tf.random.uniform(shape=[5, 1], minval=-10, maxval=10)  # argmax代表最大索引,arg代表手纸架索引
# tf.print(a)
# tf.print(tf.argmax(a))
idx = tf.squeeze(tf.argsort(a, axis=0))
# tf.print(idx)
# tf.print(tf.gather(a,idx,axis=0))
# ================== 矩阵运算
a = tf.random.uniform(shape=[5, 5], minval=-10, maxval=10, dtype=tf.float32)
tf.print(a)
tf.print('================================')
# tf.print(tf.sqrt(tf.reduce_sum(a**2)))
# tf.print(tf.linalg.norm(a))
# tf.print(tf.linalg.normalize(a, axis=-1))
