# =================================autograpy使用规范
# 1.被tf.function修饰的函数内部调用的最好是tf的函数，因为python函数不会嵌入到计算图中，每次静态图前向传播时不会运行python基本函数
# 2.避免在tf.function修饰的函数中定义tf.variable，eager模式执行时每次均会重新执行variable，而静态图只会运行一次
# 3.被tf.function不能修改函数外部定义的Python字典、列表等数据结构，python中的字典、列表等无法嵌入到计算图中，仅仅在创建计算图时完成读取
# =================================

import tensorflow as tf
import numpy as np

arr = np.array([])


# =============np.random第次调用时生成的均是相同的随机数
@tf.function
def np_random():
    arr = np.random.uniform(low=-10, high=10, size=[5, 5])
    return arr


tf.print(np_random())
tf.print(np_random())


# =============tf.random每次调用时生成的均是新的随机数
@tf.function
def tf_random():
    arr = tf.random.uniform(minval=-10, maxval=10, shape=[5, 5])
    return arr


tf.print(tf_random())
tf.print(tf_random())
