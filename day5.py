#===============tensorflow数据类型与numpy数据类型相对应:int32、int64、float32、float64、bool============================
import numpy_aa as np
import tensorflow as tf

int32 = tf.constant(1)
int64 = tf.constant(1, dtype=tf.int64)
float32 = tf.constant(1.1)
float64 = tf.constant(1.1, dtype=tf.double)
string = tf.constant('hello,world', dtype=tf.string)
bool = tf.constant(True)

assert tf.int32 == np.int32
assert tf.int64 == np.int64
assert tf.float32 == np.float32
assert tf.float64 == np.double
assert tf.bool == np.bool

#=================0、1、2、3……张量==========================
constant = tf.constant(1,dtype=tf.int32)
print(tf.rank(constant))

vector = tf.constant([1,2,3],dtype=tf.int32)
print(tf.rank(vector))

matrix = tf.constant([[1,2,3]],dtype=tf.int32)
print(tf.rank(matrix))
print(matrix.shape)
#=================通过tf.cast完成张量类型转换================
print(tf.cast(constant,dtype=tf.int64))
#==================常量与变量===============================