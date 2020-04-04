import tensorflow as tf
#=================三种计算图:静态图、动态图、Autograph==============
hello = tf.constant('hello ',dtype=tf.string)
world = tf.constant('world',dtype=tf.string)
val = tf.strings.join([hello,world])
# tf.print(val)
#=================Autograph方式==================================
@tf.function
def func(a,b):
    z = tf.strings.join([a,b])
    tf.print(z)
    return z
print(func(hello,world))
