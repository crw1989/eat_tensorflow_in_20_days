import tensorflow as tf
#=============================Autograph机制原理
#1. 创建计算图，根据将代码中的tensor加入到计算图中。创建计算图的过程中执行print('tracing')
#    a.当我们再次用相同的输入参数类型调用这个被@tf.function装饰的函数时，不会输出tracing内容
#2. 运行计算图
import tensorflow as tf
import numpy as np

@tf.function
def myadd(a,b):
    for i in tf.range(3):
        tf.print(i)
    c = a+b
    print("tracing")
    return c

#================如果传入的参数是tensor不会重新创建计算图
# myadd(tf.constant("hello"),tf.constant("world")) tracing 0 1 2
# myadd(tf.constant("hello"),tf.constant("world")) 0，1，2

#================如果传入的参数不是tensor可能会重新创建计算图（当传入参数不一样时）
myadd('hello','world')
myadd('hello','morning')

'''
g = tf.Graph()
with g.as_default():
    a = tf.placeholder(shape=[],dtype=tf.string)
    b = tf.placeholder(shape=[],dtype=tf.string)
    cond = lambda i: i<tf.constant(3)
    def body(i):
        tf.print(i)
        return(i+1)
    loop = tf.while_loop(cond,body,loop_vars=[0])
    loop
    with tf.control_dependencies(loop):
        c = tf.strings.join([a,b])
    print("tracing")

with tf.Session(graph=g) as sess:
    sess.run(c,feed_dict={a:tf.constant("hello"),b:tf.constant("world")})

'''