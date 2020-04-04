import tensorflow as tf
import numpy as np

# ===================自动微分机制=========================
x = tf.Variable(0, name='x', dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)
with tf.GradientTape() as tape:
    y = a * tf.pow(x, 2) + b * x + c

dy_dx = tape.gradient(y, [x, a, b, c])
# tf.print(dy_dx)
# ==================使用自动微分求最小值==================
x = tf.Variable(0, name='x', dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)
optimizer = tf.optimizers.Adam()
for _ in range(0):
    with tf.GradientTape() as tape:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)
    optimizer.apply_gradients([(dy_dx, x)])
    pass
tf.print('y=', y, 'x=', x)
# ========================================================
x = tf.Variable(0, dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)
optimizer = tf.optimizers.Adam()


@tf.function
def f():
    y = a * tf.pow(x, 2) + b * x + c
    return y


for _ in tf.range(0):
    optimizer.minimize(f, [x])
    pass
tf.print(x)
# ===================================================
x = tf.Variable(0, dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)
optimizer = tf.optimizers.Adam()


@tf.function
def f():
    for _ in tf.range(0):
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients([(dy_dx, x)])
        pass
    return x


tf.print(f())
# =================================================
x = tf.Variable(0, dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)
optimizer = tf.optimizers.Adam()


@tf.function
def func():
    y = a * tf.pow(x, 2) + b * x + c
    return y

@tf.function
def train(epoch):
    for _ in tf.range(epoch):
        optimizer.minimize(func, [x])
        pass
    return x


tf.print(train(10000))
