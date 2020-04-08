# =============使用tensorboard
import tensorflow as tf
logdir = './'
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()


a = tf.constant(1.0, dtype=tf.float32)
b = tf.constant(2, dtype=tf.float32)
c = tf.constant(11, dtype=tf.float32)
x = tf.Variable(1000, dtype=tf.float32)

optimizer = tf.optimizers.Nadam(0.1,clipnorm=10)
@tf.function
def train(epoch):
    for i in tf.range(epoch ** 2):
        with tf.GradientTape() as tape:
            loss = a * tf.math.pow(x, 2) + b * x + c
        gradient = tape.gradient(loss, x)
        optimizer.apply_gradients([(gradient, x)])
        tf.print(x)
        tf.summary.scalar('x',x,optimizer.iterations)
train(100000)
