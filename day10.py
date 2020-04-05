import tensorflow as tf
from tensorflow.keras import *

# =====================高阶API
n = 800
x = tf.random.uniform(shape=(n, 2), dtype=tf.float32, minval=-10, maxval=10)
w0 = tf.constant([[1], [1]], dtype=tf.float32)
b0 = tf.constant(2, dtype=tf.float32)
y = x @ w0 + b0
input = Input(shape=(2,))
out = layers.Dense(1)(input)
model = Model(input, out)

ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=1024).batch(64).prefetch(
    tf.data.experimental.AUTOTUNE)

optimizer = tf.optimizers.Adam(0.01)
train_loss = tf.metrics.MeanSquaredError('train_loss')


@tf.function
def loss(pred, label):
    return tf.reduce_mean(tf.losses.mean_squared_error(pred, label))


@tf.function
def train(epoch):
    for i in tf.range(epoch):
        for x_batch, y_batch in ds:
            with tf.GradientTape() as tape:
                y_pred = model(x_batch)
                loss_val = loss(y_pred, y_batch)
            train_loss.update_state(loss_val,y_batch)
            gradient = tape.gradient(loss_val, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        tf.print('epoch:', i, 'loss:', train_loss.result())

train(10000)
