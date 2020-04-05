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

ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=1024).batch(64).prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.optimizers.Adam()
def train(epoch):
    for i in range(epoch):
        for x_batch, y_batch in ds:
            with tf.GradientTape() as tape:
                y_pred = model(x_batch)
                loss = tf.reduce_mean(tf.losses.mean_squared_error(y_batch, y_pred))
            gradient = tape.gradient(loss,model.variables)
            optimizer.apply_gradients(zip(gradient,model.variables))
        tf.print('epoch:',i,',loss:',loss)
train(1000)