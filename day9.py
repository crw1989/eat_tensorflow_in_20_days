import tensorflow as tf
import tensorflow.keras as keras

# =====================中阶API=================


n = 800
x = tf.random.uniform(shape=(n, 2), dtype=tf.float32, minval=-10, maxval=10)
w0 = tf.constant([[2], [2]], dtype=tf.float32)
b0 = tf.constant(1, dtype=tf.float32)
y = x @ w0 + b0
ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=1024).batch(100).prefetch(
    tf.data.experimental.AUTOTUNE)
optimizer = tf.optimizers.Adam()

dense = keras.layers.Dense(1)
dense.build(input_shape=(2,))


def train(epoch):
    for i in range(epoch):
        for x_batch, y_batch in ds:
            with tf.GradientTape() as tape:
                pred = dense(x_batch)
                loss = tf.reduce_mean(tf.losses.mean_squared_error(pred, y_batch))
            gradients = tape.gradient(loss, dense.variables)
            optimizer.apply_gradients(zip(gradients, dense.variables))
        tf.print("loss:", loss, 'epoch:', i)
train(5000)
