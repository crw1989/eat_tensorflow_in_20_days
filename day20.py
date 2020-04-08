import tensorflow as tf
import tensorflow.keras as keras

# ====自定义损失函数
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
input = keras.Input(shape=[28, 28, 1])
x = keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation(tf.nn.swish)(x)
x = keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation(tf.nn.swish)(x)
x = keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation(tf.nn.swish)(x)
x = keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation(tf.nn.swish)(x)
x = keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation(tf.nn.swish)(x)
x = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation(tf.nn.swish)(x)
x = keras.layers.Dense(10, activation=tf.nn.softmax)(x)
model = keras.Model(input, x)

optimizer = tf.optimizers.Adam(0.1)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(
    batch_size=128).prefetch(tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(len(x_train)).batch(
    batch_size=32).prefetch(
    tf.data.experimental.AUTOTUNE)


@tf.function
def loss(pred, y_true):
    y_true = tf.one_hot(y_true, depth=10)
    pred = tf.maximum(pred, 1e-12)
    return -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(pred), axis=-1))


@tf.function
def train(epoch):
    for i in tf.range(epoch):
        for j, (x, y) in enumerate(train_ds):
            x = tf.expand_dims(x, -1)
            x = tf.cast(x, dtype=tf.float32)
            with tf.GradientTape() as tape:
                pred = model(x)
                loss_val = loss(pred, y)
            gradient = tape.gradient(loss_val, model.trainable_variables)
            gradient = [tf.clip_by_value(v,clip_value_min=-10,clip_value_max=10) for v in gradient]
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            if (j == 0):
                tf.print(loss_val)
            pass


train(1000)
