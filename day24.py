import tensorflow as tf
from tensorflow.keras.layers import *

BATCH_SIZE = 128
START = tf.Variable(0)
acc_test = tf.Variable(0.0)
loss_test = tf.Variable(0.0)
USE_PRE = True
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(BATCH_SIZE).prefetch(
    BATCH_SIZE)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

model = tf.keras.Sequential([
    Conv2D(32, 3, padding='same', use_bias=False, input_shape=(28, 28, 1)),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(32, 3, padding='same', strides=2, use_bias=False),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(64, 3, padding='same', use_bias=False),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(64, 3, padding='same', strides=2, use_bias=False),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(128, 3, padding='same', use_bias=False),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(128, 3, padding='same', strides=2, use_bias=False),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(256, 3, padding='same', use_bias=False),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(256, 3, padding='same', strides=2, use_bias=False),
    BatchNormalization(),
    LeakyReLU(),
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax'),
])

model.summary()
optimizers = tf.keras.optimizers.Adam()
checkpoint = tf.train.Checkpoint(m=model, optim=optimizers, start=START)
manager = tf.train.CheckpointManager(checkpoint, './checkpoint/', max_to_keep=5)


def loss_func(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False))


def train(epoch):
    if USE_PRE:
        checkpoint.restore(tf.train.latest_checkpoint('./checkpoint'))
    for i in tf.range(START, epoch):
        tf.print('epoch:', i)
        for x, y in train_data:
            x = tf.reshape(tf.cast(x, dtype=tf.float32), shape=(-1, 28, 28, 1))
            with tf.GradientTape() as tape:
                pred = model(x)
                loss_val = loss_func(y, pred)
            grads = tape.gradient(loss_val, model.trainable_variables)
            optimizers.apply_gradients(zip(grads, model.trainable_variables))

            acc = tf.reduce_mean(tf.metrics.sparse_categorical_accuracy(y, pred))
            tf.print(optimizers.iterations % (len(x_train) // BATCH_SIZE), '/', (len(x_train) // BATCH_SIZE), ' loss:',
                     loss_val, ' acc:', acc)
        acc_test.assign(tf.constant(0.0))
        loss_test.assign(tf.constant(0.0))
        for x, y in test_data:
            x = tf.reshape(tf.cast(x, dtype=tf.float32), shape=(-1, 28, 28, 1))
            pred = model(x)
            acc = tf.reduce_sum(tf.keras.metrics.sparse_categorical_accuracy(y, pred))
            loss = tf.reduce_sum(tf.keras.metrics.sparse_categorical_crossentropy(y, pred))
            acc_test.assign_add(acc)
            loss_test.assign_add(loss)
        tf.print('test data loss:', loss_test / len(x_test), ' acc:', acc_test / len(x_test))
        tf.print('\n')

        manager.save()
train(100)
