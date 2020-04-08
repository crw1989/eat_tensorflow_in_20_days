import tensorflow as tf
import tensorflow.keras as keras

# ====自定义损失函数
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(tf.math.reduce_max(y_train))
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
acc = tf.Variable(0.0)
optimizer = tf.optimizers.Adam(0.01)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(
    batch_size=128).prefetch(tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(len(x_train)).batch(
    batch_size=32).prefetch(
    tf.data.experimental.AUTOTUNE)

def loss(pred, y_true):
    y_true = tf.one_hot(y_true, depth=10)
    pred = tf.maximum(pred, 1e-12)
    return -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(pred), axis=-1))

def train(epoch):
    for i in tf.range(epoch):
        for j, (x, y) in enumerate(train_ds):
            x = tf.expand_dims(x, -1)
            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.int64)
            with tf.GradientTape() as tape:
                pred = model(x)
                acc.assign(tf.reduce_mean(tf.cast(tf.argmax(pred,axis=-1) == y,dtype=tf.float32)))
                loss_val = loss(pred, y)

            gradient = tape.gradient(loss_val, model.trainable_variables)
            gradient = [tf.clip_by_value(v, clip_value_min=-10, clip_value_max=10) for v in gradient]
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            if (j == 0):
                tf.print(loss_val, acc)
            pass


train(1000)

'''
class FocalLoss(losses.Loss):
    
    def __init__(self,gamma=2.0,alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def call(self,y_true,y_pred):
        
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -tf.sum(self.alpha * tf.pow(1. - pt_1, self.gamma) * tf.log(1e-07+pt_1)) \
           -tf.sum((1-self.alpha) * tf.pow( pt_0, self.gamma) * tf.log(1. - pt_0 + 1e-07))
        return loss
'''
