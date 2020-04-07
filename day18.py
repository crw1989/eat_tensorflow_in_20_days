import tensorflow as tf
from tensorflow import *
from tensorflow.keras import layers
# ===================模型中使用激活函数的方式
model = keras.Sequential()
model.add(layers.Dense(32))
model.add(layers.Activation(tf.nn.leaky_relu))
model.add(layers.Dense(32))
model.add(layers.Activation(tf.nn.leaky_relu))
model.build(input_shape=(5,5,5))
model.summary()
