import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =================lambda自定义模型层
# 1. 如果没有要训练的参数，那么直接使用lambda层定义前向传播即可
# 2. 如果有要训练的参数时，继承layers.layer重写其中的build和call方法
model = layers.Lambda(lambda x: tf.concat([tf.nn.relu(x), tf.nn.relu(-x)], axis=-1))


# tf.print(model(tf.constant([1, 2, 3])))

# =================layers.layer自定义模型层
class Linear(layers.Layer):
    def __init__(self, units):
        super().__init__(self)
        self.units = units
        pass

    # 如果模型中包含要训练的参数，那么在此处进行定义
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)
        super().build(input_shape)
        pass
    # call方法内部定义前向传播的内容
    def call(self, inputs, **kwargs):
        return inputs @ self.w + self.b

linear = Linear(10)
# print(linear.built)
linear.build((None,28*28))
# print(linear.built)
# tf.print(linear.call(tf.random.normal(shape=(32,28*28))))
# tf.print(linear.compute_output_shape(input_shape=(1024,28*28)))
tf.print(linear.trainable_variables)