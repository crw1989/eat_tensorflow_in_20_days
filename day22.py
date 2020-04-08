import tensorflow as tf

# ============使用不同的优化器，来计算出ax**2+b*x+c的最小值点
#1.optimizer.clipnorm和clip_value会进行梯度截断
#=============使用optimizer.apply_gradients()
# optimizer = tf.optimizers.Adam()
# optimizer.minimize()
# optimizer.apply_gradients()
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


# train(100000)

#=========================使用optimiers.minimise()

@tf.function
def loss():
    return a * x ** 2 + b * x + c


@tf.function
def train(epoch):
    for i in tf.range(epoch ** 2):
        optimizer.minimize(loss,[x])
        tf.print("epoch",i," ",optimizer.iterations,"：",x,"")
train(1000)
#=========================使用model.fit()