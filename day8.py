import tensorflow as tf

# ++++++++++低阶API++++
n = 400
x = tf.random.uniform((n, 2), minval=-10, maxval=10, dtype=tf.float32)
w0 = tf.constant([[2], [2]], dtype=tf.float32)
b0 = tf.constant(1, dtype=tf.float32)
w = tf.Variable([[0], [0]], dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)
optimizer = tf.optimizers.Adam()

@tf.function
def train(epoch):
    for _ in tf.range(epoch):
        with tf.GradientTape() as tape:
            pred = x @ w + b
            label = x @ w0 + b0
            loss = tf.losses.mean_squared_error(label, pred)
        dw,db = tape.gradient(loss,[w,b])
        optimizer.apply_gradients([(dw,w),(db,b)])
    tf.print(w, b)
    return w, b

tf.print(train(100000))
