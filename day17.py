import tensorflow as tf
import numpy as np

# ============创建数据管道、数据管道操作、效率提升

# ===========数据管道:Dataset之numpy
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('./data')
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=len(x_train)).batch(
    batch_size=32).prefetch(buffer_size=1024)


# for x, y in ds:
#     tf.print(x.shape)


# ===========数据管道：Dataset之generator
def func(batch_size=32):
    while (True):
        x, y = tf.random.uniform(shape=[batch_size, 5, 5], minval=-10, maxval=10), 1
        yield (x, y)


ds = tf.data.Dataset.from_generator(func, output_types=(tf.float32, tf.int32))
# for x,y in ds:
#     print(x.shape)
# ============数据管道：Dataset之文件名
ds = tf.data.Dataset.list_files(r'C:\Users\LenovoPC\PycharmProjects\eat_tensorflow_in_20_days\*.py')


def func(file_path):
    count = 0
    file = tf.io.read_file(file_path)
    # tf.image.decode_jpeg()
    return tf.fill(dims=[5, 5], value=count)


# for x in ds.map(func, tf.data.experimental.AUTOTUNE):
#     tf.print(x)

# ==================数据管道操作:map将函数应用到每一个元素
ds = tf.data.Dataset.from_tensor_slices(['hello world', 'hello beiging', 'hello america'])
# for v in ds.map(lambda x:tf.strings.split(x,' ')):
#     print(v)
# ==================数据管道操作：flat_map
# ds = tf.data.Dataset.from_tensor_slices(['hello world','hello beiging','hello america'])
# for v in ds.flat_map(lambda x:tf.data.Dataset.from_tensor_slices(tf.strings.split(x))):
#     print(v)
# ==================tf.data.Dataset.zip:将两个不同长度的Dataset进行合并
b1 = tf.data.Dataset.range(0, 10, 3)
a1 = tf.data.Dataset.range(1, 10, 3)
c1 = tf.data.Dataset.range(2, 10, 3)
# for p in tf.data.Dataset.zip((b1,a1,c1)):
#     tf.print(p)
# ====================tf.data.Dataset.concatenate
a1 = tf.data.Dataset.range(0, 3)
b1 = tf.data.Dataset.range(3, 6)
# for v in tf.data.Dataset.concatenate(a1, b1):
#     print(v)
# ===================tf.data.Dataset.reduce
a1 = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0, 4.0])
# tf.print(a1.reduce(0.0, lambda x, y: tf.add(x,y)))
# =================== tf.data.Dataset.shuffle
a1 = tf.data.Dataset.range(10).shuffle(buffer_size=10)
# for v in a1:
#     tf.print(v)
# =================== tf.data.Dataset.repeat
a1 = tf.data.Dataset.range(10).repeat(3).shuffle(buffer_size=30)
# for v in a1:
#     tf.print(v)
# ====================take，采样
a1 = tf.data.Dataset.range(10)
# for v in a1.take(3): # 从头开始取3个
#     print(v)
# ====================shard，间隔采集
a1 = tf.data.Dataset.range(10)
# for v in a1.shard(3,0):
#     print(v)
# ====================batch
a1 = tf.data.Dataset.range(10).batch(3)
for v in a1:
    print(v)
# ==================训练过程中的耗时：参数训练、数据准备
# 1. map的时候使用num_parallel_calls指定同时进行map处理的进程数
# 2. prefetch使用预取的方式
# 3. map转换时可以先batch再map
