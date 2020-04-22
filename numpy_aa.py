import numpy as np

arr = np.arange(12)

# ==================size和itemsize
# print(arr.itemsize*arr.size)
# ==================nonzero：返回非0元素下标
arr = np.array([1, 2, 0, 3, 0])
idx = arr.nonzero()
# print(arr[idx])
# =================== min、max、mean
# print(arr.min())
# print(arr.max())


# ===================pad
arr = np.ones((5, 5))
arr = np.pad(arr, pad_width=[(1, 2), (3, 4)], mode='constant', constant_values=0)
# print(arr)

# =================== nan、inf
# print(0*np.nan)
# print(np.inf==np.nan)
# print(0.3 == (0.1 * 3)) False

# ==================== mean、std
arr = np.random.random(size=(10, 10))
# print((arr - np.mean(arr)) / np.std(arr))

# ==================== tile
arr = np.array([[1, 0], [0, 1]])
# print(np.tile(arr,[3,3]))

# ==================== @
arr = np.ones(shape=(5, 3))
brr = np.ones(shape=(3, 2))
# print(arr @ brr)

# ==================== & 与运算 | 或运算 ~非运算
arr = np.arange(12).reshape(3, 4)
arr[(arr > 3) & (arr < 8)] *= -1
# print((arr > 3) & (arr < 9))

#  =================== 提取数据的整数部分
val = np.random.random(1)
# print(np.floor(val))
# print(np.ceil(val))
# print(val.astype(np.int32))
# print(np.trunc(val))
# print(np.round(0.49).astype(np.int))



# =================== excercise
# yesterday = np.datetime64('today','D') - np.timedelta64(1,'D')
yesterday = np.datetime64('today','D')
yesterday = np.timedelta64(1,'D')
# print(yesterday)
arr0 = np.zeros(shape=(5,5)) + np.arange(5).reshape(5,1)
# print(arr0)

# ==================== numpy 视图

arr = np.arange(3*12).reshape(3,4,3)
# print(arr)
def func(brr):
    global arr
    print(arr is brr)
    brr[:,:,0] = -1
func(arr)
# print(arr)

# ==================== numpy
arr = np.arange(36).reshape(6,6)
print(arr)
print('================')
print(arr[1:-1])