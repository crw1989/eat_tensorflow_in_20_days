import numpy as np

# =============python切片，只是返回原始数据的视图，当原始数据被改变时，其他会被改变
arr0 = np.arange(12)
arr1 = arr0[:]
# print(id(arr0))
# print(id(arr1))

arr0[0] = 100
# print(arr0)
# print(arr1)

brr0 = np.zeros_like(arr0)
brr0[...] = arr0
arr0[0] = 1000

# print(brr0)
# print(arr0)

# ============ python迭代器
# 1. 迭代器是可以记住遍历位置的对象，从第一个元素开始访问，直到所有元素访问完成为止。迭代器只能前进不能后退
# 2. 迭代器的两个方法:iter()、next()。iter()生成迭代器对象,next()用下访问下一个元素

arr = [1, 2, 3, 4]


# for v in iter(arr):
#     print(v)


class MyClass():
    def __init__(self):
        self.arr = [i for i in range(10)]
        self.idx = 0
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if (self.idx > len(self.arr)):
            raise StopIteration()
        return self.arr[self.idx - 1]


# for v in MyClass():
#     print(v)

# ===================生成器
# 1. 使用了yield关键字的函数称之为生成器
# 2. 该函数返回的是一个迭代器对象
# 3. 每次执行到yield关键字时，会保存当前状态，暂停执行。下一次调用next（）方法执行时再继续从保存的位置执行。
def fibonaccie(n):
    a, b = 1, 1
    while a < n:
        yield a
        a, b = b, a + b

# for v in fibonaccie(100):
#     print(v)


'''python模块
python模块是一个python文件，以.py结尾，包含了Python对象的定义和python语句。
python模块中可以定义类、模块、变量也可以包含可执行代码


1. import语句。可以使用import 模块名来导入模块。使用模块内部的函数名时使用"."，模块名.函数名
2. 一个模块只会导入一次，不管你使用了多少个import语句
3. from .... import ... 会将模块中的一部分导入到当前的名称空间中
4. 搜索路径，在进行模块导入时，首先会查看当前工作路径下是否包含该模块的定义
'''