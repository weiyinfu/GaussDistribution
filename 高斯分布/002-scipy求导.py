from scipy.misc import derivative

call_time = 0
"""
http://liao.cpython.org/scipy17/

调用此函数的次数为pointCount×3

使用scipy进行数值分析式的求导
"""


def f(x):
    global call_time
    call_time += 1
    return x ** 5


for x in range(1, 4):
    print(derivative(f, x, dx=1e-6))
    print(call_time)
