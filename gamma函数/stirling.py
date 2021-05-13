"""
斯特林函数
"""
import numpy as np
from scipy.special import gamma

x = 100
print(gamma(x))

"""
阶乘与幂函数是可以互相转换的
"""


def stirling(x):
    return np.sqrt(2 * np.pi) * np.exp(-x) * x ** (x - 0.5)


print(stirling(x))
