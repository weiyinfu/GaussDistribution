import numpy as np

"""
只考虑标准高斯分布的0~3mu区间上的积分
"""

mu = 0
sigma = 1


def gauss(x):
    return 1 / ((2 * np.pi) ** 0.5) * np.e ** (-0.5 * ((x - mu) / sigma) ** 2)


n = 100
x = np.linspace(0, 3 * sigma, n)
y = gauss(x)
y_sum = np.cumsum(y * (3 * sigma) / n) + 0.5

import matplotlib.pyplot as plt


def mock(x):
    # 使用公式法计算高斯分布的积分（存在一定误差），请解释这个公式是如何来的
    return 1 - 0.5 * np.exp(-0.356 * x ** 2 - np.sqrt(2 / np.pi) * x)


plt.plot(x, y_sum, label="sum")
plt.plot(x, y, label="gauss")
plt.plot(x, mock(x), label="mock")
plt.legend()
plt.show()
