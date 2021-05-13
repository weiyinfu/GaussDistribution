import matplotlib.pyplot as plt
import numpy as np

"""
泰勒展开的缺点：幂函数本身的发散性

以log(1+x)=x-1/2*x^2+1/3*x^3..为例，
在(1,2)区间上，泰勒展开拟合得非常好，但是在10区间上拟合得非常差
"""


def log1addx(x):
    # log(1+x)的泰勒展开
    s = np.sum([-x ** i * (-1) ** i / i for i in range(1, 10)], axis=0)
    return s


def draw1addx():
    x = np.linspace(-0.5, 1.5)
    y = np.log(1 + x)
    yy = log1addx(x)
    plt.plot(x, y, label="real")
    plt.plot(x, yy, label="taylor")
    plt.legend()
    plt.show()


draw1addx()
