from scipy.stats import norm
import numpy as np
from scipy.special import factorial

phi = norm.pdf


def lianfenshi(k, m):
    s = 0
    for i in range(1, m + 1):
        s = (m - i + 1) / (k + s)
    s = phi(k) / (k + s)
    return s


def standard(x):
    n = 1000
    beg = x
    end = 10
    x = np.linspace(beg, end, n)
    y = phi(x)
    s = np.sum((y[:-1] + y[1:]) / 2 * (end - beg) / (n - 1))
    return s


a = np.zeros((10, 11), dtype=np.int32)
a[0, 0] = 1
for i in range(1, a.shape[0]):
    for j in range(i + 1):
        a[i, j] = a[i - 1][j - 1] * -1 + (j + 1) * a[i - 1][j + 1]


def mine(k, m):
    x = k ** np.arange(0, m + 1)
    s = 1 / 2
    fix = norm.pdf(k, 1, 1)
    for i in range(1, m + 1):
        s += fix * np.dot(a[i - 1, :m + 1], x) * (k ** i / factorial(i)) * (-1) * (i + 1)
    return s


def taylor(x, m=9):
    s = phi(0)
    xx = x ** np.arange(m + 1)
    for i in range(1, m + 1):
        k = np.dot(a[i, :m + 1], xx)
        s += k / factorial(i) * x ** i * phi(x) * (-1) ** (i + 1)
    return s


def test():
    k = 1
    print(lianfenshi(k, 10))
    print(1 - norm.cdf(k))
    print(standard(k))


import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = [taylor(i) for i in x]
plt.plot(x, y)
plt.plot(x, phi(x))
plt.show()
