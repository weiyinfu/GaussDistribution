import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

"""
n个正态分布随机变量的平方和 服从卡方分布
"""

df = 5
a = np.random.normal(0, 1, (10000, df))
a = np.linalg.norm(a, axis=1) ** 2
y, x = np.histogram(a, bins=100)
y = y / np.sum(y) * (1 / (x[1] - x[0]))
plt.plot(x[1:], y)
x = np.linspace(np.min(a), np.max(a), 100)
y = stats.chi2.pdf(x, df=df)
plt.plot(x, y)
plt.show()
