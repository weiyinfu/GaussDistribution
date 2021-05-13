from scipy import stats

import numpy as np
import matplotlib.pyplot as plt

"""
t分布的特点：扁而宽，它是正态分布的特例
mu/sigma

分子标准正太分布
分母卡方分布除以根号n
"""
n = 1000
mu = np.random.normal(0, 1, n)
df = 4
sigma_2 = np.random.chisquare(df, n)
t = mu / np.sqrt(sigma_2 / df)
y, x = np.histogram(t, bins=100)
y = y / np.sum(y) * (1 / (x[1] - x[0]))
plt.plot(x[:-1], y)
x = np.linspace(np.min(t), np.max(t), 100)
y = stats.t.pdf(x, df=df)
plt.plot(x, y)
plt.show()
