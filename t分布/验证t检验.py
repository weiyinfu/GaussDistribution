import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

"""
t检验：t检验的本质就是t分布
t分布的特点：扁而宽，它是正态分布的特例
mu/sigma

t检验的关键在于构造t统计量：
分子是均值-mu
分母是标准差除以根号n
"""
n = 10000
normal_mu = 4.5
normal_sigma = 3
# 观测到的变量
a = np.random.normal(normal_mu, normal_sigma, (n, 10))
df = 4
t = (np.mean(a, axis=1) - normal_mu) / (np.std(a) / np.sqrt(10))
y, x = np.histogram(t, bins=100)
y = y / np.sum(y) * (1 / (x[1] - x[0]))
plt.plot(x[:-1], y)
x = np.linspace(np.min(t), np.max(t), 100)
y = stats.t.pdf(x, df=df - 1)
plt.plot(x, y, label=f"df={df - 1}")
y = stats.t.pdf(x, df=df)
plt.plot(x, y, label=f'df={df}')
plt.legend()
plt.show()
