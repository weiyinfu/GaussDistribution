import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm

"""
n个正态分布随机变量的平方和 服从卡方分布

n个小球随机投向k个盒子，投到每个盒子的概率为p1,p2,p3,p4,p5
实际投出的每个盒子中的球的个数为n1,n2,n3,n4,n5，
求统计量p的概率密度函数

卡方检验
"""

df = 5  # 类别的个数
# 设置每个盒子得到球的概率
box = np.random.random(5)
box = box / np.sum(box)
cum_box = np.cumsum(box)


def get_freq():
    n = 1000
    put = np.random.random(1000)  # 每次100个球投到盒子里面
    real = np.searchsorted(cum_box, put)
    box_count = np.bincount(real)
    bo = np.array(n * box, dtype=np.int)
    a = np.zeros_like(bo)
    a[:] = box_count
    target = np.sum((a - bo) ** 2 / bo)
    return target


a = [get_freq() for i in tqdm(range(10000))]
y, x = np.histogram(a, bins=100)
y = y / np.sum(y) * (1 / (x[1] - x[0]))
plt.plot(x[:-1], y)
x = np.linspace(x[0], x[-1], 100)
y = stats.chi2.pdf(x, df=df - 1)
plt.plot(x, y)
plt.show()
