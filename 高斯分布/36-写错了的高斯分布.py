import numpy as np
import matplotlib.pyplot as plt

"""
随机写几个随机数，让它们产生新的随机分布
求这个随机分布的表达式

这个分布有点像伊斯兰教教堂的塔尖

这类问题有无数个，随意一个就能够耗费半晌时光。
"""
n = 100000
r, theta = np.random.random((2, n))
theta *= 2 * np.pi
a = np.concatenate((r * np.sin(theta), r * np.cos(theta)))
plt.hist(a, bins=100)
plt.show()
