import numpy as np
import matplotlib.pyplot as plt

"""
Box-muller算法
"""
n = 100000
r, theta = np.random.random((2, n))
theta *= 2 * np.pi
r = np.sqrt(-2 * np.log(1 - r))
# a = np.concatenate((r * np.sin(theta), r * np.cos(theta)))
a = r * np.sin(theta)  # 即便只取一般也是对的
plt.hist(a, bins=100)
plt.show()
