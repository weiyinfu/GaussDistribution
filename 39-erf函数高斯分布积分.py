from scipy.stats import norm
import matplotlib.pyplot as plt
from math import *
import numpy  as np

x = np.linspace(0, 4, 100)
y = norm.cdf(x)


def phi(x):
    # 'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


plt.plot(x, y)
plt.plot(x, [phi(i) for i in x])
plt.show()
