import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy.special import gamma


#                                gamma((df+1)/2)
# t.pdf(x, df) = ---------------------------------------------------
#                sqrt(pi*df) * gamma(df/2) * (1+x**2/df)**((df+1)/2)
def my_t(x, df):
    return gamma((df + 1) / 2) / (np.sqrt(np.pi * df) * gamma(df / 2) * (1 + x ** 2 / df) ** ((1 + df) / 2))


print('比较t-分布与标准正态分布')
x = np.linspace(-3, 3, 100)
plt.plot(x, t.pdf(x, 1), label='df=1')
plt.plot(x, t.pdf(x, 2), label='df=20')
plt.plot(x, t.pdf(x, 100), label='df=100')
plt.plot(x, my_t(x, 2), label='my_t,df=20')
plt.plot(x[::5], norm.pdf(x[::5]), 'kx', label='normal')
plt.legend()
plt.show()
