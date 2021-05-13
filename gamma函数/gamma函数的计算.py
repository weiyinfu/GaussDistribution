from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt

n = 30
x = np.linspace(0, np.pi / 2, n)
a = np.tan(x)
b = (1 + np.tan(x) ** 2) * (np.pi / 2 / n) * np.exp(-np.tan(x))


def ga(x):
    return [np.sum(b * a ** (i - 1)) for i in x]


t = np.linspace(1, 10, 100)
plt.plot(t, ga(t), label='mine')
plt.plot(t, gamma(t), label='ans')
plt.legend()
plt.show()
