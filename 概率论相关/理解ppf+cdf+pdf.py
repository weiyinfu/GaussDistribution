from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def gauss(a, mu=0, sigma=1):
    return 1 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((a - mu) / sigma) ** 2)


def see_gauss():
    x = np.linspace(-3, 3, 100)
    plt.plot(x, norm.pdf(x), label='real')
    plt.plot(x, gauss(x), label='mine')
    plt.legend()
    plt.show()


x = 0.5
y = norm.cdf(x, loc=1, scale=2)
print(norm.ppf(y, loc=1, scale=2))
