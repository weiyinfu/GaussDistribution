import matplotlib.pyplot as plt
import numpy as np

"""
x服从一元高斯分布
求E(x^2)

D(x)=E(x)^2-E(x^2)
显然E(x^2)=mu**2-sigma**2
这不仅对于正态分布成立，对于任何一种分布都成立
"""
mu = -3
sigma = 6
xs = np.random.normal(mu, sigma, (100000))


def gauss(xs, mu, sigma):
    return 1 / (sigma * (2 * np.pi) ** 0.5) * np.exp(-(xs - mu) ** 2 / (2 * sigma ** 2))


plt.hist(xs, 100)
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
ax = plt.twinx()
ax.plot(x, gauss(x, mu, sigma), c="r")
plt.show()
print("abs(x)", np.mean(np.abs(xs)))
print("abs(x^2)", np.mean(xs ** 2), mu ** 2 + sigma ** 2)
