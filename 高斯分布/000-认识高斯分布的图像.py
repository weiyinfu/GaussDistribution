import numpy as np
import matplotlib.pyplot as plt

"""
认识研究对象
"""
def draw_gauss():
    x = np.linspace(-6, 6)
    sigma = 1
    mu = 0
    y = (
            1
            / ((2 * np.pi) ** 0.5 * sigma)
            * np.e ** (-(x - mu) ** 2 / (2 * np.pi * sigma ** 2))
    )
    plt.plot(x, y)
    plt.show()


draw_gauss()
