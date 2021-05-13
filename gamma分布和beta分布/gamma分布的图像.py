import scipy.stats as stats

import numpy as np
import matplotlib.pyplot as plt


def gamma():
    x = np.linspace(0.1, 1, 100)
    for alpha in [0.1, 1, 1.2]:
        for beta in [0.1, 1, 1.2]:
            y = stats.gamma.pdf(x, a=alpha, b=beta)
            plt.plot(x, y, label=f"{alpha},{beta}")
    plt.legend()
    plt.show()

def beta():
    x = np.linspace(0.1, 1, 100)
    for alpha in [0.1, 1, 1.2]:
        for beta in [0.1, 1, 1.2]:
            y = stats.beta.pdf(x, a=alpha, b=beta)
            plt.plot(x, y, label=f"{alpha},{beta}")
    plt.legend()
    plt.show()

# gamma()
beta()