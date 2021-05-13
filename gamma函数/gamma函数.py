import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

a = np.linspace(-5, 5, 190)
plt.plot(a, gamma(a))
plt.title('gamma')
plt.show()
