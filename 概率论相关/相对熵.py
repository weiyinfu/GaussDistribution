from scipy import stats
import numpy as np

a, b = [1, 2, 3], [4, 5, 6]


def go(a, b):
    a = np.array(a) / np.sum(a)
    b = np.array(b) / np.sum(b)
    return np.sum(a * np.log(a / b))


res = stats.entropy(a, b)
print(res, go(a, b))
