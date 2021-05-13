from statsmodels.robust import scale
import numpy as np
from scipy.stats import norm

a = np.random.random(10)

"""
MAD:median absolute devariate

mad=median(a-median(a))/norm.ppf(0.75)
"""


def go(a: np.ndarray, c=norm.ppf(3 / 4.)):
    center = np.median(a)
    return np.median((np.abs(a - center))) / c


print(norm.ppf(3 / 4))
print(a)
ans = scale.mad(a)
print(ans, go(a))
