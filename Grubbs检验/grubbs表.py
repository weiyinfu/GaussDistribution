import numpy as np
from scipy.stats import t
from tabulate import tabulate

"""
grubbs表
"""


def get(alpha, n):
    ta = t.isf(alpha / (2 * n), n - 2) ** 2
    return (n - 1) / (n ** 0.5) * (ta / (n - 2 + ta)) ** 0.5


def print_table():
    a = []
    alpha_set = [0.95, 0.99]

    for n in range(3, 11):
        row = [n]
        for alpha in alpha_set:
            row.append(get(alpha, n))
        a.append(row)
    a = np.array(a)
    print(tabulate(a, headers=list(alpha_set)))


def do_experiment():
    a = np.random.normal(loc=1.5, scale=1.2, size=100)
    a = list(a) + [100]
    thresh = get(0.05, len(a))
    g = np.max(np.abs(a - np.mean(a)) / np.std(a))
    print(g, thresh, g > thresh)


def esd(a, alpha=0.05):
    # 从一个数组中多次移除掉数据
    a = np.array(a)
    a = a.copy()
    a.sort()
    i = 0
    j = len(a) - 1
    while 1:
        if j - i + 1 <= 3:
            break
        thresh = get(alpha, j - i)
        m = np.mean(a[i:j + 1])
        sigma = np.std(a[i:j + 1])
        ri = a[j] - m
        le = m - a[i]
        g = max(le, ri) / sigma
        if g < thresh:
            # everything is ok
            break
        if le > ri:
            i += 1
        else:
            j -= 1
    return a[i:j + 1]


# print_table()
# do_experiment()
print(esd([-3, 0.1, 0.2, 0.3, 0.4, 0.5, 3]))
