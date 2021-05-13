import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


def get_iris_data():
    """加载iris数据集"""
    x, y = load_iris(return_X_y=True)
    p = PCA(2)
    x = p.fit_transform(x)
    return x, y


def show_data(x, y):
    """展示x，y表示的散点"""
    yy = np.unique(y)
    data = []
    for i in yy:
        data.append(x[y == i])
    for i in data:
        plt.scatter(i[:, 0], i[:, 1])
    plt.show()


x, y = get_iris_data()
show_data(x, y)
