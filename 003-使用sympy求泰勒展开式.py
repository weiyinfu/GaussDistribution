import sympy as s

x = s.Symbol("x")


def taylor(y, x, n=30):
    """
    对给定的y计算泰勒展开，方法是不停地求导
    :param y:
    :param x:
    :param n:
    :return:
    """
    yy = y
    a = [0] * n
    for i in range(n):
        v = yy.evalf(subs={x: 0})
        yy = s.diff(yy)
        a[i] = v
    return sum(a[i] / s.factorial(i) * (x ** i) for i in range(n))


def gauss(x):
    mu = 0
    sigma = 1
    y = 1 / ((2 * s.pi) ** 0.5 * sigma) * s.E ** (-(x - mu) ** 2 / (2 * sigma ** 2))
    return y


def show(y, x):
    yy = taylor(y, x)
    import numpy as np

    x_v = np.linspace(-1, 4, num=40)
    y_v = [yy.evalf(subs={x: i}) for i in x_v]
    yy_v = [y.evalf(subs={x: i}) for i in x_v]
    import matplotlib.pyplot as plt

    plt.plot(x_v, y_v, label="mock")
    plt.plot(x_v, yy_v, label="real")
    plt.legend()
    plt.title(str(y))
    plt.ylim(0, 1)
    plt.show()

    print(y, "y'=", yy)


# show(s.E ** x, x)
# show(s.cos(x), x)
# show(s.sin(x), x)
# show(s.log(1 + x), x)
# show(s.tan(x), x)
show(gauss(x), x)

# print(taylor(s.log(1-gauss(x)),x))
