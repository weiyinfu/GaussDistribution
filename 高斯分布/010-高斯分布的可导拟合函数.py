import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import optimize

"""
正态分布的拟合是一个很难调节的神经网络

并非每次运行都能找到合适的解
"""


def gauss(xs, mu=0, sigma=1):
    return 1 / (sigma * (2 * np.pi) ** 0.5) * np.exp(-(xs - mu) ** 2 / (2 * sigma ** 2))


def my_fun(v):
    x = np.linspace(-3.5, 0.1, 1000)
    vv = np.arange(1, len(v)) * v[1:]
    y = (
        (x.reshape(-1, 1) ** np.arange(len(vv)))
        @ vv
        * np.exp((x.reshape(-1, 1) ** np.arange(len(v))) @ v)
    )
    yy = gauss(x)
    l = y - yy
    return l


def draw(v):
    x = np.linspace(-3.5, 3.5, 100)
    vv = np.arange(1, len(v)) * v[1:]
    y = (
        (x.reshape(-1, 1) ** np.arange(len(vv)))
        @ vv
        * np.exp((x.reshape(-1, 1) ** np.arange(len(v))) @ v)
    )
    yy = gauss(x)
    plt.plot(x, y, label="mine")
    plt.plot(x, yy, label="real")
    plt.legend()
    plt.ylim(0, 1)
    plt.xlim(x.min(), x.max())
    plt.show()


def print_latex(v):
    def pow(y):
        if y == 0:
            return ""
        if y == 1:
            return "x"
        return f"x^{y}"

    s = r"""\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}="""
    pre = f"{v[1]:.3f}"
    for i in range(1, len(v) - 1):
        if v[i + 1] >= 0 and i > 0:
            pre += "+"
        pre += f"{v[i+1]:.3f} \\times {i+1}{pow(i)}"
    post = ""
    for i in range(len(v)):
        if v[i] >= 0 and i > 0:
            post += "+"
        post += f"{v[i]:.3f}{pow(i)}"
    ss = f"({pre}) \\times e^{{{post}}}"
    s = s + ss + r"\\"
    print(s)


def get_params_by_tf(n):
    x = np.linspace(-3.5, 0.1, 1000)
    y = gauss(x)
    x_place = tf.placeholder(dtype=tf.float32, shape=(None,))
    y_place = tf.placeholder(dtype=tf.float32, shape=(None,))
    v = tf.Variable(tf.random.uniform((n,), minval=0, maxval=0.1))
    learn_rate = tf.Variable(0.01, trainable=False)
    pre = 0
    for i in range(n - 1):
        pre += x_place ** i * v[i + 1] * (i + 1)
    post = 0
    for i in range(n):
        post += x_place ** i * v[i]
    y_output = pre * tf.exp(post)
    # lo = tf.reduce_max(tf.abs(y_output - y_place))
    # lo = tf.reduce_sum((y_output - y_place) ** 2)
    # lo = tf.reduce_sum(tf.abs(y_output - y_place))
    lo = tf.linalg.norm(y_output - y_place)
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(lo)
    initer = tf.global_variables_initializer()
    with tf.Session() as sess:
        l = 1
        res = None
        sess.run(initer)
        while l > 0.1:
            last = None
            for epoch in range(10000):
                _, l = sess.run((train_op, lo), feed_dict={x_place: x, y_place: y})
                if epoch % 1000 == 0:
                    rate = sess.run(learn_rate) * 0.99
                    rate = max(rate, 0.0001)
                    sess.run(tf.assign(learn_rate, rate))
                    print(epoch, l)
                    if last is None:
                        last = l
                    else:
                        if abs(last - l) < 0.001:
                            print("last loss", l)
                            break
                        last = l
            res = sess.run(v)
        return res


def get_param_by_fsolve(n):
    v = optimize.leastsq(my_fun, np.random.random(n) - 0.5)
    print("loss", v[1])
    return v[0]


v = get_params_by_tf(6)
print(v)
print_latex(v)
draw(v)
# print_latex(-0.5, np.array([0, -(2 / np.pi) ** 0.5, -0.356]))

r"""
\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}=(0.789-0.352 \times 2x) \times e^{-0.688+0.789x-0.352x^2}\\

\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}=(0.798-0.324 \times 2x+0.026 \times 3x^2) \times e^{-0.694+0.798x-0.324x^2+0.026x^3}\\

\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}=(0.798-0.319 \times 2x+0.035 \times 3x^2+0.004 \times 4x^3) \times e^{-0.693+0.798x-0.319x^2+0.035x^3+0.004x^4}\\

\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}=(0.798-0.319 \times 2x+0.035 \times 3x^2+0.004 \times 4x^3) \times e^{-0.693+0.798x-0.319x^2+0.035x^3+0.004x^4}\\

\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}=(0.798-0.318 \times 2x+0.035 \times 3x^2+0.005 \times 4x^3-0.001 \times 5x^4) \times e^{-0.694+0.798x-0.318x^2+0.035x^3+0.005x^4-0.001x^5}\\
"""
