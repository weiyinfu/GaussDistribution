import numpy as np

# 512维，每一维都有一个mu和sigma
def get():
    n = 512
    mus = np.random.random(n) * 2 - 1
    sigmas = np.random.random(n) * 2
    xs = np.array(
        [np.random.normal(mu, sigma, (100000)) for mu, sigma in zip(mus, sigmas)]
    ).T
    feat_len = np.linalg.norm(xs, axis=1)
    print("data size", feat_len.shape)

    mean_feat_len = np.mean(feat_len)
    sigma_feat_len = np.std(feat_len)

    my_mu = np.sum(mus ** 2 + sigmas ** 2) ** 0.5
    my_sigma = np.mean(sigmas ** 2)
    print("real_mu", mean_feat_len, "real_sigma", sigma_feat_len)
    print("my_mu", my_mu, "my_sigma", my_sigma)
    print("mean(sigma)", np.mean(sigmas))
    print("mean(sigma**2)**0.5", np.mean(sigmas ** 2) ** 0.5)
    print(
        "mean(sigma**2)**0.5+mean(sigma)",
        0.5 * (np.mean(sigmas) + np.mean(sigmas ** 2) ** 0.5),
    )
    print(
        "mean(sigma**2)**0.5+mean(sigma)",
        np.mean(sigmas) * 0.25 + np.mean(sigmas ** 2) ** 0.5 * 0.5,
    )
    baga = (feat_len - my_mu) ** 2
    print("mean(baga)", np.mean(baga), np.mean(baga) ** 0.5)
    return (
        sigma_feat_len,
        np.mean(sigmas),
        np.mean(sigmas ** 2),
        np.mean(sigmas ** 2) ** 0.5,
    )


import pandas as pd

points = []
for i in range(10):
    points.append(get())
f = pd.DataFrame(points)
print(f)
"""
TODO:验证X=x1+x2+...+x512的分布

mu是很容易求出来的：
E(x_1^2+x_2^2+x_3^2)=E(x_1^2)+E(x_2^2)+E(x_3^2)=sum(mus**2+sigmas**2)


"""
