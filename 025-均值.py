import matplotlib.pyplot as plt
import numpy as np

# 512维，每一维都有一个mu和sigma
n = 512
mus = np.random.random(n) * 2 - 1
sigmas = np.random.random(n) * 2
xs = np.array(
    [np.random.normal(mu, sigma, (100000)) for mu, sigma in zip(mus, sigmas)]
).T
feat_len = np.linalg.norm(xs, axis=1) ** 2
print("data size", feat_len.shape)

mean_feat_len = np.mean(feat_len)
sigma_feat_len = np.std(feat_len)


def gauss(xs, mu, sigma):
    return 1 / (sigma * (2 * np.pi) ** 0.5) * np.exp(-(xs - mu) ** 2 / (2 * sigma ** 2))


_, edges, _ = plt.hist(feat_len, 100)
ax = plt.twinx()
ax.plot(
    edges,
    gauss(edges, mean_feat_len, sigma_feat_len),
    label="$\mu=%s\quad \sigma=%s$" % (mean_feat_len, sigma_feat_len),
)
plt.legend()
my_mu = np.sum(mus ** 2 + sigmas ** 2)
print(my_mu, mean_feat_len)
plt.show()
