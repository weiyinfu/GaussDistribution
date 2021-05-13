import numpy as np


def gen_data(sample_num=100000, dim=1):
    sigmas = np.random.rand(dim) * 10
    mus = np.random.rand(dim) * 2 - 1
    xs = np.array(
        [np.random.normal(mu, sigma, (sample_num)) for mu, sigma in zip(mus, sigmas)]
    ).T
    return xs, sigmas, mus


def test():
    feats, sigmas, mus = gen_data(dim=50)
    norm_squares = np.linalg.norm(feats, axis=1) ** 2
    norm_squares = np.std(norm_squares)
    pred_norm_squares = (np.sum(sigmas ** 4) * 2) ** 0.5

    print("gt:{}, pred:{}".format(norm_squares, pred_norm_squares))


test()
