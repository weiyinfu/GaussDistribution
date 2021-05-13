from scipy.stats import norm

a = norm.pdf(3, loc=4, scale=5)
print(a)

for i in range(0, 10):
    print(i, norm.cdf(-i), norm.cdf(i))
