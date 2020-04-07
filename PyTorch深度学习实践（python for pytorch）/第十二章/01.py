import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("C:/Users/15064/Desktop/china010.png")

h, w, _ = img.shape
xs, ys = [], []
for i in range(100):
    mean = w*np.random.rand(), h*np.random.rand()
    a = 50 + np.random.randint(50, 200)
    b = 50 + np.random.randint(50, 200)
    c = (a + b)*np.random.normal()*0.2
    cov = [[a, c], [c, b]]
    count = 200
    x, y = np.random.multivariate_normal(mean, cov, size=count).T
    xs.append(x)
    ys.append(y)
x = np.concatenate(xs)
y = np.concatenate(ys)

hist, _, _ = np.histogram2d(x, y, bins=(np.arange(0, w), np.arange(0, h)))
hist = hist.T
plt.imshow(hist);

from scipy.ndimage import filters
heat = filters.gaussian_filter(hist, 10.0)
plt.imshow(heat);

