import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


# There is no point to this function, other than to make a complicated
# surface to learn.
def f(x, y):
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    a = np.sin(4.0 * r * r - 0.25 * r)
    b = np.sin(2 * theta) * np.sin(theta)
    c = np.cos(6 * (theta + 0.1))
    k = 0.3 * (np.sin(theta) + 1)
    z = k * a + ((1 - k) / 2.0) * b + ((1 - k) / 2.0) * c
    return z


def draw(filename=None):
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    z = f(x, y)
    z -= np.min(z)
    z /= np.max(z)

    cs = plt.contourf(x, y, z, np.arange(0, 1.1, 0.01), cmap=cm.viridis)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def sample(n):
    X = np.random.randn(n, 2)
    X = 2.0 * np.random.rand(n, 2) - 1.0
    y = f(X[:,0], X[:,1])
    return X, y


if __name__ == '__main__':
    draw('target.png')
    X, y = sample(200000)
    np.save('x.npy', X)
    np.save('y.npy', y)
