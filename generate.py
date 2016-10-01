import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    a = np.sin(4.0 * r * r - 0.25 * r)
    b = np.sin(2 * theta) * np.sin(theta)
    c = np.cos(6 * (theta + 0.1))
    k = 0.3 * (np.sin(theta) + 1)
    z = k * a + ((1 - k) / 2.0) * b + ((1 - k) / 2.0) * c

    #print np.min(theta), np.max(theta)
    #theta = np.sqrt(theta)
    #tmp = theta / (2 * math.pi)
    #tmp = tmp * tmp
    #theta = 2 * math.pi * theta
    #a = np.sin(1.5 * r)
    #a = np.sin(theta)
    #z = 0.6 * np.sin(4.0 * r * r - 0.25 * r) + \
    #    np.sin(2 * theta) * np.sin(theta)
    return z


def draw():
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    z = f(x, y)
    z -= np.min(z)
    z /= np.max(z)
    print np.min(z), np.max(z)

    cs = plt.contourf(x, y, z, np.arange(0, 1.1, 0.01), cmap=cm.viridis)
    plt.show()


def sample(n):
    #X = np.random.randn(n, 2)
    X = 2.0 * np.random.rand(n, 2) - 1.0
    print X.shape, np.min(X), np.max(X)
    y = f(X[:,0], X[:,1])
    return X, y


if __name__ == '__main__':
    draw()
    #X, y = sample(200000)
    #np.save('x.npy', X)
    #np.save('y.npy', y)
