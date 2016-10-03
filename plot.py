import glob
import os
import re
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import sgd


def plot(basedir, ts, output_file):
    template = os.path.join(basedir, '%d_%s_%s.npy')
    layer1 = sgd.Layer(2, 24, sgd.Sigmoid())
    layer2 = sgd.Layer(24, 12, sgd.Sigmoid())
    layer3 = sgd.Layer(12, 1, sgd.Linear())

    layer1.w = np.load(template % (ts, 'l1', 'w'))
    layer1.b = np.load(template % (ts, 'l1', 'b'))
    layer2.w = np.load(template % (ts, 'l2', 'w'))
    layer2.b = np.load(template % (ts, 'l2', 'b'))
    layer3.w = np.load(template % (ts, 'l3', 'w'))
    layer3.b = np.load(template % (ts, 'l3', 'b'))

    def predict(x, y):
        h1 = layer1.feed_forward(np.array([x, y]))
        h2 = layer2.feed_forward(h1)
        h3 = layer3.feed_forward(h2)
        return h3[0]

    X, Y, Z = [], [], []
    ys = np.linspace(-1, 1, num=100)
    for y in ys:
        xs = np.linspace(-1, 1, num=100)
        zs = []
        for x in xs:
            zs.append(predict(x, y))
        X.append(xs)
        Y.append([y for _ in xs])
        Z.append(zs)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    Z -= np.min(Z)
    Z /= np.max(Z)

    cs = plt.contourf(X, Y, Z, np.arange(0, 1.1, 0.01), cmap=cm.viridis)
    plt.savefig(output_file)


def main():
    for fname in glob.glob('progress/*_l1_w.npy'):
        mo = re.match(r'progress/(\d+)_.*', fname)
        ts = int(mo.group(1))
        outputfile = 'synthetic_plots/%d.png' % (ts,)
        plot('progress', ts, outputfile)


if __name__ == '__main__':
    main()
