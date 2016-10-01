import argparse

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class Activation(object):
    def __call__(self, x):
        raise NotImplementedError()

    def derivative(self, x):
        raise NotImplementedError()


class Sigmoid(Activation):
    def __call__(self, x):
        return 1. / (1. + np.exp(-x))

    def derivative(self, x):
        tmp = self(x)
        return tmp * (1 - tmp)


class Linear(Activation):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)


class Layer(object):
    def __init__(self, input_dim, output_dim, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.w = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(output_dim)
        self.partial_w = np.zeros(self.w.shape)
        self.partial_b = np.zeros(self.b.shape)

        self.feed_forward(np.zeros(input_dim))

    def feed_forward(self, x):
        self.x = x
        self.affine_output = np.dot(self.w, x) + self.b
        self.output = self.activation(self.affine_output)
        return self.output

    def backprop(self, partial_output):
        # Given a partial derivative of some function wrt our output,
        # return the partial derivative wrt our inputs.
        # Also compute the derivative wrt our weights and biases, and
        # save those for later.
        error = partial_output * self.activation.derivative(self.affine_output)
        partial_input = np.dot(self.w.T, error)
        # dL / dw
        self.partial_w = np.dot(
            partial_output.reshape((self.output_dim, 1)),
            self.x.reshape((1, self.input_dim)))
        # dL / db
        self.partial_b = partial_output
        return partial_input

    def descend(self, learning_rate):
        """Adjust weights in the opposite direction of the last
        computed gradient.
        """
        self.w -= learning_rate * self.partial_w
        self.b -= learning_rate * self.partial_b


def save_graph(layers, output_file):
    def predict(x, y):
        h1 = layers[0].feed_forward(np.array([x, y]))
        h2 = layers[1].feed_forward(h1)
        h3 = layers[2].feed_forward(h2)
        return h3[0]
    X, Y, Z = [], [], []
    ys = np.linspace(-1, 1, num=50)
    for y in ys:
        xs = np.linspace(-1, 1, num=300)
        zs = []
        for x in xs:
            zs.append(predict(x, y))
        X.append(xs)
        Y.append([y for _ in xs])
        Z.append(zs)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    print X.shape, Y.shape, Z.shape
    Z -= np.min(Z)
    Z /= np.max(Z)

    cs = plt.contourf(X, Y, Z, np.arange(0, 1.1, 0.01), cmap=cm.viridis)
    plt.savefig(output_file)

def main():
    X = np.load('x.npy')
    y = np.load('y.npy')

    layer1 = Layer(2, 24, Sigmoid())
    layer2 = Layer(24, 12, Sigmoid())
    layer3 = Layer(12, 1, Linear())

    num_epochs = 100
    learning_rate = 0.001
    num_examples = X.shape[0]

    save_graph([layer1, layer2, layer3], 'my_sgd_plots/epoch_000.png')

    for i in range(num_epochs):
        print 'Epoch %d...' % (i + 1,)
        sum_squares = 0.0
        for j in range(num_examples):
            x = X[j]
            expected = y[j]
            h1 = layer1.feed_forward(x)
            h2 = layer2.feed_forward(h1)
            h3 = layer3.feed_forward(h2)
            output = h3[0]
            delta = output - expected
            cost_contribution = (delta * delta) / 2.
            sum_squares += cost_contribution
            cost_derivative = output - expected
            partials3 = layer3.backprop(np.array([delta]))
            partials2 = layer2.backprop(partials3)
            layer1.backprop(partials2)

            layer3.descend(learning_rate)
            layer2.descend(learning_rate)
            layer1.descend(learning_rate)
        learning_rate *= 0.99
        mse = sum_squares / num_examples
        print 'Complete. MSE %.06f' % (mse,)
        save_graph([layer1, layer2, layer3], 'my_sgd_plots/epoch_%03d.png' % (i + 1,))


if __name__ == '__main__':
    main()
