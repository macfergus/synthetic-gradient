import argparse
import thread
import time
import sys

import sgd
import synthgrad

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-server', default='http://localhost:5000')
    parser.add_argument('--oracle-server', default='http://localhost:5001')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('training_input')
    parser.add_argument('training_output')
    args = parser.parse_args()

    X = np.load(args.training_input)
    y = np.load(args.training_output)
    num_examples = X.shape[0]

    layer1 = sgd.Layer(2, 24, sgd.Sigmoid())

    front_client = synthgrad.LayerClient(args.output_server)
    oracle_client = synthgrad.OracleClient(args.oracle_server)

    batch = []
    for i in range(args.num_epochs):
        print 'Epoch %d...' % (i + 1,)
        start = time.time()
        for j in range(num_examples):
            if (j + 1) % 5000 == 0:
                sys.stdout.write('*')
                sys.stdout.flush()
            x = X[j]
            expected = y[j]
            h1 = layer1.feed_forward(x)

            batch.append((j, h1, expected))
            if len(batch) > 100:
                front_client.provide_training_examples(batch)
                batch = []

            # Block until we get a gradient.
            gradient = oracle_client.estimate_gradient(h1)
            layer1.backprop(gradient)
            layer1.descend(args.learning_rate)
        elapsed = time.time() - start
        print ' complete (%.1f seconds).'


if __name__ == '__main__':
    main()
