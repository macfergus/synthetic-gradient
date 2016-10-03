import argparse
import os
import Queue
import random
import sys
import threading
import time

import sgd
import synthgrad

import numpy as np


def save_weights(layer1, ts):
    template = os.path.join('progress/%d_%s_%s.npy')
    np.save(template % (ts, 'l1', 'w'), layer1.w)
    np.save(template % (ts, 'l1', 'b'), layer1.b)


class Sender(threading.Thread):
    def __init__(self, q, client):
        super(Sender, self).__init__()
        self.q = q
        self.client = client

    def run(self):
        while True:
            batch = []
            while not self.q.empty():
                try:
                    item = self.q.get(block=False)
                    if item is None:
                        return
                    batch.append(item)
                except Queue.Empty:
                    break
            if not batch:
                time.sleep(1)
                continue
            # Dedupe results. We may have computed the same example
            # multiple times.
            batch = {i: (i, x, y) for i, x, y in batch}.values()
            if len(batch) > 2000:
                batch = random.sample(batch, 2000)
            while batch:
                # Send in 500-example chunks.
                cur, batch = batch[:500], batch[500:]
                try:
                    self.client.provide_training_examples(cur)
                except Exception:
                    print "Exception sending chunk; discarding."


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

    send_q = Queue.Queue()
    sender = Sender(send_q, front_client)
    sender.start()

    minibatch_size = 10

    try:
        for i in range(args.num_epochs):
            print 'Epoch %d...' % (i + 1,)
            start = time.time()
            minibatch = []
            for j in range(num_examples):
                if (j + 1) % 5000 == 0:
                    sys.stdout.write('*')
                    sys.stdout.flush()
                x = X[j]
                expected = y[j]
                h1 = layer1.feed_forward(x)

                # Hand the example off to the other half of the network.
                send_q.put((j, h1, expected))

                minibatch.append(h1)

                if len(minibatch) == minibatch_size:
                    gradients = oracle_client.estimate_gradients(minibatch)
                    # Average the gradients over our minibatch.
                    gradient = np.sum(gradients, axis=0)
                    gradient /= float(minibatch_size)
                    layer1.backprop(gradient)
                    layer1.descend(args.learning_rate)
                    minibatch = []

                    now = int(time.time())
                    if now % 100 == 0:
                        save_weights(layer1, now)
            elapsed = time.time() - start
            print ' complete (%.1f seconds).' % (elapsed,)
    finally:
        send_q.put(None)
        sender.join()


if __name__ == '__main__':
    main()
