import logging
import Queue
import os
import random
import threading
import time
from multiprocessing import Process
from multiprocessing import Queue as PQueue

import numpy as np
from flask import Flask, current_app, jsonify, request

import sgd
import synthgrad

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def save_weights(layer2, layer3, ts):
    template = os.path.join('progress/%d_%s_%s.npy')
    np.save(template % (ts, 'l2', 'w'), layer2.w)
    np.save(template % (ts, 'l2', 'b'), layer2.b)
    np.save(template % (ts, 'l3', 'w'), layer3.w)
    np.save(template % (ts, 'l3', 'b'), layer3.b)


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
            batch = {i: (i, h, grad) for i, h, grad in batch}.values()
            if len(batch) > 5000:
                print "Reducing batch from %d to 5000" % (len(batch),)
                batch = random.sample(batch, 5000)
            while batch:
                # Send in 500-example chunks.
                cur, batch = batch[:500], batch[500:]
                try:
                    self.client.provide_gradients(cur)
                except Exception:
                    print "Exception sending chunk; discarding."


def train_forever(q):
    layer2 = sgd.Layer(24, 12, sgd.Sigmoid())
    layer3 = sgd.Layer(12, 1, sgd.Linear())
    learning_rate = 0.001
    oracle_client = synthgrad.OracleClient('http://127.0.0.1:5001')

    send_q = Queue.Queue()
    sender = Sender(send_q, oracle_client)
    sender.start()

    examples = {}

    while True:
        while not q.empty():
            try:
                item = q.get(block=False)
            except Queue.Empty:
                break
            if item is None:
                send_q.put(None)
                sender.join()
                return
            i, x, y = item
            examples[i] = (x, y)
        if len(examples) == 0:
            time.sleep(1)
            continue

        indices, X, y = [], [], []
        for i, (h, grad) in examples.items():
            indices.append(i)
            X.append(h)
            y.append(grad)
        X = np.array(X)
        y = np.array(y)

        num_examples = X.shape[0]
        print "Training on %d examples..." % (num_examples,)
        sum_squares = 0.0
        start = time.time()
        batch = []
        for j in range(num_examples):
            index = indices[j]
            x = X[j]
            target = y[j]

            h2 = layer2.feed_forward(x)
            h3 = layer3.feed_forward(h2)
            output = h3[0]

            # Find the error.
            delta = output - target
            loss_contribution = (delta * delta) / 2.
            loss_derivative = delta
            sum_squares += loss_contribution

            partials3 = layer3.backprop(np.array([loss_derivative]))
            partials2 = layer2.backprop(partials3)
            layer3.descend(learning_rate)
            layer2.descend(learning_rate)

            send_q.put((index, x, partials2))

            now = int(time.time())
            if now % 100 == 0:
                save_weights(layer2, layer3, now)
        mse = sum_squares / float(num_examples)
        elapsed = time.time() - start
        print "MSE %.06f; took %.1f seconds" % (mse, elapsed)
        # Give the worker thread a chance to do stuff.
        time.sleep(0.05)


@app.route('/training_example', methods=['POST'])
def training_example():
    payload = request.json
    for example in payload:
        x = synthgrad.json_to_ndarray(example['x'])
        y = synthgrad.json_to_ndarray(example['y'])

        current_app.q.put((example['i'], x, y))

    return jsonify(ok=True)


if __name__ == '__main__':
    training_queue = PQueue()
    with app.app_context():
        current_app.q = training_queue

    trainer_proc = Process(target=train_forever, args=(training_queue,))
    trainer_proc.start()
    try:
        app.run('localhost', 5000, debug=False)
    finally:
        training_queue.put(None)
        trainer_proc.join()
