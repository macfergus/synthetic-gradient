import logging
import thread
import time
from multiprocessing import Process, Queue

import numpy as np
from flask import Flask, current_app, jsonify, request

import sgd
import synthgrad

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def train_forever(q):
    layer2 = sgd.Layer(24, 12, sgd.Sigmoid())
    layer3 = sgd.Layer(12, 1, sgd.Linear())
    learning_rate = 0.001
    oracle_client = synthgrad.OracleClient('http://127.0.0.1:5001')

    examples = {}

    while True:
        while not q.empty():
            try:
                item = q.get(block=False)
            except Queue.Empty:
                break
            if item is None:
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

            oracle_client.provide_gradient(index, x, partials2)
        mse = sum_squares / float(num_examples)
        elapsed = time.time() - start
        print "MSE %.06f; took %.1f seconds" % (mse, elapsed)


@app.route('/training_example', methods=['POST'])
def training_example():
    payload = request.json
    x = synthgrad.json_to_ndarray(payload['x'])
    y = synthgrad.json_to_ndarray(payload['y'])

    current_app.q.put((payload['i'], x, y))

    return jsonify(ok=True)


if __name__ == '__main__':
    training_queue = Queue()
    with app.app_context():
        current_app.q = training_queue

    trainer_proc = Process(target=train_forever, args=(training_queue,))
    trainer_proc.start()
    try:
        app.run('localhost', 5000, debug=False)
    finally:
        training_queue.put(None)
        trainer_proc.join()
