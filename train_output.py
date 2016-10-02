import logging
import Queue
import threading

import numpy as np
from flask import Flask, jsonify, request

import sgd
import synthgrad

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

STOP = object()

class ErrorCollector(object):
    def __init__(self):
        self.reset()

    def record_loss(self, loss_contribution):
        self.sum_squares += loss_contribution
        self.num_examples += 1

    def mse(self):
        return self.sum_squares / float(self.num_examples)

    def reset(self):
        self.sum_squares = 0
        self.num_examples = 0

    @property
    def count(self):
        return self.num_examples


class TrainingThread(threading.Thread):
    def __init__(self, queue):
        super(TrainingThread, self).__init__()
        self.q = queue
        self.error_collector = ErrorCollector()
        self.layer2 = sgd.Layer(24, 12, sgd.Sigmoid())
        self.layer3 = sgd.Layer(12, 1, sgd.Linear())
        self.learning_rate = 0.001


    def run(self):
        while True:
            pair = self.q.get()
            if pair is STOP:
                return
            x, y = pair
            self.train(x, y)

    def train(self, x, y):
        h2 = self.layer2.feed_forward(x)
        h3 = self.layer3.feed_forward(h2)
        output = h3[0]

        # Find the error.
        delta = output - y
        loss_contribution = (delta * delta) / 2.
        loss_derivative = delta

        self.error_collector.record_loss(loss_contribution)

        partials3 = self.layer3.backprop(np.array([loss_derivative]))
        partials2 = self.layer2.backprop(partials3)
        self.layer3.descend(self.learning_rate)
        self.layer2.descend(self.learning_rate)

        if self.error_collector.count > 1000:
            print 'Average loss over last 1000 examples: %.6f' % (
                self.error_collector.mse(),)
            self.error_collector.reset()


training_queue = Queue.Queue()


@app.route('/training_example', methods=['POST'])
def training_example():
    payload = request.json
    x = synthgrad.json_to_ndarray(payload['x'])
    y = synthgrad.json_to_ndarray(payload['y'])

    training_queue.put((x, y))

    return jsonify(ok=True)


if __name__ == '__main__':
    training_thread = TrainingThread(training_queue)
    training_thread.start()
    try:
        app.run('localhost', 5000, debug=True)
    finally:
        training_queue.put(STOP)
        training_thread.join()
