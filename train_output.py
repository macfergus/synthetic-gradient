import logging

import numpy as np
from flask import Flask, jsonify, request

import sgd
import synthgrad

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

layer2 = sgd.Layer(24, 12, sgd.Sigmoid())
layer3 = sgd.Layer(12, 1, sgd.Linear())

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


error_collector = ErrorCollector()


@app.route('/training_example', methods=['POST'])
def training_example():
    payload = request.json
    x = synthgrad.json_to_ndarray(payload['x'])
    y = synthgrad.json_to_ndarray(payload['y'])

    h2 = layer2.feed_forward(x)
    h3 = layer3.feed_forward(h2)
    output = h3[0]

    # Find the error.
    delta = output - y
    loss_contribution = (delta * delta) / 2.
    loss_derivative = delta

    error_collector.record_loss(loss_derivative)

    partials3 = layer3.backprop(np.array([loss_derivative]))
    partials2 = layer2.backprop(partials3)

    if error_collector.count > 1000:
        print 'Average loss over last 1000 examples: %.6f' % (
            error_collector.mse(),)
        error_collector.reset()

    return jsonify(gradient=synthgrad.ndarray_to_json(partials2))


if __name__ == '__main__':
    app.run('localhost', 5000, debug=True)
