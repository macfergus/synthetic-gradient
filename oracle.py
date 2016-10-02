import numpy as np
from flask import Flask, jsonify, request
from keras.models import Sequential
from keras.layers.core import Dense

import synthgrad

app = Flask(__name__)

DIM = 24


class GradientModel(object):
    def __init__(self):
        self._model = Sequential()
        self._model.add(Dense(12, input_dim=DIM, activation='sigmoid'))
        self._model.add(Dense(DIM, activation='linear'))
        self._model.compile(loss='mean_squared_error', optimizer='adadelta')

    def estimate_gradient(self, activation):
        y = self._model.predict(np.array([activation]))
        return y[0]

    def add_training_example(self, activation, gradient):
        X = np.array([activation])
        y = np.array([gradient])
        self._model.fit(X, y, nb_epoch=1)


model = GradientModel()


@app.route('/estimate_gradient', methods=['POST'])
def estimate_gradient():
    payload = request.json
    activation = synthgrad.json_to_ndarray(payload['h'])
    gradient = model.estimate_gradient(activation)
    return jsonify(gradient=synthgrad.ndarray_to_json(gradient))


@app.route('/provide_gradient', methods=['POST'])
def provide_gradient():
    payload = request.json
    activation = synthgrad.json_to_ndarray(payload['h'])
    gradient = synthgrad.json_to_ndarray(payload['gradient'])
    model.add_training_example(activation, gradient)
    return jsonify(ok=True)


if __name__ == '__main__':
    app.run('localhost', 5001, debug=True)
