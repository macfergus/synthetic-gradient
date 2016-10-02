import logging
import time
from multiprocessing import Manager, Process, Queue

import numpy as np
from flask import Flask, jsonify, request
from keras.models import Sequential
from keras.layers.core import Dense

import synthgrad

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

DIM = 24

predict_model = Sequential()
predict_model.add(Dense(12, input_dim=DIM, activation='sigmoid'))
predict_model.add(Dense(DIM, activation='linear'))
predict_model.compile(loss='mean_squared_error', optimizer='adadelta')
shared_weights = None


def train_forever(q):
    global shared_weights

    model = Sequential()
    model.add(Dense(12, input_dim=DIM, activation='sigmoid'))
    model.add(Dense(DIM, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adadelta')

    examples = {}

    while True:
        # Collect training set from queue.
        while not q.empty():
            try:
                item = q.get(block=False)
            except Queue.Empty:
                break
            if item is None:
                # Stop sentinel.
                return
            i, h, grad = item
            examples[i] = (h, grad)
        if not examples:
            # Sleep, then try to get more.
            time.sleep(1)
            continue
        # Train.
        X, y = [], []
        for h, grad in examples.values():
            X.append(h)
            y.append(grad)
        X = np.array(X)
        y = np.array(y)
        model.fit(X, y, nb_epoch=2)
        # Pass weights back to server process.
        shared_weights[:] = model.get_weights()


@app.route('/estimate_gradient', methods=['POST'])
def estimate_gradient():
    payload = request.json
    activation = synthgrad.json_to_ndarray(payload['h'])
    if shared_weights:
        predict_model.set_weights(shared_weights)
    predictions = predict_model.predict(np.array([activation]))
    gradient = predictions[0]
    return jsonify(gradient=synthgrad.ndarray_to_json(gradient))


@app.route('/provide_gradient', methods=['POST'])
def provide_gradient():
    payload = request.json
    for example in payload:
        activation = synthgrad.json_to_ndarray(example['h'])
        gradient = synthgrad.json_to_ndarray(example['gradient'])
        training_queue.put((example['i'], activation, gradient))
    return jsonify(ok=True)


if __name__ == '__main__':
    manager = Manager()
    shared_weights = manager.list()

    training_queue = Queue()
    trainer_proc = Process(target=train_forever, args=(training_queue,))
    trainer_proc.start()
    try:
        app.run('localhost', 5001, debug=False)
    finally:
        training_queue.put(None)
        trainer_proc.join()
