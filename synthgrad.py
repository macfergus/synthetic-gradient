import json

import numpy as np
import requests


def ndarray_to_json(ndarray):
    return {
        'shape': ndarray.shape,
        'values': list(ndarray.flatten()),
    }


def json_to_ndarray(ndarray_as_json):
    return np.array(ndarray_as_json['values']).\
        reshape(tuple(ndarray_as_json['shape']))


class LayerClient(object):
    def __init__(self, base_url):
        self.base_url = base_url

    def provide_training_example(self, x, y):
        payload = {
            'x': ndarray_to_json(x),
            'y': ndarray_to_json(y),
        }
        response = requests.post(self.base_url + '/training_example', json=payload)
        return json_to_ndarray(response.json()['gradient'])


class OracleClient(object):
    def __init__(self, base_url):
        self.base_url = base_url

    def estimate_gradient(self, activation):
        payload = {
            'h': ndarray_to_json(activation),
        }
        response = requests.post(
            self.base_url + '/estimate_gradient', json=payload)
        gradient = json_to_ndarray(response.json()['gradient'])
        return gradient
