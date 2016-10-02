import json
import logging

import numpy as np
import requests



def ndarray_to_json(ndarray):
    return {
        'shape': ndarray.shape,
        'values': [float(x) for x in ndarray.flatten()],
    }


def json_to_ndarray(ndarray_as_json):
    return np.array(ndarray_as_json['values']).\
        reshape(tuple(ndarray_as_json['shape']))


class LayerClient(object):
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        # Disable proxies to circumvent a bug with multiprocessing on
        # Mac OS.
        self.session.trust_env = False

    def provide_training_examples(self, batch):
        payload = [{
            'i': i,
            'x': ndarray_to_json(x),
            'y': ndarray_to_json(y),
        } for i, x, y in batch]
        self.session.post(self.base_url + '/training_example', json=payload)


class OracleClient(object):
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        # Disable proxies to circumvent a bug with multiprocessing on
        # Mac OS.
        self.session.trust_env = False

    def estimate_gradient(self, activation):
        payload = {
            'h': ndarray_to_json(activation),
        }
        response = self.session.post(
            self.base_url + '/estimate_gradient', json=payload)
        gradient = json_to_ndarray(response.json()['gradient'])
        return gradient

    def provide_gradients(self, batch):
        payload = [
            {
                'i': i,
                'h': ndarray_to_json(activation),
                'gradient': ndarray_to_json(gradient),
            } for i, activation, gradient in batch]
        self.session.post(self.base_url + '/provide_gradient', json=payload)
