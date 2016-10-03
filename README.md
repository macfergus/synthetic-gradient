# synthetic-gradient
Reference implementation of decoupled training with synthetic gradients.

This is my attempt at implementing the algorithm described in "Decoupled Neural Interfaces using Synthetic Gradients" (https://arxiv.org/abs/1608.05343). Or at least, something inspired by that algorithm. (To be clear, I have absolutely no affiliation with Google DeepMind or any of the authors of that paper.)

I implemented this for our reading group at work.

## Contents

`generate.py` generates training data for the function we are trying to learn. The target function has no significance at all; it's just a lumpy function on R^2 -> R.

`sgd.py` learns the reference function with old-fashioned gradient descent. The purpose of this code is to illustrate SGD and backprop for people who are unfamiliar. It's implemented with plain python + numpy, and it sacrifices performance for readability.

`train_input.py`, `train_output.py`, and `oracle.py` learn the reference function asynchronously with the synthetic gradient technique (more detail below).

`synthgrad.py` contains some common code for the interface between the three processes.

`plot.py` plots intermediate training results from the decoupled training process.

## Notes on decoupled training

Standard SGD is perfectly fine on the example problem; in fact, the decoupled training process takes at least 10x longer. The ideas makes more sense on much deeper networks. That said, it's very cool conceptually.

We are training a 3-layer network. I've split it up into two parts. `train_input.py` learns the weights for the first layer, and `train_output.py` learns the weights for the second and third layers.

`train_input.py` loads the training set and starts cranking through it. After doing the feed-forward pass, it tosses its activations over to the `train_output.py` process in a fire-and-forget fashion. Meanwhile, it needs gradients in order to update its own weights. It consults the "oracle", which will synchronously provide its best guess at what the gradients would be for those activations. Then `train_input.py` can complete the backprop and update its weights.

Meanwhile, `train_output.py` always has only a partial set; it only gets activations as fast as `train_input.py` can generate them. It continuously trains over whatever training set it currently has in a loop. As it generates true gradients, it passes them over to the oracle so that the oracle can improve its own predictions, again in a fire-and-forget fashion.

`oracle.py` is very similar to `train_output.py`: it concurrently collects new training example and re-trains over its current examples. Just for giggles, I implemented its predictions using Keras instead of my home-grown SGD code. I cheated a little bit in that the oracle trains a fairly high-capacity model. The paper says they had success where an oracle model has much lower capacity than the true model it approximated, but this implementation does not test that claim.

The three processes communicate over a simple HTTP protocol. Hopefully the code in `synthgrad.py` describes it adequately.

Unfortunately, the implementation is not as clear as the plain SGD implementation. The most naive implementation was impossibly slow, and I had to add a lot of extra code to make sure things were parallelized efficiently. Even so, it's still many times slower than the traditional implementation. With a model this small, the overhead of a network request is slower than just pushing the numbers through the rest of the network. Nevertheless, it's still very cool to see the two half-models learn at their own pace in separate processes-- and kinda mind-blowing that this works at all.

If you want to run this, start `oracle.py` first, then `train_output.py`, then `train_input.py`.
