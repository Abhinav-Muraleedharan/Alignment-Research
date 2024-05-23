"""

Sample Code to study linearization
in jax on toy networks

Note: Linearization is trivial for small networks.
We just have to call jax.linearize() and extract 
the jacobian vector product and output value of 
network. 

"""
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp


# function to randomly initialize weights and biases.
def random_layer_params(m,n,key,scale=1e-2):
    w_key, b_key = random.split(key)
    return scale*random.normal(w_key,(n,m)) , scale*random.normal(b_key,(n,))

def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(x):
  return jnp.maximum(0, x)


def network(params, x):
  # per-example predictions
  activations = x
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)
  
  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)


if __name__ == '__main__':

    layer_sizes = [784, 512, 512, 10]
    step_size = 0.01
    num_epochs = 10
    batch_size = 128
    n_targets = 10
    params = init_network_params(layer_sizes, random.key(0))
    x = random.normal(random.key(1), (28 * 28,))
    output = network(params,x)
    print(output)
    y, f_jvp = jax.linearize(network,params,x)
    print(y)
    x_bar = jnp.ones((28*28,))
    print(f_jvp(params,x_bar))
   