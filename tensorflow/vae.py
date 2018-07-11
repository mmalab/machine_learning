"""Vanilla Autoencoder"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.distributions as  tfd
from tensorflow.examples.tutorials.mnist import input_data


def _summary(name, tensor):
  shape = tensor.shape.as_list()
  if tensor.shape.ndims == 0:
    return tf.summary.scalar(name, tensor)
  elif shape == [10, 28, 28]:
    return tf.summary.image(
      name, tf.reshape(tensor, shape + [1]), 10)
  else:
    return tf.summary.histogram(name, tensor)


def _prod(shape):
  return functools.reduce(operator.mul, shape)


def _make_encoder(data, code_size):
  with tf.variable_scope('encoder'):
    x = tf.layers.flatten(data)
    x = tf.layers.dense(x, 200, tf.nn.relu)
    x = tf.layers.dense(x, 200, tf.nn.relu)
    loc = tf.layers.dense(x, code_size)
    scale = tf.layers.dense(x, code_size, tf.nn.softplus)
    return tfd.MultivariateNormalDiag(loc, scale)


def make_prior(code_size):
  with tf.variable_scope('prior'):
    loc = tf.zeros(code_size)
    scale = tf.ones(code_size)
    return tfd.MultivariateNormalDiag(loc, scale)


def _make_decoder(code, data_shape):
  with tf.variable_scope('decoder'):
    x = code
    x = tf.layers.dense(x, 200, tf.nn.relu)
    x = tf.layers.dense(x, 200, tf.nn.relu)
    logit = tf.layers.dense(x, _prod(data_shape))
    logit = tf.reshape(logit, [-1] + data_shape)
    return tfd.Independent(tfd.Bernoulli(logit), 2)


def plot_codes(ax, codes, labels):
  ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
  ax.set_aspect('equal')
  ax.set_xlim(codes.min() - .1, codes.max() + .1)
  ax.set_ylim(codes.min() - .1, codes.max() + .1)
  ax.tick_params(
    axis='both', which='both', left='off', bottom='off',
    labelleft='off', labelbottom='off')


def plot_samples(ax, samples):
  for index, sample in enumerate(samples):
    ax[index].imshow(sample, cmap='gray')
    ax[index].axis('off')


def main(_):
  epoch_size = 20
  logdir = './logdir'
  
  make_encoder = tf.make_template('encoder', _make_encoder)
  # In TensorFlow, if you call a network function twice,
  # it will create two separate networks.
  # TensorFlow templates allow you to wrap a function
  # so that multiple calls to it will reuse the same network parameters.
  make_decoder = tf.make_template('decoder', _make_decoder)
  
  data = tf.placeholder(tf.float32, [None, 28, 28])
  
  prior = make_prior(code_size=2)
  posterior = make_encoder(data, code_size=2)
  code = posterior.sample()
  
  likelihood = make_decoder(code, [28, 28]).log_prob(data)
  divergence = tfd.kl_divergence(posterior, prior)
  elbo = tf.reduce_mean(likelihood - divergence)
  
  optimizer = tf.train.AdamOptimizer(0.001).minimize(-elbo)
  samples = make_decoder(prior.sample(10), [28, 28]).mean()
  
  mnist = input_data.read_data_sets('/tmp/MNIST_data/')
  fig, ax = plt.subplots(nrows=epoch_size, ncols=11, figsize=(10, 20))
  
  # Merged all summaries.
  _summary('likelihood', likelihood)
  _summary('divergence', divergence)
  _summary('elbo', elbo)
  _summary('samples', samples)
  merged = tf.summary.merge_all()
  saver = tf.train.Saver()
  
  _global_step = tf.get_variable('global_step', [], dtype=tf.int32, trainable=False)
  global_step_op = tf.assign_add(_global_step, 1)
  with tf.train.MonitoredSession() as sess:
    writer = tf.summary.FileWriter(logdir, sess.graph)
    for epoch in range(epoch_size):
      feed = {data: mnist.test.images.reshape([-1, 28, 28])}
      test_elbo, test_codes, test_samples = sess.run(
        [elbo, code, samples], feed)
      
      test_likelihood, test_divergence =  sess.run([likelihood, divergence], feed)
      print('likeli {}, ')
      
      
      # Plot codes and samples
      ax[epoch, 0].set_ylabel('Epoch {}'.format(epoch))
      plot_codes(ax[epoch, 0], test_codes, mnist.test.labels)
      plot_samples(ax[epoch, 1:], test_samples)
      print('\rEpoch {}, elbo {}, labes {}, test_codes {}, test_samples {}'.format(
        epoch, test_elbo, mnist.test.labels.shape, test_codes.shape, test_samples.shape),
        end='', flush=True)
      
      for step in range(1, 600):
        feed = {data: mnist.train.next_batch(100)[0].reshape([-1, 28, 28])}
        _, summary, global_step = sess.run([optimizer, merged, global_step_op], feed)
        writer.add_summary(summary, global_step=global_step)


plt.show()

if __name__ == '__main__':
  tf.app.run()