"""Vanilla Variational Auto Encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import matplotlib.pyplot as plt
import torch
import torch.distributions as D
import torch.distributions.kl as kl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorflow.contrib.learn.python.learn.datasets import mnist as input_data
from tqdm import trange


def select_device(gpu_id):
  if gpu_id >= 0:
    DEVICE = torch.device('cuda:{}'.format(gpu_id))
  else:
    DEVICE = torch.device('cpu:0')
  return DEVICE


DEVICE = select_device(-1)
print('DEVICE {}'.format(DEVICE))


def _prob(shape):
  return functools.reduce(operator.mul, shape)


def tensor(x):
  if isinstance(x, torch.Tensor):
    return x
  x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
  return x


class Encoder(nn.Module):
  
  def __init__(self, data_shape, code_size):
    self.code_size = code_size
    self.data_shape = _prob(data_shape)
    super(Encoder, self).__init__()
    self._fc1 = nn.Linear(self.data_shape, 200)
    self._fc2 = nn.Linear(200, 200)
    self._loc_fc = nn.Linear(200, code_size)
    self._scale_fc = nn.Linear(200, code_size)
    self.to(DEVICE)
  
  def forward(self, inputs):
    x = tensor(inputs)
    x = x.view([-1] + [self.data_shape])
    x = F.relu(self._fc1(x))
    x = F.relu(self._fc2(x))
    loc = self._loc_fc(x)
    scale = F.softplus(self._scale_fc(x))
    return D.MultivariateNormal(loc, scale_tril=torch.diag(torch.diag(scale)))


class Decoder(nn.Module):
  
  def __init__(self, code_size, data_shape):
    self.code_size = code_size
    self.data_shape = data_shape
    super(Decoder, self).__init__()
    self._fc1 = nn.Linear(code_size, 200)
    self._fc2 = nn.Linear(200, 200)
    self._fc3 = nn.Linear(200, _prob(data_shape))
    self.to(DEVICE)
  
  def forward(self, inputs):
    code = tensor(inputs)
    x = F.relu(self._fc1(code))
    x = F.relu(self._fc2(x))
    logits = self._fc3(x)
    logits = logits.view([-1] + self.data_shape)
    return D.Independent(D.Bernoulli(logits=logits), 2)


def make_prior(code_size):
  loc = torch.zeros(code_size)
  scale = torch.ones(code_size)
  return kl.MultivariateNormal(loc, torch.diag(scale))


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


epoch_size = 20
t = trange(epoch_size, desc='ELBO', leave=True)
code_size = 2
logdir = './logdir'

mnist = input_data.read_data_sets('/tmp/MNIST_data')
data_shape = [28, 28]
prior = make_prior(code_size=code_size)
encoder = Encoder(data_shape, code_size=code_size)
decoder = Decoder(code_size, [28, 28])
# fig, ax = plt.subplots(nrows=epoch_size, ncols=11, figsize=(10, 20))


def _forward(data):
  posterior = encoder(data)
  code = posterior.sample()
  samples = decoder(prior.sample((10,)))
  likelihood = decoder(code).log_prob(tensor(data))
  divergence = D.kl_divergence(posterior, prior)
  elbo = torch.mean(likelihood - divergence)
  return elbo, code.cpu().detach().numpy(), samples.mean.cpu().detach().numpy()


optimizer = optim.Adam(list(decoder.parameters()) + list(encoder.parameters()),
                       lr=0.001)


def main():
  for epoch in t:
    # Test
    data = mnist.test.images.reshape([-1, 28, 28])
    elbo, test_codes, test_samples = _forward(data)
    t.set_description('ELBO {0:.3f}'.format(elbo))
    t.refresh()
    
    # ax[int(epoch), 0].set_ylabel('Epoch {}'.format(epoch))
    # plot_codes(ax[int(epoch), 0], test_codes, mnist.test.labels)
    # plot_samples(ax[int(epoch), 1:], test_samples)
    
    # Train
    for step in range(1, 600):
      data = mnist.train.next_batch(100)[0].reshape([-1, 28, 28])
      elbo, _, _ = _forward(data)
      loss = -elbo
      optimizer.zero_grad()  # zero the gradient buffers
      loss.backward(retain_graph=True)
      optimizer.step()


# plt.savefig('vae_mnist2.png', dpi=300, transparent=True, bbox_inches='tight')
# plt.show()

if __name__ == '__main__':
  main()