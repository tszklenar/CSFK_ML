#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print('')
print('==================================================================================')
print('=============================== Importing packages ===============================')
print('==================================================================================')
print('')
import arviz
import astropy
import astroquery
import autograd
import bs4
import celerite
import corner
try:
  import cuvarbase
except ModuleNotFoundError as e:
  print(e,'Installion guide: https://johnh2o2.github.io/cuvarbase/install.html')
import exoplanet
try:
    import FATS
except SyntaxError:
    print('FATS only works with python2!')
import h5py
import html5lib
import ipykernel
from tensorflow import keras
import lightkurve
import llvmlite
import lmfit
import matplotlib.pyplot as plt
import nfft
import numba
import numpy as np
import pandas
import photutils
import PIL
import psutil
import PyAstronomy
import pycuda
import pymc3
import sklearn
import skcuda
import scipy
import seaborn
import six
import tensorboard
import tensorflow as tf
import theano
import torch
import torchsummary
import torchvision
import tqdm
import uncertainties
import unittest
import urllib3
import wotan

print('=============================== Package importing passed =========================')


# matplotlib test
def plot_square(x, y):
  y_squared = np.square(y)
  return plt.plot(x, y_squared)

def test_plot_square1():
  x, y = [0, 1, 2], [0, 1, 2]
  line, = plot_square(x, y)
  x_plot, y_plot = line.get_xydata().T
  np.testing.assert_array_equal(y_plot, np.square(y))

print('')
print('==================================================================================')
print('=============================== Matplotlib test  =================================')
print('==================================================================================')
print('')
test_plot_square1()
print('=============================== Matplotlib test passed ===========================')

print('')
print('==================================================================================')
print('=============================== NumPy test  ======================================')
print('==================================================================================')
print('')
np.test()

print('')
print('==================================================================================')
print('=============================== SciPy test  ======================================')
print('==================================================================================')
print('')
scipy.test()

print('')
print('==================================================================================')
print('=============================== Looking for GPUs  ================================')
print('==================================================================================')
print('')
assert tf.config.list_physical_devices('GPU'),'No GPU available!'
print('=============================== GPU(s) found: ====================================')
for gpu in tf.config.list_physical_devices('GPU'):
    print(gpu)


def get_entry_np(t, indices_d1, indices_d2, batch_size):
  result = np.zeros(batch_size)
  for i in range(batch_size):
    result[i] = t[i, indices_d1[i], indices_d2[i]]
  return result

def get_entry_tf(t, indices_d1, indices_d2, batch_size):
  indices = tf.stack([tf.range(batch_size), indices_d1, indices_d2], axis=1)
  return tf.gather_nd(t, indices)

class Test(unittest.TestCase):
  def test_get_entry(self):
    success = True
    for _ in range(10):
      # sample input
      batch_size, d1, d2 = map(int, np.random.randint(low=2, high=100, size=3))
      test_input = np.random.random([batch_size, d1, d2])
      test_indices_d1 = np.random.randint(low=0, high=d1-1, size=[batch_size])
      test_indices_d2 = np.random.randint(low=0, high=d2-1, size=[batch_size])
      # evaluate the numpy version
      test_result = get_entry_np(test_input, test_indices_d1, test_indices_d2, batch_size)
      # evaluate the tensorflow version
      with tf.compat.v1.Session() as sess:
        tf_input = tf.constant(test_input, dtype=tf.float32)
        tf_indices_d1 = tf.constant(test_indices_d1, dtype=tf.int32)
        tf_indices_d2 = tf.constant(test_indices_d2, dtype=tf.int32)
        tf_result = get_entry_tf(tf_input, tf_indices_d1, tf_indices_d2, batch_size)
        tf_result = sess.run(tf_result)
        # check that outputs are similar
        success = success and np.allclose(test_result, tf_result)

    self.assertEqual(success, True,'Tensorflow tensor not equal to numpy array!')

print('')
print('==================================================================================')
print('=============================== Tensorflow test  =================================')
print('==================================================================================')
print('')
Test().test_get_entry()
print('=============================== Tensorflow test passed ===========================')


#cuvarbase test

from cuvarbase.utils import weights
from cuvarbase.pdm import pdm2_cpu,PDMAsyncProcess


def data(seed=100, sigma=0.1, ndata=250):

    rand = np.random.RandomState(seed)

    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * (10./(max(t) - min(t))) * t)

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, y, err


def test_cuda_pdm():

    kind = 'binned_linterp'
    nbins = 20
    seed = 100
    nfreqs = 100
    ndata = 250

    t, y, err = data(seed=seed, ndata=ndata)

    w = weights(err)
    freqs = np.linspace(0, 100./(max(t) - min(t)), nfreqs)
    freqs += 0.5 * (freqs[1] - freqs[0])

    pow_cpu = pdm2_cpu(t, y, w, freqs,
                       linterp=(kind == 'binned_linterp'),
                       nbins=nbins)

    pdm_proc = PDMAsyncProcess()
    results = pdm_proc.run([(t, y, w, freqs)], kind=kind, nbins=nbins)
    pdm_proc.finish()

    pow_gpu = results[0]

    np.testing.assert_allclose(pow_cpu, pow_gpu, atol=1E-2, rtol=0)

print('')
print('==================================================================================')
print('=============================== CUDA PDM test  ===================================')
print('==================================================================================')
print('')
test_cuda_pdm()
print('=============================== CUDA PDM test passed =============================')

print('')
print('==================================================================================')
print('=================================== Done =========================================')
print('==================================================================================')
