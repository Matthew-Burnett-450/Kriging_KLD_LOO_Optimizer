from typing import Any
import numpy as np
import time
import tensorflow as tf
from numba import jit,vectorize,float32
import scipy.linalg
from scipy.spatial import distance

def run_on_gpu():
    with tf.device('/gpu:0'):
        # Your TensorFlow code for GPU execution
        print("Running on GPU")

def run_on_cpu():
    with tf.device('/cpu:0'):
        # Your TensorFlow code for CPU execution
        print("Running on CPU")

try:
    # Attempt to set the device to the first available GPU
    tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    run_on_gpu()
except Exception as e:
    print("Error occurred while trying to run on GPU: ", e)
    # If an exception occurs, switch to CPU
    run_on_cpu()

class __VariogramWrapper:
    def __init__(self, a=0, C=0):
        self.a = tf.constant(a, dtype=tf.float16)
        self.C = tf.constant(C, dtype=tf.float16)
    def set_a_C(self, a, C):
        self.a = tf.constant(a, dtype=tf.float16)
        self.C = tf.constant(C, dtype=tf.float16)
    def __call__(self, h):
        pass

# Spherical Variogram
class SphericalVariogram(__VariogramWrapper):
    def spherical_variogram(self, h):
        h = tf.constant(h, dtype=tf.float16)
        output = self.C * (1 - ((3 / 2) * (h / self.a) - (1 / 2) * (h / self.a)**3))
        return output.numpy()

    def __call__(self, h):
        return self.spherical_variogram(h)

# Exponential Variogram
class ExponentialVariogram(__VariogramWrapper):
    def exponential_variogram(self, h):
        h = tf.constant(h, dtype=tf.float16)
        output = self.C * (1 - tf.exp(-h / self.a))
        return output.numpy()

    def __call__(self, h):
        return self.exponential_variogram(h)

# Linear Variogram
class LinearVariogram(__VariogramWrapper):
    def linear_variogram(self, h):
        h = tf.constant(h, dtype=tf.float16)
        output = self.C * (h / self.a)
        return output.numpy()

    def __call__(self, h):
        return self.linear_variogram(h)

# Power Variogram
class PowerVariogram(__VariogramWrapper):
    def power_variogram(self, h):
        h = tf.constant(h, dtype=tf.float16)
        output = self.C * (h / self.a)**2
        return output.numpy()

    def __call__(self, h):
        return self.power_variogram(h)
# Gaussian Variogram
class GaussianVariogram(__VariogramWrapper):
    def gaussian_variogram(self, h):
        h = tf.constant(h, dtype=tf.float16)
        output = self.C * (1 - tf.exp(-((h / self.a)**2)))
        return output.numpy()

    def __call__(self, h):
        return self.gaussian_variogram(h)


def numba_dist_matrix(points):
    return distance.cdist(points, points, 'euclidean')

def numba_distances_to_point0(points, Xo, Yo):
    single_point = np.array([[Xo, Yo]])
    
    # Compute distances from a single point to all other points using cdist
    distances_to_point0 = distance.cdist(single_point, points, 'euclidean').flatten()
    
    return distances_to_point0




   
n=10000








@jit(fastmath=True, forceobj=True)
def expand_dist_matrix_with_point(dmat, points, Xo, Yo):
    num_atoms = dmat.shape[0]
    
    # Pre-allocate memory for new_dmat
    new_dmat = np.empty((num_atoms + 1, num_atoms + 1))
    
    # Fill in the old distance matrix
    new_dmat[:num_atoms, :num_atoms] = dmat
    
    # Compute distances from the new point to all other points
    dists = numba_distances_to_point0(points, Xo, Yo)
    
    # Fill in the last row and last column
    new_dmat[-1, :num_atoms] = dists
    new_dmat[:num_atoms, -1] = dists
    
    # Fill in the last cell (distance to itself)
    new_dmat[-1, -1] = 0.0

    return new_dmat




