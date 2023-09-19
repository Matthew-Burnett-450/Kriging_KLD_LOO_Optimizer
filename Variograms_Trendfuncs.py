from typing import Any
import numpy as np
import time
import tensorflow as tf
from numba import jit,vectorize,float32

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


@jit(fastmath=True,forceobj=True)
def numba_dist_matrix(points):
    num_atoms = len(points)
    
    # Create expanded arrays for broadcasting
    points_expanded1 = points[:, np.newaxis, :]
    points_expanded2 = points[np.newaxis, :, :]
    
    # Compute the difference
    diff = points_expanded1 - points_expanded2
    
    # Reshape to 2D array to use with np.linalg.norm
    diff_reshaped = diff.reshape(-1, diff.shape[-1])
    
    # Compute the norm (Euclidean distance)
    dist_reshaped = np.linalg.norm(diff_reshaped, axis=1)
    
    # Reshape back to original 2D matrix shape
    dmat = dist_reshaped.reshape(num_atoms, num_atoms)
    
    return dmat

@jit(fastmath=True, forceobj=True)
def numba_distances_to_point0(points, Xo, Yo):
    # Single point to a 2D shape for broadcasting
    single_point = np.array([Xo, Yo])[np.newaxis, :]
    
    # Compute the differences between the single point and all other points
    diff = single_point - points
    
    # Compute the norm (Euclidean distance) for each pair
    distances_to_point0 = np.linalg.norm(diff, axis=1)
    
    return distances_to_point0




"""        
n=1000
listofns=[5,10,20,50,100,200,1000,2000,3000,5000]


        
np.random.seed(80085069) # Seed for reproducibility
# Sample points for testing

sample_points = np.random.rand(n, 2,)
# Run the function locally to get the distance matrix
start=time.time()
matrix=numba_dist_matrix(sample_points)
end=time.time()
print('_______Distance_Matrix_______')
print(f'{end-start}s total')
print(f'{(end-start)/(n)*10**6}μs per point')

#____Benchmarking____#


variogram=GaussianVariogram(100,5)

start2=time.time()
Cmatrix=variogram(matrix.astype(np.float16))
end2=time.time()
print('________Variogram____________')
print(end2-start2)
print(f'{(end2-start2)/(np.shape(matrix)[0]*np.shape(matrix)[1])*10**6} μs')
print(f'{np.shape(matrix)[0]*np.shape(matrix)[1]} elements')

print('_______Total_________________')
print(end2-start)
print(f'{(end2-start)/(n)*10**6} μs per point')"""
