import os
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D

metric = tf.keras.metrics.MeanSquaredError()
print(metric([5.],[2.]))
print(metric([0.],[1.]))
print(metric.result())
