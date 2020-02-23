#!/usr/bin/python
import argparse
import csv
import decimal
import logging
import os
import pickle
import sys
import time
import typing # Typing introduces slight startup time penalties
from decimal import Decimal, getcontext
from functools import reduce
from math import pi
from operator import mul
from typing import Any, Tuple
import gzip
 
import matplotlib.cm as cm
import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
getcontext().prec = 7 # Precision for decimal

parser = argparse.ArgumentParser(description='Kernel Density Estimation')
parser.add_argument("--data", default="cifar", help="data type: either cifar or mnist")
parser.add_argument("--sample_size", default=100, help="number of data points to compute density estimates")



"""
 Function to unzip and unpickle the given datasets
 returns: unpickled file
"""
def load_zip_pickle(f):
    with gzip.open(f, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            u = u.load()
            return(u)
"""
 Function to unpickle the given datasets
 returns: unpickled file
"""
def load_pickle(f):
    with open(f, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            u = u.load()
            return(u)
"""    
 Fuction to load and preprocess MNIST dataset 
 returns: Preprocessed train, valid and test datasets from MNIST
"""
def load_data_mnist(mnist_f="mnist.pkl.gz"):
    u = load_zip_pickle(mnist_f)
    seed = 111
    X_training = u[0][0]
    np.random.seed(seed)
    np.random.shuffle(X_training)
    X_train = X_training[0:10000]
    X_valid = X_training[10000:20000]
    X_test = u[2][0]
    return X_train, X_valid, X_test

"""    
 Fuction to load and preprocess CIFAR dataset 
 returns: Preprocessed train, valid and test datasets from MNIST
"""
def load_cifar(train_dir, test_dir):
    seed = 111
    np.random.seed(seed)
    unpickled_train, unpickled_test = (load_pickle(train_dir), load_pickle(test_dir))
    print(unpickled_train["data"].shape)
    x_train_raw = unpickled_train["data"].astype(np.float64)
    x_train_raw = x_train_raw / 255  # Scaling pixel values between 0 and 1
    np.random.shuffle(x_train_raw) # Shuffling the original training set
    x_train = x_train_raw[0:10000]
    x_val = x_train_raw[10000:20000] # Creating the validation set from the split
    x_test = unpickled_test["data"].astype(np.float64)
    # Using the original 10K test set as it is.
    x_test = x_test / 255  # Scaling pixel values between 0 and 1
    return x_train, x_val, x_test

"""
 Function to visualize the loaded dataset
 returns: plot data
"""
def visualize(image_data, num_img_edge, pixel_rows, pixel_cols,image_channels):
    N = num_img_edge**2
    n=num_img_edge
    D = image_channels
    R = pixel_rows
    C = pixel_cols
    M = image_data[0:N].reshape(N, D, R, C).reshape(n, n, D, R, C).transpose(0, 1, 3, 4, 2)
    img = M.swapaxes(1, 2).reshape(R*n, C*n, D)
    if D == 1:
        img = img.reshape(R*n, C*n)
    fig = plt.imshow(img, cmap=cm.get_cmap("gray"))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    return(plt)

"""
 Function to implement kernel density estimation
"""
def kde_scratch(sigma, D_A, D_B):
	getcontext().prec = 7
	mu, prob_x = D_A.astype(np.float64), 0
	len_D_A, len_D_B, d = len(D_A), len(D_B), len(D_A[0])
	t_1 = -Decimal(0.5 * d) * Decimal(2 * pi * (sigma ** 2)).ln()
	log_k = Decimal(len_D_A).ln()
	
	for i in  range(0, len_D_A):
		t_0 = np.sum((-((np.matlib.repmat(D_B[i], len_D_A, 1).astype(np.float64) - mu) ** 2)) / (2 * (sigma ** 2)), axis=1)
		elements_sum = 0
		for j in  range(0, len_D_B):
			elements_sum += Decimal(t_0[j]).exp()
		prob_x += t_1 - log_k + elements_sum.ln()
	return prob_x / len_D_B

if __name__=="__main__":
    args = parser.parse_args()
    dataset = args.data
    n_samples=args.sample_size
    print("processing dataset ...",dataset)
    if (dataset == 'mnist' or dataset == 'MNIST'):
        X_train, X_valid, X_test = load_data_mnist(mnist_f="./mnist.pkl.gz")
        print(X_train.shape)
        img = visualize(X_train, num_img_edge=20, pixel_rows=28, pixel_cols=28, image_channels=1)
        img.savefig(dataset+'.png')
    elif dataset in ["CIFAR100", "cifar100", "CIFAR", "cifar"]:
        X_train, X_valid, X_test = load_cifar(train_dir="./cifar-100-python/train",test_dir="./cifar-100-python/test")
        print(X_train.shape)
        img = visualize(X_train, num_img_edge=20, pixel_rows=32, pixel_cols=32, image_channels=3)
        img.savefig(dataset+'.png')
    else:
        print("Please Enter one of CIFAR or MNIST")
        
    sigma = [0.05, 0.08, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00] # Grid search
    L_valid = [] # list to store mean of log-likelihood values
    for sg in sigma:
        print("Training with sigma = {}".format(sg))
        kde_prob = kde_scratch(sg, X_train[0:n_samples], X_valid[0:n_samples])
        print ("L_D_valid with sigma {} = {}".format(sg, kde_prob))
        L_valid.append(kde_prob)
