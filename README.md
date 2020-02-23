# Kernel Density Estimation using Guassian Kernels

In statistics, maximum likelihood estimation (MLE) is a method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the likelihood of making the observations P(x) given the parameters. Prior to applying MLE, we need to make an assumption about the data distribution. A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.
This concept is similar to using Kernel Density Estimation (KDE) which is intended to estimate the probability density function of a random variable using sampling distribution. However **KDE is a non-parametric approach** in a sense that any kernel function can be used for smoothing as long as it always produces a value greater than or equal to zero and it should integrate to 1 over the sample space.

Guassian function can be used for both kernels and P(x) which makes MLE and KDE very similar in practice. 

Here the task is to estimate the parameters of Guassian distribution for MNIST and CIFAR datasets. The main code kernel_density_estimate.py inputs one of those datasets and number of samples to use in the estimation and outputs the liklihood value over different values of a sigma. 
