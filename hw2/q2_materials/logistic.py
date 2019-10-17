""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid
import math

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    # TODO: Finish this function
    N, M = data.shape[0], data.shape[1]
    y = np.zeros((N,1))
    new_data=np.hstack((data, np.ones((N,1))))
    y = np.dot(new_data, weights)
    y = sigmoid(y)
    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    # N = targets.shape[0]
    ce = - np.sum(np.dot(targets.T, np.log(y)))
    correct_prediction = len(zip(*np.where((y > 0.5) & (targets == 1)))) + len(zip(*np.where((y<0.5) & (targets==0))))
    frac_correct = float(correct_prediction) / len(targets)
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # TODO: Finish this function
    N, M = data.shape[0], data.shape[1]
    if not hyperparameters["weight_regularization"]:
        y = logistic_predict(weights, data)
        f = np.sum(-np.dot(targets.T,np.log(y))) - np.sum(np.dot((1-targets.T), np.log(1 - y)))
        new_data = np.hstack((data, np.ones((N,1))))
        df = np.dot(new_data.T, (y - targets))
    else:
        f, df, y = logistic_pen(weights, data, targets, hyperparameters)
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """
    N, M = data.shape[0], data.shape[1]
    wd = hyperparameters["weight_decay"]
    y = logistic_predict(weights, data)
    w_without_b = weights[:-1,:]
    penalty = (wd/2) * np.sum(w_without_b * w_without_b)
    constant = -((M/2) * np.log((2 * math.pi)/wd)) if wd != 0 else 0
    # Removed constant since it won't actually penalize weight, and there will be issue when lambda=0
    f = np.sum(-np.dot(targets.T,np.log(y))) - np.sum(np.dot((1-targets.T), np.log(1 - y))) + penalty + constant
    new_data = np.hstack((data, np.ones((N,1))))
    reg = np.pad(wd * w_without_b, ((0,1),(0,0)), 'constant')
    df = np.dot(new_data.T, (y - targets)) + reg
    return f, df, y
