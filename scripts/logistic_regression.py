import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
# from least_square_SGD import *
# from least_squares_GD import *
# from least_squares_ridge_regression import *
# from reg_log_reg import *

import os
PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'

def sigmoid(t):
    """apply the sigmoid function on t."""
    sigma_t = (1+np.exp(-t))**(-1)
    t[t>500] = 500
    t[t<-500] = -500
    sigma_t = 1.0/(1+np.exp(-t))
    return sigma_t


def calculate_loss_log_reg(y, tx, w):
    """compute the loss: negative log likelihood."""
    sigma_t = sigmoid(tx@w)
    N = y.shape[0]
    # Avoid log RunTimeWarnings becasue of illegal values, and put limits on the log function
    sigma_t[sigma_t == 0] = 0.0000000001
    sigma_t[sigma_t == 1] = 0.9999999999
    L =  y.T@np.log(sigma_t) + (1 - y).T@np.log(1 - sigma_t)
    L = np.squeeze(-L)
    return L

def calculate_gradient_log_reg(y, tx, w):
    """compute the gradient of loss."""
    sigma_t = sigmoid(tx@w)
    y = np.reshape(y, (-1, 1))
    G = tx.T@(sigma_t - y)
    return G

def calculate_hessian_log_reg(y, tx, w):
    """return the Hessian of the loss function."""

    # calculate Hessian:
    predictions = sigmoid(tx@w)
    H = tx.T@(predictions*(1-predictions)*tx)
    return H

def logistic_regression(y, tx, w):
    """return the loss, gradient, and Hessian."""

    L = calculate_loss_log_reg(y, tx, w)
    G = calculate_gradient_log_reg(y, tx, w)
    H = calculate_hessian_log_reg(y, tx, w)
    # return loss, gradient, and Hessian
    return L, G, H


def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    # return loss, gradient and Hessian
    L, G, H = logistic_regression(y, tx, w)
    w = w - gamma*np.linalg.solve(H,G)

    return L, w

def logistic_regression_newton_method_demo(y, x, initial_w, max_iter, gamma):
    # init parameters

    threshold = 1e-3
    losses = []

    # To add a bias term, uncomment the 3 lines below, and comment the two following lines
    # tx = np.c_[np.ones((y.shape[0], 1)), x]
    # w = np.zeros((tx.shape[1], 1))
    # initial_w = w
    # build tx
    tx = np.c_[x]
    w = initial_w
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses
