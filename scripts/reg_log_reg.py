import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
import os
PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'

def sigmoid_reg_log(t):
    """apply the sigmoid_reg_log function on t."""
    sigma_t = (1+np.exp(-t))**(-1)
    t[t>500] = 500
    t[t<-500] = -500
    sigma_t = 1.0/(1+np.exp(-t))
    return sigma_t


def calculate_penalized_loss(y, tx, w, lambda_):
    """compute the loss: negative log likelihood."""
    sigma_t = sigmoid_reg_log(tx@w)
    N = y.shape[0]
    L = 0
    sigma_t[sigma_t == 0] = 0.0000000001
    sigma_t[sigma_t == 1] = 0.9999999999
    for i in range(0,N):
        L = L + (-1*(y[i]*np.log(sigma_t[i]) + (1-y[i])*np.log(1-sigma_t[i])))
    L = L + lambda_*np.linalg.norm(w.T@w, ord=2)
    return L

def calculate_penalized_gradient(y, tx, w, lambda_):
    """compute the gradient of loss."""
    sigma_t = sigmoid_reg_log(tx@w)
    y = np.reshape(y, (-1, 1))
    G = tx.T@(sigma_t - y) + 2*lambda_*w
    return G

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""

    L = calculate_penalized_loss(y, tx, w, lambda_)
    G = calculate_penalized_gradient(y, tx, w, lambda_)
    # return loss, gradient
    return L, G

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """

    # return loss, gradient
    loss, G = penalized_logistic_regression(y, tx, w, lambda_)

    # update w
    w = w - gamma*G
    return loss, w


def reg_logisitic_regression(y, x, initial_w, max_iter, gamma):
    # init parameters
    threshold = 1e-2
    gamma = 0.01
    lambda_ = 0.0001
    losses = []

    # build tx
    # tx = np.c_[np.ones((y.shape[0], 1)), x]
    tx = np.c_[x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    # visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_penalized_gradient_descent",True)
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, losses[-1]
