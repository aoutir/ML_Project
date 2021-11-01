import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
import os
PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'

def sigmoid_reg_log(t):
    """apply the sigmoid_reg_log function on t."""
    t[t>500] = 500
    t[t<-500] = -500
    sigma_t = 1.0/(1+np.exp(-t))
    return sigma_t


def calculate_penalized_loss(y, tx, w, lambda_):
    """compute the cost by negative log likelihood."""
    sigma_t = sigmoid_reg_log(tx@w)
    # Avoid log RunTimeWarnings becasue of illegal values on sigma
    sigma_t[sigma_t == 0] = 0.00000001
    sigma_t[sigma_t == 1] = 0.99999999
    sigma_t[sigma_t<0.00000001] = 0.00000001
    loss = y.T@np.log(sigma_t) + (1 - y).T@np.log(1 - sigma_t)
    L =  np.squeeze(-loss) + lambda_ * np.squeeze(w.T@w)
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
    L, G = penalized_logistic_regression(y, tx, w, lambda_)

    # update w
    w = w - gamma*G
    return L, w


def reg_logisitic_regression(y, x, initial_w, max_iter, gamma):
    # init parameters
    threshold = 1e-2
    gamma = 0.0001
    lambda_ = 0.00000001
    losses = []

    # tx = np.c_[np.ones((y.shape[0], 1)), x]
    # tx = np.c_[x]
    # build tx
    tx = x
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    # visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_penalized_gradient_descent",True)
    # print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, losses[-1]
