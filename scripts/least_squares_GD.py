import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
import os
from sklearn import preprocessing

PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'

def calculate_mse_LS_GD(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss_LS_GD(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse_LS_GD(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)

    return grad, err

def standardize(x):
    """standardize the data with mean and standard deviation"""
    if x.shape[1] > 30:
        pass
    else:
        x = x - np.mean(x, axis=0)
        x /= np.std(x, axis=0)
    return x



def gradient_descent_demo(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    tx = standardize(tx)
    y = np.reshape(y, (-1, 1))
    gamma = 0.1
    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)
        loss = compute_loss_LS_GD(y, tx, w)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        print("Current iteration={i}, the loss={l}".format(i=n_iter, l=loss))
        #print("Gradient Descent : loss={l} ".format( l=loss ))
        if(n_iter == 999):
            print("Gradient Descent : loss={l} ".format( l=loss ))
            we ="["
            for i in w:
                we += str(i)
                we += ","

            we += "]"
            print(we)
    return loss, w
