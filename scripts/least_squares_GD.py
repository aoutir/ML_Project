import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
import os
from sklearn import preprocessing

PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)

    return grad, err


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    tx = standardize(tx)
    y = np.reshape(y, (-1, 1))

    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent : loss={l} ".format( l=loss ))

        if(n_iter == 999):
            print("Gradient Descent : loss={l} ".format( l=loss ))
            we ="["
            for i in w:
                we += str(i)
                we += ","

            we += "]"
            print(we)


    return losses, w


def standardize(x):

    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)

    return std_data
def split_data(y, tX, ids):
    tX_0 = []
    tX_1 = []
    tX_23 = []
    y_0 = []
    y_1 = []
    y_23 = []
    ids_0 = []
    ids_1 = []
    ids_23 = []
    for i in range (0, len(tX)):
        if tX[i,22] == 0:
            tX_0.append(tX[i,:])
            y_0.append(y[i])
            ids_0.append(ids[i])
        elif tX[i,22] == 1:
            tX_1.append(tX[i,:])
            y_1.append(y[i])
            ids_1.append(ids[i])
        else:
            tX_23.append(tX[i,:])
            y_23.append(y[i])
            ids_23.append(ids[i])
    y_0, tX_0, ids_0, y_1, tX_1, ids_1, y_23, tX_23, ids_23 = np.asarray(y_0), np.asarray(tX_0), np.asarray(ids_0), np.asarray(y_1), np.asarray(tX_1), np.asarray(ids_1), np.asarray(y_23), np.asarray(tX_23), np.asarray(ids_23)
    tX_0 = np.delete(tX_0, [0,4,5,6,12,22,23,24,25,26,27,28,29], 1)
    tX_1 = np.delete(tX_1, [4,5,6,12,22,26,27,28], 1)
    return y_0, tX_0, ids_0, y_1, tX_1, ids_1, y_23, tX_23, ids_23

if __name__ == "__main__":
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    #tX_scaled = standardize(tX)
    #w_initial =  np.zeros((tX_scaled.shape[1], 1))
    #losses, w= gradient_descent(y, tX_scaled, w_initial, max_iters, gamma)

    max_iters = 1000
    gamma = 0.0001
    y_0, tX_0, ids_0, y_1, tX_1, ids_1, y_23, tX_23, ids_23 = split_data(y, tX, ids)

    losses0, w_0= gradient_descent(y_0, tX_0, np.zeros((tX_0.shape[1], 1)), max_iters, gamma)
    losses_1, w_1 = gradient_descent(y_1, tX_1, np.zeros((tX_1.shape[1], 1)), max_iters, gamma)
    losses_23, w_23 = gradient_descent(y_23, tX_23, np.zeros((tX_23.shape[1], 1)), max_iters, gamma)


    y_pred = np.zeros((len(tX_test),1))
    for i in range(0,len(tX_test)):
        if tX_test[i,22] == 0:
            tmp = np.delete(tX_test[i,:], [0,4,5,6,12,22,23,24,25,26,27,28,29])
            y_pred[i] = np.dot(tmp, w_0)
        if tX_test[i,22] == 1:
            tmp = np.delete(tX_test[i,:], [4,5,6,12,22,26,27,28])
            y_pred[i] = np.dot(tmp, w_1)
        else:
            y_pred[i] = np.dot(tX_test[i], w_23)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    create_csv_submission(ids_test, y_pred, 'resultsSGD.csv')
