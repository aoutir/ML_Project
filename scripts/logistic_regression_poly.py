import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
# from least_square_SGD import *
# from least_squares_GD import *
from least_squares_ridge_regression import *
# from reg_log_reg import *

import os
PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'

def sigmoid(t):
    """apply the sigmoid function on t."""
    sigma_t = (1+np.exp(-t))**(-1)
    return sigma_t


def calculate_loss_log_reg(y, tx, w):
    """compute the loss: negative log likelihood."""
    sigma_t = sigmoid(tx@w)
    N = y.shape[0]
    L = 0
    for i in range(0,N):
        L = L + (-1*(y[i]*np.log(sigma_t[i]) + (1-y[i])*np.log(1-sigma_t[i])))
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

def standardize(x):
    ''' fill your code in here...
    '''
    x = x - np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    return x

def logistic_regression_newton_method_demo(y, x, initial_w, max_iter, gamma):
    # init parameters

    threshold = 1e-6
    losses = []

    # build tx
    tx = np.c_[x]
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    temp = np.ones((len(x),1))
    for i in range(1,degree+1):
        new = np.power(x,i)
        temp = np.c_[temp,new]
    return temp


def preprocessing(y, tX, ids):
    """ Splitting the Data based on the Jet Experiment number
    Grouping into Exp 0, Exp 1 and Exp 2&3
    input: data (label, features, ids)
    output: data (label, features, ids) of each experiment in a numpy array
    """
    # initilizing the matrices
    tX_0 = []
    tX_1 = []
    tX_23 = []
    y_0 = []
    y_1 = []
    y_23 = []
    ids_0 = []
    ids_1 = []
    ids_23 = []
    # splitting the data based on the experiment
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
    # converting the data to numpy arrays
    y_0, tX_0, ids_0, y_1, tX_1, ids_1, y_23, tX_23, ids_23 = np.asarray(y_0), np.asarray(tX_0), np.asarray(ids_0), np.asarray(y_1), np.asarray(tX_1), np.asarray(ids_1), np.asarray(y_23), np.asarray(tX_23), np.asarray(ids_23)
    # removing unnecessary features
    tX_0 = np.delete(tX_0, [4,5,6,12,22,23,24,25,26,27,28,29], 1)
    tX_1 = np.delete(tX_1, [4,5,6,12,22,26,27,28], 1)
    tX_23 = np.delete(tX_23, [22], 1)
    # getting the median of every feature for each experiment
    feature_median_0 = np.median(tX_0, axis = 0)
    feature_median_1 = np.median(tX_1, axis = 0)
    feature_median_23 = np.median(tX_23, axis = 0)
    # Replacing missing values (-999) values with the median
    for i in range(0,len(tX_0)):
        temp = tX_0[i,:]
        temp[temp == -999] = feature_median_0[temp == -999]
        tX_0[i,:] = temp
    for i in range(0,len(tX_1)):
        temp = tX_1[i,:]
        temp[temp == -999] = feature_median_1[temp == -999] #+ np.random.rand(1)*0.01
        tX_1[i,:] = temp
    # standardizizing the data by removing the mean and the standard deviation
    tX_0 = standardize(tX_0)
    tX_1 = standardize(tX_1)
    tX_23 = standardize(tX_23)
    # returning the data
    return y_0, tX_0, ids_0, y_1, tX_1, ids_1, y_23, tX_23, ids_23

def main():
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    # setting hyper parameters
    max_iter = 10000
    gamma = 0.1
    degree = 3
    lambda_ridge_regression = 0.016
    y_0, tX_0, ids_0, y_1, tX_1, ids_1, y_23, tX_23, ids_23 = preprocessing(y, tX, ids)

    augm = input("Do you want to do data augmenting? [1] Yes, [2] No ")
    augm = int(augm)
    if augm == 1:
        tx_0_p = build_poly(tX_0, degree)
        tx_1_p = build_poly(tX_1, degree)
        tx_23_p = build_poly(tX_23, degree)
        print('Data augmentation done')
    else:
        tx_0_p = tX_0
        tx_1_p = tX_1
        tx_23_p = tX_23
        print('Data augmentation skipped')
    # picking which algorithm to use to get the weights
    methd = input("Which Method do you want to use: \n [1] Least Squares GD \n [2] Least Squares SGD \n [3] Least Squares \n [4] Ridge Regression \n [5] Logistic Regression \n [6] Regulated Logistic Regression \n ")
    methd = int(methd)
    if methd == 1:
        print('Running Least Squares GD')
        w_0, losses_0 = logistic_regression_newton_method_demo(y_0, tx_0_p, np.zeros((tx_0_p.shape[1], 1)), max_iter, gamma)
        w_1, losses_1 = logistic_regression_newton_method_demo(y_1, tx_1_p, np.zeros((tx_1_p.shape[1], 1)), max_iter, gamma)
        w_23, losses_23 = logistic_regression_newton_method_demo(y_23, tx_23_p, np.zeros((tx_23_p.shape[1], 1)), max_iter, gamma)
    if methd == 2:
        print('Running Least Squares SGD')
        w_0, losses_0 = logistic_regression_newton_method_demo(y_0, tx_0_p, np.zeros((tx_0_p.shape[1], 1)), max_iter, gamma)
        w_1, losses_1 = logistic_regression_newton_method_demo(y_1, tx_1_p, np.zeros((tx_1_p.shape[1], 1)), max_iter, gamma)
        w_23, losses_23 = logistic_regression_newton_method_demo(y_23, tx_23_p, np.zeros((tx_23_p.shape[1], 1)), max_iter, gamma)
    if methd == 3:
        print('Running Least Squares')
        w_0, losses_0 = least_squares_demo(y_0, tx_0_p)
        w_1, losses_1 = least_squares_demo(y_1, tx_1_p)
        w_23, losses_23 = least_squares_demo(y_23, tx_23_p)
    if methd == 4:
        print('Running Ridge Regression')
        w_0, losses_0 = ridge_regression_demo( y_0, tx_0_p , lambda_)
        w_1, losses_1 = ridge_regression_demo( y_1, tx_1_p , lambda_)
        w_23, losses_23 = ridge_regression_demo( y_23, tx_23_p, lambda_)
    elif methd == 5:
        print('Running Logistic Regression')
        w_0, losses_0 = logistic_regression_newton_method_demo(y_0, tx_0_p, np.zeros((tx_0_p.shape[1], 1)), max_iter, gamma)
        w_1, losses_1 = logistic_regression_newton_method_demo(y_1, tx_1_p, np.zeros((tx_1_p.shape[1], 1)), max_iter, gamma)
        w_23, losses_23 = logistic_regression_newton_method_demo(y_23, tx_23_p, np.zeros((tx_23_p.shape[1], 1)), max_iter, gamma)
    elif methd == 6:
        print('Running Regularized Logistic Regression')
        w_0, losses_0 = reg_logisitic_regression(y_0, tx_0_p, np.zeros((tx_0_p.shape[1], 1)), max_iter, gamma)
        w_1, losses_1 = reg_logisitic_regression(y_1, tx_1_p, np.zeros((tx_1_p.shape[1], 1)), max_iter, gamma)
        w_23, losses_23 = reg_logisitic_regression(y_23, tx_23_p, np.zeros((tx_23_p.shape[1], 1)), max_iter, gamma)
    else:
        print('Pick Something between 1 and 6')
        main()
    # deciding whether to use polynomial expension of the data
    if augm == 1:
        y_pred = np.zeros((len(tX_test),1))
        tX_test = build_poly(tX_test, degree)
        # Get the results for the testg data with the different weights based on the experiment number
        delete_0 = np.array([4,5,6,12,22,23,24,25,26,27,28,29])
        delete_1 = np.array([4,5,6,12,22,26,27,28])
        delete_23 = np.array([22])
        delete_0_i = delete_0
        delete_1_i = delete_1
        delete_2_i = delete_23
        if degree >1:
            # Remove columns of the augmented data
            for i in range(1,degree):
                delete_0_i = np.concatenate((delete_0_i, delete_0+(30*i)), axis=None)
                delete_1_i = np.concatenate((delete_1_i, delete_1+(30*i)), axis=None)
                delete_2_i = np.concatenate((delete_2_i, delete_23+(30*i)), axis=None)
        for i in range(0,len(tX_test)):
            if tX_test[i,22] == 0:
                tmp = np.delete(tX_test[i,:], delete_0_i)
                y_pred[i] = np.dot(tmp, w_0)
            if tX_test[i,22] == 1:
                tmp = np.delete(tX_test[i,:], delete_1_i)
                y_pred[i] = np.dot(tmp, w_1)
            else:
                tmp = np.delete(tX_test[i,:], delete_2_i)
                y_pred[i] = np.dot(tmp, w_23)
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
        create_csv_submission(ids_test, y_pred, 'Results_with_Augmentation.csv')
    else:
        y_pred = np.zeros((len(tX_test),1))
        # Get the results for the testg data with the different weights based on the experiment number
        for i in range(0,len(tX_test)):
            if tX_test[i,22] == 0:
                tmp = np.delete(tX_test[i,:], [4,5,6,12,22,23,24,25,26,27,28,29])
                y_pred[i] = np.dot(tmp, w_0)
            if tX_test[i,22] == 1:
                tmp = np.delete(tX_test[i,:], [4,5,6,12,22,26,27,28])
                y_pred[i] = np.dot(tmp, w_1)
            else:
                tmp = np.delete(tX_test[i,:], [22])
                y_pred[i] = np.dot(tmp, w_23)
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
        create_csv_submission(ids_test, y_pred, 'Results_No_Augmentation.csv')

if __name__ == "__main__":
    main()
