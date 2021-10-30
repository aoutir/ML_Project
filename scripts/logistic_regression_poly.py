import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
import os
PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'

def sigmoid(t):
    """apply the sigmoid function on t."""
    sigma_t = (1+np.exp(-t))**(-1)
    return sigma_t

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    sigma_t = sigmoid(tx@w)
    N = y.shape[0]
    L = 0
    for i in range(0,N):
        L = L + (-1*(y[i]*np.log(sigma_t[i]) + (1-y[i])*np.log(1-sigma_t[i])))
    return L

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sigma_t = sigmoid(tx@w)
    y = np.reshape(y, (-1, 1))
    G = tx.T@(sigma_t - y)
    return G

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""

    # calculate Hessian:
    predictions = sigmoid(tx@w)
    H = tx.T@(predictions*(1-predictions)*tx)
    return H

def logistic_regression(y, tx, w):
    """return the loss, gradient, and Hessian."""

    L = calculate_loss(y, tx, w)
    G = calculate_gradient(y, tx, w)
    H = calculate_hessian(y, tx, w)
    # return loss, gradient, and Hessian
    return L, G, H


def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """

    # return loss, gradient and Hessian
    L, G, H = logistic_regression(y, tx, w)

    # update w:
    # print('w: ', w)
    w = w - gamma*np.linalg.solve(H,G)

    return L, w

def standardize(x):
    ''' fill your code in here...
    '''
    # x = x - np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    return x

def logistic_regression_newton_method_demo(y, x, initial_w, max_iter, gamma):
    # init parameters

    threshold = 7
    losses = []

    # build tx
    tx = np.c_[x]
    w = initial_w
# 30
    # w =  [[ 1.04538050e-04], [-6.54171426e-03], [-5.85284348e-03], [-2.98997870e-04], [-3.53059268e-02], [ 4.19694173e-04], [-2.53438312e-02], [ 2.97947313e-01], [ 1.81651633e-04], [-2.17972191e+00], [-2.35488901e-01], [ 8.05650981e-02], [ 7.73573471e-02], [ 2.18648324e+00], [-3.69594433e-04], [-7.34404175e-04], [ 2.19221093e+00], [-7.37638260e-04], [ 1.01818252e-03], [ 3.06951075e-03], [ 3.87778139e-04], [-5.66688019e-04], [-4.29103632e-01], [-2.58669723e-03], [ 1.39939245e-03], [ 1.72453043e-03], [-3.27394412e-03], [-4.18565691e-03], [-9.05138346e-03], [ 2.18144973e+00]]

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and losses[-1] < 0:
            break
    # visualization
    # visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_newton_method",True)
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
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
    tX_0 = np.delete(tX_0, [4,5,6,8,12,22,23,24,25,26,27,28,29], 1)
    tX_1 = np.delete(tX_1, [4,5,6,12,22,26,27,28,29], 1)
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

if __name__ == "__main__":
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    # setting hyper parameters
    max_iter = 10000
    gamma = 0.1
    degree = 3
    y_0, tX_0, ids_0, y_1, tX_1, ids_1, y_23, tX_23, ids_23 = preprocessing(y, tX, ids)

    augm = input("Do you want to do data augmenting? [1] Yes, [2]No ")
    if augm == 1:
        tx_0_p = build_poly(tX_0, degree)
        tx_1_p = build_poly(tX_1, degree)
        tx_23_p = build_poly(tX_23, degree)
    else:
        tx_0_p = tX_0
        tx_1_p = tX_1
        tx_23_p = tX_23
    # picking which algorithm to use to get the weights
    meth = input("Which Method do you want to use: [1] Logistic Regression, [2] Regulated Logistic Regression ")
    if meth == 1:
        w_0, losses_0 = logistic_regression_newton_method_demo(y_0, tx_0_p, np.zeros((tx_0_p.shape[1], 1)), max_iter, gamma)
        w_1, losses_1 = logistic_regression_newton_method_demo(y_1, tx_1_p, np.zeros((tx_1_p.shape[1], 1)), max_iter, gamma)
        w_23, losses_23 = logistic_regression_newton_method_demo(y_23, tx_23_p, np.zeros((tx_23_p.shape[1], 1)), max_iter, gamma)
    # deciding whether to use polynomial expension of the data
    if augm == 1:
        y_pred = np.zeros((len(tX_test),1))
        tX_test = build_poly(tX_test, degree)
        # Get the results for the testg data with the different weights based on the experiment number
        for i in range(0,len(tX_test)):
            if tX_test[i,22] == 0:
                tmp = np.delete(tX_test[i,:], [4,5,6,8,12,22,23,24,25,26,27,28,29,4+30,5+30,6+30,8+30,12+30,22+30,23+30,24+30,25+30,26+30,27+30,28+30,29+30,4+60,5+60,6+60,8+60,12+60,22+60,23+60,24+60,25+60,26+60,27+60,28+60,29+60])
                y_pred[i] = np.dot(tmp, w_0)
            if tX_test[i,22] == 1:
                tmp = np.delete(tX_test[i,:], [4,5,6,12,22,26,27,28,29,4+30,5+30,6+30,12+30,22+30,26+30,27+30,28+30,29+30,4+60,5+60,6+60,12+60,22+60,26+60,27+60,28+60,29+60])
                y_pred[i] = np.dot(tmp, w_1)
            else:
                tmp = np.delete(tX_test[i,:], [22, 22+30, 22+60])
                y_pred[i] = np.dot(tmp, w_23)
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
        create_csv_submission(ids_test, y_pred, 'resultsmedian.csv')
    else:
        y_pred = np.zeros((len(tX_test),1))
        # Get the results for the testg data with the different weights based on the experiment number
        for i in range(0,len(tX_test)):
            if tX_test[i,22] == 0:
                tmp = np.delete(tX_test[i,:], [4,5,6,8,12,22,23,24,25,26,27,28,29])
                y_pred[i] = np.dot(tmp, w_0)
            if tX_test[i,22] == 1:
                tmp = np.delete(tX_test[i,:], [4,5,6,12,22,26,27,28,29])
                y_pred[i] = np.dot(tmp, w_1)
            else:
                tmp = np.delete(tX_test[i,:], [22])
                y_pred[i] = np.dot(tmp, w_23)
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
        create_csv_submission(ids_test, y_pred, 'resultsmedian.csv')