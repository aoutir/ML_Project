import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
import os
PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'



def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def standardize(x):
    ''' Removing the mean and the standard deviation
    '''
    x = x - np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    return x

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
    # returning the data
    return y_0, tX_0, ids_0, y_1, tX_1, ids_1, y_23, tX_23, ids_23


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
    # Avoid log RunTimeWarnings becasue of illegal values
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

    # build tx
    # tx = np.c_[np.ones((y.shape[0], 1)), x]
    # w = np.zeros((tx.shape[1], 1))
    # initial_w = w
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


if __name__ == "__main__":
    ''' load the data '''
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    ''' Proprocess '''
    # setting hyper parameters
    max_iter = 10000
    gamma = 0.01
    # preprocess
    y_0, tX_0, ids_0, y_1, tX_1, ids_1, y_23, tX_23, ids_23 = preprocessing(y, tX, ids)

    '''Obtain the model '''
    # Switch from [-1,1] to [0,1]
    y_0[y_0==-1] = 0
    y_1[y_1==-1] = 0
    y_23[y_23==-1] = 0
    w_0, losses_0 = logistic_regression_newton_method_demo(y_0, tX_0, np.zeros((tX_0.shape[1], 1)), max_iter, gamma)
    w_1, losses_1 = logistic_regression_newton_method_demo(y_1, tX_1, np.zeros((tX_1.shape[1], 1)), max_iter, gamma)
    w_23, losses_23 = logistic_regression_newton_method_demo(y_23, tX_23, np.zeros((tX_23.shape[1], 1)), max_iter, gamma)

    ''' Make Predictions on the test data'''
    y_pred = np.zeros((len(tX_test),1))
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
    ''' Create Output file '''
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    create_csv_submission(ids_test, y_pred, 'Results_No_Augmentation.csv')
