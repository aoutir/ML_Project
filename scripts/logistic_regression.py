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
    print('w: ', w)
    w = w - gamma*np.linalg.solve(H,G)

    return L, w

def logistic_regression_newton_method_demo(y, x, initial_w, max_iter, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    # build tx
    tx = np.c_[x]
    w = initial_w

# 30
    # w =  [[ 1.04538050e-04], [-6.54171426e-03], [-5.85284348e-03], [-2.98997870e-04], [-3.53059268e-02], [ 4.19694173e-04], [-2.53438312e-02], [ 2.97947313e-01], [ 1.81651633e-04], [-2.17972191e+00], [-2.35488901e-01], [ 8.05650981e-02], [ 7.73573471e-02], [ 2.18648324e+00], [-3.69594433e-04], [-7.34404175e-04], [ 2.19221093e+00], [-7.37638260e-04], [ 1.01818252e-03], [ 3.06951075e-03], [ 3.87778139e-04], [-5.66688019e-04], [-4.29103632e-01], [-2.58669723e-03], [ 1.39939245e-03], [ 1.72453043e-03], [-3.27394412e-03], [-4.18565691e-03], [-9.05138346e-03], [ 2.18144973e+00]]
# 31
    # Trial 1
    # w = [[-7.45075824e-01], [ 8.54492615e-05], [-6.34296177e-03], [-6.18561249e-03], [-1.51331396e-04], [-2.53142937e-03], [ 3.65525627e-04], [-2.07625476e-02], [ 3.31363478e-01], [ 2.83513885e-05], [-2.38555314e+00], [-2.21722370e-01], [ 8.08709847e-02], [ 3.90296143e-02], [ 2.39288264e+00], [-2.21416067e-04], [-8.31304188e-04], [ 2.39797389e+00], [-3.29741683e-04], [ 7.42458179e-04], [ 3.03798982e-03], [ 9.67857090e-05], [-4.72486868e-04], [-1.78797152e-01], [-3.39880571e-05], [ 3.77572048e-05], [ 2.82727413e-04], [ 1.56272740e-04], [-5.62989747e-03], [-1.03916590e-02], [ 2.38469304e+00]]
    # Trial 2
    # w = [[-7.50607482e-01], [ 8.65030619e-05], [-6.39174510e-03], [-6.24309744e-03], [-1.48397165e-04], [-2.54897935e-03], [ 3.67890867e-04], [-2.09156477e-02], [ 3.34337134e-01], [ 2.98678197e-05], [-2.40119608e+00], [-2.23893334e-01], [ 8.14380489e-02], [ 3.93147194e-02], [ 2.40857787e+00], [-2.22527890e-04], [-8.37139930e-04], [ 2.41372555e+00], [-3.32833055e-04], [ 7.47871635e-04], [ 3.05740381e-03], [ 9.58797066e-05], [-4.76506251e-04], [-1.80029492e-01], [-3.18165854e-05], [ 3.58989799e-05], [ 2.84570775e-04], [ 1.60409162e-04], [-5.67058492e-03], [-1.04702200e-02], [ 2.40032662e+00]]
    
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
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_newton_method",True)
    print("loss={l}".format(l=calculate_loss(y, tx, w)))


if __name__ == "__main__":
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    # _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    initial_w =  np.zeros((tX.shape[1], 1))
    max_iter = 10000
    gamma = 0.00001
    logistic_regression_newton_method_demo(y, tX, initial_w, max_iter, gamma)
