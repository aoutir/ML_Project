import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
import os
PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'



""" LEAST SQUARES GRADIENT DESCENT CODE"""

def optimize_hyperparamters_GD():
    ''' Grid search to find the best hyperparameter gamma
    '''
    learning_rates = [0.001, 0.003, 0.001, 0.03, 0.1, 0.3]
    print('Running Grid search for Least Squares GD')
    minloss = float('inf')
    for step in learning_rates:
        loss, w= least_squares_GD(y, tX, np.zeros((tX.shape[1], 1)), max_iter, step)
        if(loss < minloss):
            optimalstep = step
            minloss = loss
    return optimalstep


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
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


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    y = np.reshape(y, (-1, 1))
    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
    return loss, w



""" LEAST SQUARES STOCHASTIC GRADIENT DESCENT CODE"""


def optimize_hyperparamters_SGD():
    ''' Grid search to find the best hyperparameter gamma for SGD
    '''
    batch_size = 1
    learning_rates = [0.0001, 0.0003, 0.00001, 0.003, 0.02, 0.3]
    print('Running Grid search for Least Squares SGD')
    minloss = float('inf')
    for step in learning_rates:
        loss, w= least_squares_SGD(y, tX, np.zeros((tX.shape[1], 1)), batch_size, max_iter, step)
        if(loss < minloss):
            optimalstep = step
            minloss = loss
    return optimalstep

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):

            grad, _ = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
    return loss, w


""" RIDGE REGRESSION CODE"""


def ridge_regression(y, tx, lambda_):
    """implementation of ridge regression method for a given lambda value."""
    lambda_p = 2*tx.shape[0]*lambda_
    eye = np.identity(tx.shape[1])
    a = tx.T.dot(tx) + lambda_p*eye
    b = tx.T.dot(y)
    ridge_weights = np.linalg.solve(a,b)
    return ridge_weights


def ridge_regression_demo( y , tx , lambda_):
    """ridge regression demo.
    standadizes the data , computes the weight using ridge regression and
    returns the weight and the loss RMSE"""
    weight = ridge_regression(y, tx, lambda_)
    rmse_tr = np.sqrt(2 * compute_mse(y-tx.dot(weight)))
    print("  lambda={l:.3f}, Training RMSE={tr:.3f}".format(
           l=lambda_, tr=rmse_tr ))
    return weight , rmse_tr



""" LEAST SQUARES CODE"""


def least_squares(y, tx):
    """calculate the least squares solution."""

    # returns mse, and optimal weights
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    opt_weights_w = np.linalg.solve(a,b)
    mse = 1/2*np.mean((y-tx.dot(opt_weights_w))**2)
    return mse , opt_weights_w


def least_squares_demo(y , tx):
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression."""
    # returns rmse, and weight

    tx = standardize(tx)
    # define parameters
    # define the structure of the figure
    num_row = 2
    num_col = 2
    f, axs = plt.subplots(num_row, num_col)
    # calculate weight through least square
    mse_tr , w = least_squares(y, tx)
    # calculate RMSE for train data,
    # and store them in rmse_tr
    rmse_tr = np.sqrt(2 * compute_mse(y-tx.dot(w)))
    print("Training RMSE={tr:.3f}".format(tr=rmse_tr))
    return w , mse_tr


""" LOGISITIC REGRESSION CODE"""


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


"""REGULARIZED LOGISITIC REGRESSION"""

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

    # To add a bias term, uncomment the 3 lines below, and comment the two following lines
    # tx = np.c_[np.ones((y.shape[0], 1)), x]
    # w = np.zeros((tx.shape[1], 1))
    # initial_w = w
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
    return w, losses[-1]
