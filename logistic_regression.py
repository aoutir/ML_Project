import numpy as np

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
    w = w - gamma*np.linalg.solve(H,G)

    return L, w

def logistic_regression_newton_method_demo(y, tx, initial_w, max_iter, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

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


logistic_regression_newton_method_demo(y, tx, initial_w, max_iter, gamma)
