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
