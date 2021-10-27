#!/usr/bin/env python
# coding: utf-8

# In[75]:


# Useful starting lines
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[76]:


import matplotlib.pyplot as plt
import os
from proj1_helpers import *
from plots import *
PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'


# In[77]:


def standardize(x):
   
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data


# In[78]:


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x) , 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


# In[79]:


def compute_mse(e):
    """calculate the mean square error."""
    return 1/2*np.mean(e**2)


# In[80]:


def least_squares(y, tx):
    """calculate the least squares solution."""    
    # returns mse, and optimal weights
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    opt_weights_w = np.linalg.solve(a,b)    
    mse = 1/2*np.mean((y-tx.dot(opt_weights_w))**2)
    return mse , opt_weights_w


# In[81]:


def least_squares_demo():
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression."""
    
    y_train ,x_train , ids_train = load_csv_data(DATA_TRAIN_PATH)
    y_test ,x_test ,ids_test = load_csv_data(DATA_TEST_PATH)
    
    plt.plot(x_train ,y_train , "kx" )
    x_train = standardize(x_train)
    x_test = standardize(x_test)
    # define parameters
    degrees = [1, 3, 7, 12]
    # define the structure of the figure
    num_row = 2
    num_col = 2
    f, axs = plt.subplots(num_row, num_col)

    for ind, degree in enumerate(degrees):   

        # form train and test data with polynomial basis function: TODO
        tx_train_poly = build_poly(x_train, degree)    
        tx_test_poly = build_poly(x_test, degree)    
        
        # calculate weight through least square: TODO

        mse_tr , weight = least_squares(y_train, tx_train_poly)

        # calculate RMSE for train and test data,
        # and store them in rmse_tr and rmse_te respectively: TODO
        print("error train" , compute_mse(y_train-tx_train_poly.dot(weight)) )
        rmse_tr = np.sqrt(2 * compute_mse(y_train-tx_train_poly.dot(weight)))
        rmse_te = np.sqrt(2 * compute_mse(y_test-tx_test_poly.dot(weight)))  

        OUTPUT_PATH = 'test0.1' # TODO: fill in desired name of output file for submission
        y_pred = predict_labels(weight, tx_test_poly)
        create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

        print("Processing {i}th experiment, degree={d}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
               i=ind + 1 , d=degree, tr=rmse_tr, te=rmse_te))
       
        # plot fitted curve of test data 
        
        #plot_fitted_curve(y_test, tx_test_poly, weight, degree, axs[ind // num_col][ind % num_col])
    #plt.tight_layout()
    #plt.savefig("visualize_polynomial_regression")
    #plt.show()


# In[ ]:


least_squares_demo()


# In[ ]:


def ridge_regression(y, tx, lambda_):
    """implementation of ridge regression method for a given lambda value."""
    lambda_p = 2*tx.shape[0]*lambda_
    eye = np.identity(tx.shape[1])
    a = tx.T.dot(tx) + lambda_p*eye
    b = tx.T.dot(y)
    ridge_weights = np.linalg.solve(a,b)  
    return ridge_weights


# In[ ]:


def ridge_regression_demo( degree):
    """ridge regression demo."""
    # define parameter
    lambdas = np.logspace(-5, 0, 15)
   
    # split the data, and return train and test data: TODO    
    y_train ,x_train  , ids_train = load_csv_data(DATA_TRAIN_PATH)
    y_test , x_test ,ids_test = load_csv_data(DATA_TEST_PATH)
    
    x_train = standardize(x_train)
    x_test = standardize(x_test)
    # form train and test data with polynomial basis function: TODO
    tx_train_poly = build_poly(x_train, degree)    
    tx_test_poly = build_poly(x_test, degree)        
    rmse_tr = []
    rmse_te = []
    for ind, lambda_ in enumerate(lambdas):
        
        # ridge regression with a given lambda
        weight = ridge_regression(y_train, tx_train_poly, lambda_)
        print(compute_mse(y_train-tx_train_poly.dot(weight)))
        rmse_tr.append(np.sqrt(2 * compute_mse(y_train-tx_train_poly.dot(weight))))
        rmse_te.append(np.sqrt(2 * compute_mse(y_test-tx_test_poly.dot(weight))))

        print(" degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
                d=degree, l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))
    
    # GENERATE PREDICTIONS      
    OUTPUT_PATH = 'test0.2' # TODO: fill in desired name of output file for submission
    y_pred = predict_labels(weight, tx_test_poly)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH) 
    # Plot the obtained results
    plot_train_test(rmse_tr, rmse_te, lambdas, degree)


# In[ ]:


degree = 3
ridge_regression_demo(degree)


# In[ ]:




