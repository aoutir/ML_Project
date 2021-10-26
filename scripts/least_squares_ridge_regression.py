#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Useful starting lines
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[16]:


import os
from proj1_helpers import *
from plots import *
PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'


# In[17]:


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x) , 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


# In[18]:


def compute_mse(e):
    """calculate the mean square error."""
    return 1/2*np.mean(e**2)


# In[19]:


def least_squares(y, tx):
    """calculate the least squares solution."""    
    # returns mse, and optimal weights
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    opt_weights_w = np.linalg.solve(a,b)    
    mse = 1/2*np.mean((y-tx.dot(opt_weights_w))**2)
    return mse , opt_weights_w


# In[20]:


def least_squares_demo(degree):
    """polynomial regression using least squares method with a given degree."""
    
    y_train ,x_train , ids_train = load_csv_data(DATA_TRAIN_PATH)
    y_test ,x_test ,ids_test = load_csv_data(DATA_TEST_PATH)
    
    # form train and test data with polynomial basis function: TODO
    tx_train_poly = build_poly(x_train, degree)    
    tx_test_poly = build_poly(x_test, degree)    
    print(np.shape(x_train))
    print(np.shape(tx_train_poly))
    # calculate weight through least square: TODO
    
    mse_tr , weight = least_squares(y_train, tx_train_poly)
    
    # calculate RMSE for train and test data,
    # and store them in rmse_tr and rmse_te respectively: TODO
    print("weight" , np.shape(weight))
    print("error train" , compute_mse(y_train-tx_train_poly.dot(weight)) )
    rmse_tr = np.sqrt(2 * compute_mse(y_train-tx_train_poly.dot(weight)))
    rmse_te = np.sqrt(2 * compute_mse(y_test-tx_test_poly.dot(weight)))  
    
    OUTPUT_PATH = 'test0.1' # TODO: fill in desired name of output file for submission
    y_pred = predict_labels(weight, tx_test_poly)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
  
    print(" degree={d}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
           d=degree, tr=rmse_tr, te=rmse_te))


# In[21]:


degree = 3
least_squares_demo( degree)


# In[22]:


def ridge_regression(y, tx, lambda_):
    """implementation of ridge regression method for a given lambda value."""
    lambda_p = 2*tx.shape[0]*lambda_
    eye = np.identity(tx.shape[1])
    a = tx.T.dot(tx) + lambda_p*eye
    b = tx.T.dot(y)
    ridge_weights = np.linalg.solve(a,b)  
    return ridge_weights


# In[23]:


def ridge_regression_demo( degree):
    """ridge regression demo."""
    # define parameter
    lambdas = np.logspace(-5, 0, 15)
   
    # split the data, and return train and test data: TODO    
    y_train ,x_train  , ids_train = load_csv_data(DATA_TRAIN_PATH)
    y_test , x_test ,ids_test = load_csv_data(DATA_TEST_PATH)
    
    # form train and test data with polynomial basis function: TODO
    tx_train_poly = build_poly(x_train, degree)    
    tx_test_poly = build_poly(x_test, degree)        
    rmse_tr = []
    rmse_te = []
    for ind, lambda_ in enumerate(lambdas):
        
        # ridge regression with a given lambda
        weight = ridge_regression(y_train, tx_train_poly, lambda_)
        print(np.shape(weight))
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


# In[24]:


degree = 3
ridge_regression_demo(degree)


# In[ ]:




