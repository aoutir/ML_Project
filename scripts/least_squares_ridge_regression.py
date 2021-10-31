#!/usr/bin/env python
# coding: utf-8

# In[251]:


# Useful starting lines
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[252]:


import matplotlib.pyplot as plt
import os
from proj1_helpers import *
from plots import *


# In[254]:


def standardize(x):
    """standardize the data with mean and standard deviation"""
    x = x - np.mean(x, axis=0)
    x /= np.std(x, axis=0)
    return x


# In[255]:


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x) , 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


# In[256]:


def compute_mse(e):
    """calculate the mean square error."""
    return (1/2)*np.mean(e**2)


# In[257]:


def least_squares(y, tx):
    """calculate the least squares solution."""    
    # returns mse, and optimal weights
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    opt_weights_w = np.linalg.solve(a,b)    
    mse = 1/2*np.mean((y-tx.dot(opt_weights_w))**2)
    return mse , opt_weights_w


# In[267]:


def least_squares_demo(y , tx):
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression."""
    # returns rmse, and weight

    #plt.plot(x_train ,y_train , "kx" )
    tx = standardize(tx)
    # define parameters
    degrees = [3]
    # define the structure of the figure
    num_row = 2
    num_col = 2
    f, axs = plt.subplots(num_row, num_col)

    for ind, degree in enumerate(degrees):      
        # calculate weight through least square
        mse_tr , w = least_squares(y, tx)
   
        # calculate RMSE for train data,
        # and store them in rmse_tr 
        rmse_tr = np.sqrt(2 * compute_mse(y_train-tx_train_poly.dot(w)))
        print("Processing {i}th experiment, degree={d}, Training RMSE={tr:.3f}".format(
               i=ind + 1 , d=degree, tr=rmse_tr))
    
    return w , mse_tr
       
    
    # plot fitted curve of test data 
        
        #plot_fitted_curve(y_test, tx_test_poly, weight, degree, axs[ind // num_col][ind % num_col])
    #plt.tight_layout()
    #plt.savefig("visualize_polynomial_regression")
    #plt.show()
    #return weight


# In[259]:


def ridge_regression(y, tx, lambda_):
    """implementation of ridge regression method for a given lambda value."""
    lambda_p = 2*tx.shape[0]*lambda_
    eye = np.identity(tx.shape[1])
    a = tx.T.dot(tx) + lambda_p*eye
    b = tx.T.dot(y)
    ridge_weights = np.linalg.solve(a,b)  
    return ridge_weights


# In[260]:


def ridge_regression_demo( y , tx , lambda_):
    """ridge regression demo.
    standadizes the data , computes the weight using ridge regression and 
    returns the weight and the loss RMSE"""   

    x_train = standardize(x_train)        
    weight = ridge_regression(y, tx, lambda_)
    rmse_tr = np.sqrt(2 * compute_mse(y_train-tx_train_poly.dot(w)))
    print("  lambda={l:.3f}, Training RMSE={tr:.3f}".format(
           l=lambda_, tr=rmse_tr[ind] ))
    return weight , loss
    

