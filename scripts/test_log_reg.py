import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
import os
PROJECT_PATH = os.path.dirname(os.getcwd())
DATA_TEST_PATH = PROJECT_PATH + '/data/test.csv'

if __name__ == "__main__":
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    weights = [[ 1.04538050e-04], [-6.54171426e-03], [-5.85284348e-03], [-2.98997870e-04], [-3.53059268e-02], [ 4.19694173e-04], [-2.53438312e-02], [ 2.97947313e-01], [ 1.81651633e-04], [-2.17972191e+00], [-2.35488901e-01], [ 8.05650981e-02], [ 7.73573471e-02], [ 2.18648324e+00], [-3.69594433e-04], [-7.34404175e-04], [ 2.19221093e+00], [-7.37638260e-04], [ 1.01818252e-03], [ 3.06951075e-03], [ 3.87778139e-04], [-5.66688019e-04], [-4.29103632e-01], [-2.58669723e-03], [ 1.39939245e-03], [ 1.72453043e-03], [-3.27394412e-03], [-4.18565691e-03], [-9.05138346e-03], [ 2.18144973e+00]]
    y_pred = predict_labels(weights, tX_test)
    create_csv_submission(ids_test, y_pred, 'results.csv')
