# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:46:32 2016
@author: andrejk
"""
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import r2_score
import pandas as pd


# -----------------------------------------------------------------------------
# Time series regression
def tsr_pcf(y, x, nlags=10):
    x_tld = x - x.mean()
    y_tld = y - y.mean()
    d_len = len(x)
    pcf = np.zeros(nlags)
    p_vals = np.zeros(nlags)

    for k in range(1, nlags + 1):  # Independent fit
        curr_X_tab = []
        curr_y = y_tld[k - 1:]
        for i in range(0, k):  # Enter [x[t-k+1], x[t-k+2], ..., x[t]
            curr_X_tab.append(x_tld[i:d_len + i - k + 1])
        curr_X = np.array(curr_X_tab).T

        curr_mod = sm.OLS(curr_y, curr_X)
        curr_res = curr_mod.fit()

        pcf[k - 1] = curr_res.params[-1]
        p_vals[k - 1] = curr_res.pvalues[-1]

    return pcf, p_vals


# -----------------------------------------------------------------------------
# Adjust Rsq
# @param n num of samplen
# @param m num of variables
def rsq_adjIt(R_sq, n, m):
    return 1 - (1 - R_sq) * (n - 1) / (n - m - 1)


# -----------------------------------------------------------------------------
# Stack shifted vecs
# @basic It stacks intepednet variables as columns in X
# No stacking means p=0
def tsr_stack_indep(y, X, p_lags, s_lag=0):
    p_lags = (p_lags + np.ones(len(p_lags))).astype(int)
    p_max = np.max(p_lags)
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1)
    x_len, x_num = X.shape[0], X.shape[1]

    y_ext = y[p_max - 1 + s_lag:]  # Cut y

    X_ext = np.array([]).reshape(x_len - p_max + 1 - s_lag, 0)
    for j in range(0, x_num):  # Cut all X
        curr_X_tab = []
        for i in range(0, p_lags[j]):
            curr_X_tab.append(X[p_max - i - 1 + s_lag:x_len - i, j])
            curr_X = np.array(curr_X_tab).T
        X_ext = np.concatenate((X_ext, curr_X), axis=1)

    return y_ext, X_ext


# A fitting function to predict scalar y from signals in X where lags are time legs of each particular model
# @brief: it regress y from X
# @param: y independent scalar variable
# @param: X dependent variables
# @param: p_lags: p lags used to stack variables
# @param: s_lag: where lags are starting
# @return: fitted model
def ts_regress(y, X, p_lags, s_lag=0):
    y_e, X_e = tsr_stack_indep(y, X, p_lags, s_lag)  # Add delayed data

    # print(X.shape, y.shape)
    xT, yT = test_to_Supervised(X, y, p_lags[0])  # TODO hardkodirano
    # print('x, y: ')
    # print(X)
    # print(y)
    # print('xe, ye: ')
    # print(X_e)
    # print(y_e)
    # print('test: ')
    # print(xT)
    # print(yT)
    # print('p_lags', p_lags)
    # print(xT.shape)
    model_fited = sm.OLS(yT, xT).fit()

    # params = model_fited.params
    # rsq_value = model_fited.rsquared_adj
    # rsq_value = results.rsquared
    # p_value = model_fited.f_pvalue
    # p_vals = results.pvalues

    # print 'X_e.shape(' + str(X_e.shape[0]) + ', ' + str(X_e.shape[1]) + ')'

    return model_fited


def test_to_Supervised(X, y, lag):
    Xdf = pd.DataFrame(X)
    Xcolumns = [Xdf.shift(i) for i in range(1, lag + 1)]
    Xcolumns.append(Xdf)
    X_ext = pd.concat(Xcolumns, axis=1)
    X_ext = X_ext.dropna(0)

    Xnp = np.roll(X_ext.values, 1, axis=1)

    # try dropping the first column (current feature data)
    Xnp = Xnp[:, 1:]

    ynp = np.array(y[lag:])


    return Xnp, ynp


# Compute Rsq adjuste on fitted model
def ts_regress_eval(y_eval, X_eval, p_lags, model_fited, s_lag=0):
    y_e, X_e = tsr_stack_indep(y_eval, X_eval, p_lags, s_lag)  # Add delayed data
    y_eval_pred = model_fited.predict(X_e)

    p_max = np.max(p_lags) + 1
    y_eval = y_eval[p_max - 1:]  # Cut y

    rsq = r2_score(y_eval_pred, y_eval)
    rsq_eval = rsq_adjIt(rsq, X_e.shape[0], X_e.shape[1])
    return rsq_eval
