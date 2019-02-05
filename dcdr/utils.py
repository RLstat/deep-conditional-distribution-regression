# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 23:15:02 2019

@author: Rui Li
"""

import numpy as np
import pandas as pd

def _check_input(cdf, test_y, y_grid):
    if isinstance(cdf, pd.DataFrame):
        y_grid = cdf.columns.values
        cdf_matrix = cdf.values
    elif isinstance(cdf, np.ndarray):
        cdf_matrix = cdf
        if y_grid is None:
            raise ValueError('cdf is a numpy array, its corresponding grid value need to be provided')
    
    if test_y.ndim ==1:
        test_y = test_y.reshape(-1, 1)
    
    return cdf_matrix, test_y, y_grid
    

def evaluate_coverage(cdf, test_y, interval, y_grid=None):
    
    cdf_matrix, test_y, y_grid = _check_input(cdf, test_y, y_grid)

    ntest = test_y.shape[0]    
    test_density_gridM = np.tile(y_grid, ntest).reshape(-1, len(y_grid))
    test_cvrM = cdf_to_quantile(cdf_matrix, interval, test_density_gridM)
    cvr_indM  = np.where((test_y <= test_cvrM), 1, 0).sum(axis = 1)
    cover_percent = (cvr_indM == 1).sum()/cvr_indM.shape[0]
    
    return cover_percent

def evaluate_crps(cdf, test_y, y_grid=None):
    
    cdf_matrix, test_y, y_grid = _check_input(cdf, test_y, y_grid)
            
    ntest = test_y.shape[0]    
    test_density_gridM = np.tile(y_grid, ntest).reshape(-1, len(y_grid))
    Test_indicator_matrix = np.where((test_y <= test_density_gridM), 1, 0)
    test_score = np.mean(np.square(cdf_matrix - Test_indicator_matrix))

    return test_score

def evaluate_quantile_loss(cdf, test_y, quantiles, y_grid=None):
    
    cdf_matrix, test_y, y_grid = _check_input(cdf, test_y, y_grid)
            
    ntest = test_y.shape[0]    
    test_density_gridM = np.tile(y_grid, ntest).reshape(-1, len(y_grid))

    if not isinstance(quantiles, list):
        if isinstance(quantiles, np.ndarray):
            quantiles = quantiles.tolist()
        else:
            quantiles = [quantiles]
    
    test_qtM = cdf_to_quantile(cdf_matrix, quantiles, test_density_gridM)
    test_score = ave_quant_loss(test_y, test_qtM, quantiles)

    return test_score

def evaluate_rmse(cdf, test_y, y_grid=None):
    
    cdf_matrix, test_y, y_grid = _check_input(cdf, test_y, y_grid)

    grid_width = np.diff(y_grid).mean()
    
    test_mean = (cdf_matrix[:,-1]*y_grid[-1] 
                - cdf_matrix[:,0]*y_grid[0] 
                - cdf_matrix.sum(axis=1)*grid_width).reshape(test_y.shape)
    
    test_score = np.sqrt(np.mean(np.square(test_y - test_mean)))

    return test_score


def cdf_to_quantile(cdf, quantiles, vseq_qM):
    nn_quantM = np.zeros((cdf.shape[0], len(quantiles)))
    for i, q in enumerate(quantiles):
        cdf_ind = np.argmin(np.abs(cdf-q), axis=1)
        nn_quantM[:,i] = vseq_qM[np.arange(cdf.shape[0]),cdf_ind]

    return nn_quantM 


def ave_quant_loss(testY, qtM, quantiles):
    testY = testY.ravel()
    qt_loss = 0
    for i, qt in enumerate(quantiles):
        qt_loss += np.sum(np.where(qtM[:,i]>=testY, (1-qt)*np.abs(qtM[:,i]-testY), 
                 qt*np.abs(qtM[:,i]-testY)))
        
    ave_qt_loss = qt_loss/(qtM.shape[0]*qtM.shape[1])
    
    return ave_qt_loss 