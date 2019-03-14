# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 09:31:58 2019

@author: Rui Li
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from dcdr.deep_hist import Binning_CDF
from dcdr.utils import cdf_to_quantile, evaluate_crps, \
evaluate_quantile_loss, evaluate_rmse, evaluate_coverage

class LogisticRegressionCDF:
    
    def __init__(self, num_cut, solver='lbfgs', multi_class='multinomial',
                 cutpoint_distribution='uniform',
                 max_iter=1000, tol=1e-5, seeding=1234):
        
        self.num_cut = num_cut
        self.solver = solver
        self.multi_class = multi_class
        self.cutpoint_distribution = cutpoint_distribution
        self.max_iter = max_iter
        self.tol = tol
        self.seeding = seeding
        self.lr_model = LR(solver=self.solver,multi_class=self.multi_class, 
                           max_iter=self.max_iter, tol=self.tol, 
                           random_state=self.seeding)
        
    
    def fit_cdf(self, train_x, train_y, ylim=None, y_margin=0.1):
        
        train_x = np.array(train_x)
        train_y = np.array(train_y)
            
        nobs = train_x.shape[0]
        self.p = train_x.shape[1]      
              
        self.x_scaler = StandardScaler()
        scaled_TrainX = self.x_scaler.fit_transform(train_x)           
        
        self.y_min = np.min(train_y)
        self.y_max = np.max(train_y)
        
        if ylim is None:
            self.y_range = self.y_max - self.y_min
            self.ylim = [self.y_min - y_margin*self.y_range, self.y_max + y_margin*self.y_range]
        else:
            self.ylim = ylim.copy()

        if self.ylim[0] >= self.y_min:
            self.ylim[0] = self.y_min
        
        if self.ylim[1] <= self.y_max:
            self.ylim[1] = self.y_max
        
        if self.num_cut < 1:
            self.num_cut_int = np.floor(self.num_cut*nobs).astype(np.int64)
        else:
            self.num_cut_int = self.num_cut
            
        ncut = self.num_cut_int + 2
        fixed_cut = Binning_CDF.cut_generator(ncut, self.ylim[0], self.ylim[1], 
                                              random=False, empirical_data=train_y, 
                                              dist=self.cutpoint_distribution)
        
        fixed_cut = fixed_cut[1:-1]
        fixed_bin  = np.insert(fixed_cut, 0, self.ylim[0])
        fixed_bin  = np.append(fixed_bin, self.ylim[1])            

        self.fixed_bin = fixed_bin
            
        Train_label = np.digitize(train_y, fixed_cut)
        self.unique_label = np.unique(Train_label)
        
        self.lr_model.fit(scaled_TrainX, Train_label.reshape(-1,))
        
    def predict_cdf(self, test_x, y_grid=None, pred_margin=0.1, 
                    ngrid=1000, keep_cdf_matrix=True, 
                    overwrite_y_grid=True, keep_test_x=True):
        
        if y_grid is None:
            self.pred_lim = [self.y_min - pred_margin*self.y_range, self.y_max + pred_margin*self.y_range]
            y_grid = np.linspace(self.pred_lim[0], self.pred_lim[1], num=ngrid)
            
        if not isinstance(test_x, np.ndarray):
            test_x = np.array(test_x)
            
        if test_x.ndim <2:
            test_x = test_x.reshape(-1, self.p)
            
        y_grid = y_grid.flatten()
        
        scaled_test_x = self.x_scaler.transform(test_x)
        
        TestX_CDF_matrix = np.zeros((test_x.shape[0], len(y_grid)))
        
        if keep_test_x:
            self.test_x = test_x
            
        fixed_cut  = self.fixed_bin[1:-1]
        bin_ids    =  np.digitize(y_grid, fixed_cut)
        
        output = self.lr_model.predict_proba(scaled_test_x)
        
        for j, nbin in enumerate(bin_ids):
            
            colid = np.sum(self.unique_label<nbin)
            
            if colid == 0:
                cdf_v = output[:,colid]*(y_grid[j]-self.fixed_bin[0])/\
                (self.fixed_bin[self.unique_label[colid]+1] - self.fixed_bin[0])
            elif colid < len(self.unique_label):
                cdf_v = output[:,:colid].sum(axis=1) +\
                output[:,colid]*(y_grid[j]-self.fixed_bin[self.unique_label[colid-1]+1])/\
                (self.fixed_bin[self.unique_label[colid]+1]-self.fixed_bin[self.unique_label[colid-1]+1])
            else:
                cdf_v = 1                     
        
            TestX_CDF_matrix[:,j] = cdf_v
            
        if keep_cdf_matrix:
            self.TestX_CDF_matrix = TestX_CDF_matrix
            
        if overwrite_y_grid:
            self.y_grid = y_grid
            
        cdf_df = pd.DataFrame(TestX_CDF_matrix, columns=y_grid)
         
        return cdf_df
    
    def predict_mean(self, test_x, y_grid=None, pred_margin=0.1, ngrid=1000):
        
        cdf_matrix = self.predict_cdf(test_x, y_grid=y_grid, ngrid=ngrid, pred_margin=pred_margin,
                                      keep_cdf_matrix=False, overwrite_y_grid=True).values
                                      
        grid_width = np.diff(self.y_grid).mean()
        
        test_mean = (cdf_matrix[:,-1]*self.y_grid[-1] 
                    - cdf_matrix[:,0]*self.y_grid[0] 
                    - cdf_matrix.sum(axis=1)*grid_width)
        
        return test_mean    
    
    def predict_quantile(self, test_x, quantiles, y_grid=None, pred_margin=0.1, ngrid=1000):
        
        cdf_matrix = self.predict_cdf(test_x, y_grid=y_grid, ngrid=ngrid, pred_margin=pred_margin,
                                  keep_cdf_matrix=False, overwrite_y_grid=True).values
        
        if not isinstance(quantiles, list):
            if isinstance(quantiles, np.ndarray):
                quantiles = quantiles.tolist()
            else:
                quantiles = [quantiles]
        
        test_qtM = cdf_to_quantile(cdf_matrix, quantiles, self.y_grid)
        
        test_qt_df = pd.DataFrame(test_qtM, columns=quantiles)

        return test_qt_df 
     
    def plot_cdf(self, index=0, test_x=None, test_y=None, grid=None, pred_margin=0.1,
                 true_cdf_func=None, figsize=(12, 8), title=None):
        
        if test_x is None:
            cdf = self.TestX_CDF_matrix[index, :].copy()
            xval = self.test_x[index, :]
            grid = self.y_grid.copy()
        else:
            cdf = self.predict_cdf(test_x, y_grid=grid, pred_margin=pred_margin,
                                   keep_cdf_matrix=False, 
                                   overwrite_y_grid=True,
                                   keep_test_x=False).values.flatten()
            xval = test_x
            grid = self.y_grid.copy()
        
        cdf = cdf[grid.argsort()]
        grid.sort()
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(grid, cdf, label='predicted cdf', lw=3)
        
        if true_cdf_func is not None:
            true_cdf = true_cdf_func(xval, grid)
            ax.plot(grid, true_cdf, label='true cdf', lw=3)
            
        ax.legend(loc='best', prop={'size':16})
        
        if test_y is not None:
            if test_x is None:
                ax.axvline(x=test_y[index], color='black',  lw=3)
            else:
                ax.axvline(x=test_y, color='black', lw=3)

        if title:
            ax.set_title(title, fontsize=20)
            tlt = ax.title
            tlt.set_position([0.5, 1.02])
            
        ax.get_xaxis().set_tick_params(direction='out', labelsize=16)
        ax.get_yaxis().set_tick_params(direction='out', labelsize=16)
            
        ax.set_xlim(self.pred_lim)
        
        return ax

         
    def plot_density(self, index=0, test_x=None, test_y=None, grid=None, pred_margin=0.1, 
                     window=1, true_density_func=None, figsize=(12, 8), title=None):

        if test_x is None:
            cdf = self.TestX_CDF_matrix[index, :].copy()
            xval = self.test_x[index, :]
            grid = self.y_grid.copy()

        else:
            cdf = self.predict_cdf(test_x, y_grid=grid, 
                                   keep_cdf_matrix=False, 
                                   overwrite_y_grid=True,
                                   keep_test_x=False).values.flatten()
            xval = test_x
            grid = self.y_grid.copy()
            
            
        if len(grid) < 2*window + 1:
            raise ValueError('''The density of the most left {0} and the most right {1} 
                             grid points won't be plotted, so it requires at least 
                             {2} grid points to make density plot'''.format(window, window, 2*window + 1))        
        
        cdf = cdf[grid.argsort()]
        grid.sort()
        
        density_binwidth = grid[(2*window):] - grid[:-(2*window)]
        cdf_diff = cdf[(2*window):] - cdf[:-(2*window)]
        
        density = cdf_diff/density_binwidth
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(grid[window:-window], density, label='predicted density', lw=3)
        
        if true_density_func is not None:
            true_density = true_density_func(xval, grid[window:-window])
            ax.plot(grid[window:-window], true_density, label='true density', lw=3)
            
        ax.legend(loc='best', prop={'size':16})
            
        if title:
            ax.set_title(title, fontsize=20)
            tlt = ax.title
            tlt.set_position([0.5, 1.02])
        
        if test_y is not None:
            if test_x is None:
                ax.axvline(x=test_y[index], color='black',  lw=3)
            else:
                ax.axvline(x=test_y, color='black', lw=3)
            
        ax.get_xaxis().set_tick_params(direction='out', labelsize=16)
        ax.get_yaxis().set_tick_params(direction='out', labelsize=16)
        
        ax.set_xlim(self.pred_lim)
        
        return ax          
            
    def evaluate(self, test_x, test_y, y_grid=None, pred_margin=0.1, 
                 ngrid=1000, quantiles=None, interval=None, mode='CRPS'):
        
        if mode == 'QuantileLoss' and quantiles is not None:
            quantile_matrix = self.predict_quantile(test_x, quantiles,
                                                    y_grid=y_grid, 
                                                    pred_margin=pred_margin,
                                                    ngrid=ngrid).values
            test_score = evaluate_quantile_loss(quantile_matrix, test_y, quantiles)
        else:
            cdf_matrix = self.predict_cdf(test_x, y_grid=y_grid, 
                                          pred_margin=pred_margin,
                                          ngrid=ngrid).values                   
            if mode == 'CRPS':
                test_score = evaluate_crps(cdf_matrix, test_y, self.y_grid)
            elif mode == 'RMSE':            
                test_score = evaluate_rmse(cdf_matrix, test_y, self.y_grid)
            elif mode == 'Coverage' and interval is not None:
                test_score = evaluate_coverage(cdf_matrix, test_y, interval, self.y_grid)
        
        return test_score
        
        