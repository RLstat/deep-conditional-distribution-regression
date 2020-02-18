# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 00:29:53 2019

@author: RLstat
"""

import os
import numpy as np
import pandas as pd
import csv
from scipy.stats import skewnorm
from dcdr.utils import evaluate_crps, \
evaluate_quantile_loss, evaluate_rmse, evaluate_coverage


def data_simulator1(ntrain, ntest, p, seeding, return_b=False, return_mean=False):
    nobs = ntrain + ntest
    np.random.seed(seeding)
    X = np.random.normal(size=(nobs, p))
    b1 = np.random.normal(size=p)
    b2 = np.random.normal(size=p)/5
    if not return_mean:
        Y = np.matmul(X,b1) + \
        np.exp(np.matmul(X,b2))*np.random.normal(size=nobs)
    else:
        Y = np.matmul(X, b1)
        
    Y = Y.reshape(-1,1)
    
    TrainX = X[:ntrain,:]
    TrainY = Y[:ntrain,:]
    
    TestX  = X[ntrain:,:]
    TestY  = Y[ntrain:,:]
    
    if return_b:
        return TrainX, TrainY, TestX, TestY, b1, b2
    else:
        return TrainX, TrainY, TestX, TestY


def conditional_sample_func1(X, b1, b2, nobs, seeding=1234):
    
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    np.random.seed(seed=seeding)
    samples = np.matmul(X,b1) + \
    np.exp(np.matmul(X,b2))*np.random.normal(size=nobs)  
    
    return samples


def data_simulator2(ntrain, ntest, p, seeding, return_mean=False):
    nobs = ntrain + ntest    
    np.random.seed(seeding)
    X = np.random.uniform(size=(nobs, p))
    ind = np.random.binomial(1,0.5,size=nobs)
    if not return_mean:
        Y = (10*np.sin(2*np.pi*X[:,0]*X[:,1]) + 
                  10*X[:,3] + np.random.normal(scale=1.5, size=nobs))*ind + \
                  (20*np.square((X[:,2] - 0.5)) + 5*X[:,4] + 
                   np.random.normal(size=nobs))*(1 - ind)
    else:
        Y = (10*np.sin(2*np.pi*X[:,0]*X[:,1]) + 
                  10*X[:,3])*0.5 + \
                  (20*np.square((X[:,2] - 0.5)) + 5*X[:,4])*0.5
                  
    Y = Y.reshape(-1,1)
    
    TrainX = X[:ntrain,:]
    TrainY = Y[:ntrain,:]
    
    TestX  = X[ntrain:,:]
    TestY  = Y[ntrain:,:]
    
    return TrainX, TrainY, TestX, TestY


def density_func2(X, grid):  
    from scipy.stats import norm
    
    loc1 = 10*np.sin(2*np.pi*X[0]*X[1]) + 10*X[3]
    scale1 = 1.5
    
    loc2 = 20*np.square((X[2] - 0.5)) + 5*X[4]
    scale2 = 1
    
    rv1 = norm(loc=loc1, scale=scale1)
    rv2 = norm(loc=loc2, scale=scale2)
    return rv1.pdf(grid)*0.5 + rv2.pdf(grid)*0.5 

def cdf_func2(X, grid):
    from scipy.stats import norm
    
    loc1 = 10*np.sin(2*np.pi*X[0]*X[1]) + 10*X[3]
    scale1 = 1.5
    
    loc2 = 20*np.square((X[2] - 0.5)) + 5*X[4]
    scale2 = 1
    
    rv1 = norm(loc=loc1, scale=scale1)
    rv2 = norm(loc=loc2, scale=scale2)
    return rv1.cdf(grid)*0.5 + rv2.cdf(grid)*0.5

def conditional_sample_func2(X, nobs, seeding=1234):
    from scipy.stats import norm
    
    loc1 = 10*np.sin(2*np.pi*X[0]*X[1]) + 10*X[3]
    scale1 = 1.5
    
    loc2 = 20*np.square((X[2] - 0.5)) + 5*X[4]
    scale2 = 1
    
    rv1 = norm(loc=loc1, scale=scale1)
    rv2 = norm(loc=loc2, scale=scale2)
    
    np.random.seed(seed=seeding)
    ind = np.random.binomial(1,0.5,size=nobs)
    
    s1 = rv1.rvs(size=nobs)
    s2 = rv2.rvs(size=nobs)
    
    samples = s1*ind + s2*(1-ind)
    
    return samples


def data_simulator3(ntrain, ntest, p, seeding, return_mean=False):
    nobs = ntrain + ntest    
    np.random.seed(seeding)
    X = np.random.uniform(high = 10, size=(nobs, p))
    ind = np.random.binomial(1,0.5,size=nobs)
    if not return_mean:
        Y = (np.sin(X[:,0]) + np.random.normal(scale=0.3, size=nobs))*ind + \
        (2*np.sin(1.5*X[:,0] + 1) + np.random.normal(scale=0.8, size=nobs))*(1 - ind)
    else:
        Y = (np.sin(X[:,0]))*0.5 + \
        (2*np.sin(1.5*X[:,0] + 1))*0.5
        
    Y = Y.reshape(-1,1)
    
    TrainX = X[:ntrain,:]
    TrainY = Y[:ntrain,:]
    
    TestX  = X[ntrain:,:]
    TestY  = Y[ntrain:,:]
    
    return TrainX, TrainY, TestX, TestY

def density_func3(X, grid):  
    from scipy.stats import norm
    
    loc1 = np.sin(X[0])
    scale1 = 0.3
    
    loc2 = 2*np.sin(1.5*X[0] + 1)
    scale2 = 0.8
    
    rv1 = norm(loc=loc1, scale=scale1)
    rv2 = norm(loc=loc2, scale=scale2)
    return rv1.pdf(grid)*0.5 + rv2.pdf(grid)*0.5

def cdf_func3(X, grid):  
    from scipy.stats import norm
    
    loc1 = np.sin(X[0])
    scale1 = 0.3
    
    loc2 = 2*np.sin(1.5*X[0] + 1)
    scale2 = 0.8
    
    rv1 = norm(loc=loc1, scale=scale1)
    rv2 = norm(loc=loc2, scale=scale2)
    return rv1.cdf(grid)*0.5 + rv2.cdf(grid)*0.5


def conditional_sample_func3(X, nobs, seeding=1234):
    from scipy.stats import norm
    
    if (not isinstance(X, np.ndarray)) or (not isinstance(X, list)):
        X = [X]
        
    loc1 = np.sin(X[0])
    scale1 = 0.3
    
    loc2 = 2*np.sin(1.5*X[0] + 1)
    scale2 = 0.8
    
    rv1 = norm(loc=loc1, scale=scale1)
    rv2 = norm(loc=loc2, scale=scale2)
    
    np.random.seed(seed=seeding)
    ind = np.random.binomial(1,0.5,size=nobs)
    
    s1 = rv1.rvs(size=nobs)
    s2 = rv2.rvs(size=nobs)
    
    samples = s1*ind + s2*(1-ind)
    
    return samples


def data_simulator4(ntrain, ntest, p, seeding, return_mean=False):
    nobs = ntrain + ntest    
    np.random.seed(seeding)
    X = np.random.uniform(size=(nobs, p))
    if not return_mean:
        Y = (10*np.sin(2*np.pi*X[:,0]*X[:,1]) 
        + 10*X[:,3] + 20*np.square((X[:,2] - 0.5)) 
        + 5*X[:,4]) + skewnorm.rvs(-5, size=nobs)
    else:
        Y = (10*np.sin(2*np.pi*X[:,0]*X[:,1]) 
        + 10*X[:,3] + 20*np.square((X[:,2] - 0.5)) 
        + 5*X[:,4])       
    
    Y = Y.reshape(-1,1)
    
    TrainX = X[:ntrain,:]
    TrainY = Y[:ntrain,:]
    
    TestX  = X[ntrain:,:]
    TestY  = Y[ntrain:,:]
    
    return TrainX, TrainY, TestX, TestY

def cdf_func4(X, grid):
    
    loc1 = (10*np.sin(2*np.pi*X[0]*X[1]) + 10*X[3] + 20*np.square((X[2]-0.5))
    + 5*X[4])
    
    rv = skewnorm(a=-5, loc=loc1)
    return rv.cdf(grid)


def conditional_sample_func4(X, nobs, seeding=1234):
    
    loc1 = (10*np.sin(2*np.pi*X[0]*X[1]) + 10*X[3] + 20*np.square((X[2]-0.5))
    + 5*X[4])
    
    np.random.seed(seed=seeding)
    samples = loc1 + skewnorm.rvs(-5, size=nobs)
    
    return samples


def check_seed_exist(csv_file_path, seed_col_name, seeding):
    
    if os.path.isfile(csv_file_path):
        result_df = pd.read_csv(csv_file_path)
        seed_existed = result_df.loc[:,seed_col_name].unique() 
        
        if seeding in seed_existed:
            return True
        
    return False


def generate_unique_seeds(init_seed, sim_iter):
    np.random.seed(init_seed)
    seedlist = np.ceil(np.random.uniform(
            size=sim_iter)*1000000).astype(np.int32)
    
    seedlist = np.unique(seedlist)
    
    while len(seedlist) < sim_iter:
        newseed = np.ceil(np.random.uniform(size=1)*1000000).astype(np.int32)
        if newseed not in seedlist:
            seedlist = np.append(seedlist, newseed)
            
    return seedlist
    

def data_sim_wrapper(ntrain, ntest, p, filename, target_dir, coverage_list,
                     quantiles, data_generator, model_class, init_kwargs, 
                     fit_kwargs, y_grid=None, pred_margin=0.1, 
                     ngrid=1000, init_seed=1234, sim_iter=100):
    
    seedlist = generate_unique_seeds(init_seed, sim_iter)
    
    csv_file_path = os.path.join(target_dir, filename)
    
    for i, seeding in enumerate(seedlist):
        
        if check_seed_exist(csv_file_path, 'Random Seed', seeding):
            continue 
        
        TrainX, TrainY, TestX, TestY = data_generator(ntrain, ntest, p, seeding)
        
        init_kwargs['seeding'] = seeding
        
        model_obj = model_class(**init_kwargs)
        
        model_obj.fit_cdf(TrainX, TrainY, **fit_kwargs)
        
        crps_score = model_obj.evaluate(TestX, TestY, mode='CRPS', 
                                        y_grid=y_grid, pred_margin=pred_margin, 
                                        ngrid=ngrid)
        
        quantile_loss = model_obj.evaluate(TestX, TestY, quantiles=quantiles, 
                                            mode='QuantileLoss', y_grid=y_grid, 
                                            pred_margin=pred_margin, ngrid=ngrid)  
        
        emp_cover = [[] for _ in coverage_list]
        
        for ncov, interval in enumerate(coverage_list):
            cover_percent = model_obj.evaluate(TestX, TestY, mode='Coverage', 
                                               interval=interval, y_grid=y_grid, 
                                               pred_margin=pred_margin, ngrid=ngrid)
            emp_cover[ncov].append(cover_percent)
            
        rmse = model_obj.evaluate(TestX, TestY, mode='RMSE', y_grid=y_grid, 
                                  pred_margin=pred_margin, ngrid=ngrid)
        
        
        ##
        # Save results
        ##
        ncut_points = model_obj.num_cut_int
        
        header = ['result', 'eval_metric', 'ncut_points', 'Random Seed']
    
        crps_results = [crps_score, 'CRPS', ncut_points, seeding]
        
        qt_results = [quantile_loss, 'QuantileLoss', ncut_points, seeding]
        
        coverage_results = [[] for _ in range(len(coverage_list))]
        for i in range(len(coverage_list)):
            coverage_results[i].extend([emp_cover[i][0], coverage_list[i], ncut_points, seeding])
            
        rmse_results = [rmse,'RMSE', ncut_points, seeding]
        
        try:
            output_file = open(csv_file_path, 'a',newline='')
        except EnvironmentError as exception_:
            print (exception_)
            print ('Failed to open or create ' + csv_file_path + '!')
        
        if os.stat(csv_file_path).st_size > 0:
            csv.writer(output_file).writerow(crps_results)
            csv.writer(output_file).writerow(qt_results)
            for i in range(len(coverage_list)):
                csv.writer(output_file).writerow(coverage_results[i])
            csv.writer(output_file).writerow(rmse_results)
        else:
            csv.writer(output_file).writerow(header)
            csv.writer(output_file).writerow(crps_results)
            csv.writer(output_file).writerow(qt_results)
            for i in range(len(coverage_list)):
                csv.writer(output_file).writerow(coverage_results[i])
            csv.writer(output_file).writerow(rmse_results)
            
        output_file.close()
        
        
        
        
    