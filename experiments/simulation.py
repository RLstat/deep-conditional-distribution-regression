# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 00:02:32 2019

@author: Rui Li
"""

import os
import csv
import numpy as np
from dcdr.deep_hist import Binning_CDF 
from qrf.qrf_cdf import QRFCDF
from logistic.logistic_cdf import LogisticRegressionCDF as LR
from simulator.data_sims import data_simulator1, data_simulator2, data_simulator3, data_simulator4
from simulator.data_sims import generate_unique_seeds, check_seed_exist

## This is an example of simulation using this module
## You can change to a different data_simulator for the simulation studies.
## You can also use real data to replace the TrainX, TrainY, TestX, TestY.
## You can also change some of the hyperparameters below. 
ntrain = 6000
ntest = 1000
p = 10
data_simulator = data_simulator1
init_seed=1234
num_cut = 0.1
hidden_list = [100,100,100]
dropout_list = [0.5,0.5,0.5]
histogram_bin = 'fixed'
loss_model = 'multi-binary'
sim_iter = 100
target_dir = 'your-file-directory-to-save'
filename = 'simulation_2.csv'
coverage_list = [[0.05, 0.95]]
quantiles = np.linspace(0.01, 0.99, num=99)

seedlist = generate_unique_seeds(init_seed, sim_iter)

csv_file_path = os.path.join(target_dir, filename)

for i, seeding in enumerate(seedlist):
    
    if check_seed_exist(csv_file_path, 'Random Seed', seeding):
        continue 
    
    TrainX, TrainY, TestX, TestY = data_simulator(ntrain, ntest, p, seeding)
    
    dcdr_model = Binning_CDF(num_cut=num_cut, hidden_list=hidden_list,
                             histogram_bin=histogram_bin, dropout_list=dropout_list,
                             seeding=seeding, loss_model=loss_model, 
                             niter=20)
    
    dcdr_model.fit_cdf(TrainX, TrainY, batch_size=32, merge_empty_bin=True)
    
    dcdr_crps = dcdr_model.evaluate(TestX, TestY, mode='CRPS')
    
    dcdr_aqtl = dcdr_model.evaluate(TestX, TestY, quantiles=quantiles, 
                                    mode='QuantileLoss')  
    
    dcdr_cover = [[] for _ in coverage_list]
    
    for ncov, interval in enumerate(coverage_list):
        cover_percent = dcdr_model.evaluate(TestX, TestY,interval=interval, mode='Coverage')
        dcdr_cover[ncov].append(cover_percent)
        
    qrf_model = QRFCDF(random_state=seeding, min_samples_split=10, n_estimators=500)
    
    qrf_model.fit_cdf(TrainX, TrainY)
    
    qrf_crps = qrf_model.evaluate(TestX, TestY, mode='CRPS')
    
    qrf_aqtl = qrf_model.evaluate(TestX, TestY, quantiles=quantiles, 
                                    mode='QuantileLoss')  
    
    qrf_cover = [[] for _ in coverage_list]
    
    for ncov, interval in enumerate(coverage_list):
        cover_percent = qrf_model.evaluate(TestX, TestY, interval=interval,mode='Coverage')
        qrf_cover[ncov].append(cover_percent)
        
    lr_model = LR(num_cut=num_cut, seeding=seeding)
    
    lr_model.fit_cdf(TrainX, TrainY)
    
    lr_crps = lr_model.evaluate(TestX, TestY, mode='CRPS')
    
    lr_aqtl = lr_model.evaluate(TestX, TestY, quantiles=quantiles, 
                                    mode='QuantileLoss')  
    
    lr_cover = [[] for _ in coverage_list]
    
    for ncov, interval in enumerate(coverage_list):
        cover_percent = lr_model.evaluate(TestX, TestY, interval=interval, mode='Coverage')
        lr_cover[ncov].append(cover_percent)   
    
    
    ##
    # Save results
    ##  
    header = ['model', 'result', 'eval_metric', 'ncut_points', 'Random Seed']

    dcdr_crps_results = ['dcdr',dcdr_crps, 'CRPS', num_cut, seeding]
    
    dcdr_aqtl_results = ['dcdr', dcdr_aqtl, 'QuantileLoss', num_cut, seeding]
    
    dcdr_cover_results = [[] for _ in range(len(coverage_list))]
    for i in range(len(coverage_list)):
        dcdr_cover_results[i].extend(['dcdr', dcdr_cover[i][0], coverage_list[i], num_cut, seeding])

    qrf_crps_results = ['qrf',qrf_crps, 'CRPS', num_cut, seeding]
    
    qrf_aqtl_results = ['qrf', qrf_aqtl, 'QuantileLoss', num_cut, seeding]
    
    qrf_cover_results = [[] for _ in range(len(coverage_list))]
    for i in range(len(coverage_list)):
        qrf_cover_results[i].extend(['qrf', qrf_cover[i][0], coverage_list[i], num_cut, seeding])
        
    lr_crps_results = ['lr',lr_crps, 'CRPS', num_cut, seeding]
    
    lr_aqtl_results = ['lr', lr_aqtl, 'QuantileLoss', num_cut, seeding]
    
    lr_cover_results = [[] for _ in range(len(coverage_list))]
    for i in range(len(coverage_list)):
        lr_cover_results[i].extend(['lr', lr_cover[i][0], coverage_list[i], num_cut, seeding])
        
    try:
        output_file = open(csv_file_path, 'a',newline='')
    except EnvironmentError as exception_:
        print (exception_)
        print ('Failed to open or create ' + csv_file_path + '!')
    
    if os.stat(csv_file_path).st_size > 0:
        csv.writer(output_file).writerow(dcdr_crps_results)
        csv.writer(output_file).writerow(dcdr_aqtl_results)
        for i in range(len(coverage_list)):
            csv.writer(output_file).writerow(dcdr_cover_results[i])
        csv.writer(output_file).writerow(qrf_crps_results)
        csv.writer(output_file).writerow(qrf_aqtl_results)
        for i in range(len(coverage_list)):
            csv.writer(output_file).writerow(qrf_cover_results[i])
        csv.writer(output_file).writerow(lr_crps_results)
        csv.writer(output_file).writerow(lr_aqtl_results)
        for i in range(len(coverage_list)):
            csv.writer(output_file).writerow(lr_cover_results[i])            
    else:
        csv.writer(output_file).writerow(header)
        csv.writer(output_file).writerow(dcdr_crps_results)
        csv.writer(output_file).writerow(dcdr_aqtl_results)
        for i in range(len(coverage_list)):
            csv.writer(output_file).writerow(dcdr_cover_results[i])
        csv.writer(output_file).writerow(qrf_crps_results)
        csv.writer(output_file).writerow(qrf_aqtl_results)
        for i in range(len(coverage_list)):
            csv.writer(output_file).writerow(qrf_cover_results[i])
        csv.writer(output_file).writerow(lr_crps_results)
        csv.writer(output_file).writerow(lr_aqtl_results)
        for i in range(len(coverage_list)):
            csv.writer(output_file).writerow(lr_cover_results[i])   
        
    output_file.close()

