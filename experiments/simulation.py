# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 00:02:32 2019

@author: RLstat
"""

import sys
sys.path.insert(0, '/home/ubuntu/deep-conditional-distribution-regression')

import os
import csv
import gc
import numpy as np
from keras import backend
from dcdr.deep_hist import Binning_CDF 
from qrf.qrf_cdf import QRFCDF
from logistic.logistic_cdf import LogisticRegressionCDF as LR
from simulator.data_sims import data_simulator1, data_simulator2, data_simulator3, data_simulator4
from simulator.data_sims import generate_unique_seeds, check_seed_exist
from timeit import default_timer

def write_to_csv(csv_file_path, header, crps_results, aqtl_results, cover_results):
    try:
        output_file = open(csv_file_path, 'a',newline='')
    except EnvironmentError as exception_:
        print (exception_)
        print ('Failed to open or create ' + csv_file_path + '!')
    
    if os.stat(csv_file_path).st_size > 0:
        csv.writer(output_file).writerow(crps_results)
        csv.writer(output_file).writerow(aqtl_results)
        for i in range(len(cover_results)):
            csv.writer(output_file).writerow(cover_results[i])           
    else:
        csv.writer(output_file).writerow(header)
        csv.writer(output_file).writerow(crps_results)
        csv.writer(output_file).writerow(aqtl_results)
        for i in range(len(cover_results)):
            csv.writer(output_file).writerow(cover_results[i])    
        
    output_file.close()
    
def unpack_setting_to_str(settings):
    setting_str = ''
    for key, val in settings.items(): 
        setting_str += f'{key}({val})'
    
    return setting_str
    

## This is an example of simulation using this module
## You can change to a different data_simulator for the simulation studies.
## You can also use real data to replace the TrainX, TrainY, TestX, TestY.
## You can also change some of the hyperparameters below. 

init_seed=1234
num_cut_list = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]
hidden_list = [100,100,100]
dropout_list = [0.5,0.5,0.5]
niter = 20 # only for ensemble method
histogram_bin_list = ['fixed']
loss_model_list = ['multi-binary', 'multi-class']
sim_iter = 3
target_dir = '/home/ubuntu/simulations'
coverage_list = [[0.05, 0.95]]
quantiles = np.linspace(0.01, 0.99, num=99)

seedlist = generate_unique_seeds(init_seed, sim_iter)

simulation_setting_1 = dict(
    ntrain=6000,
    ntest=1000,
    p=5,
)

simulation_setting_2 = dict(
    ntrain=6000,
    ntest=1000,
    p=10,
)

simulation_setting_3 = dict(
    ntrain=6000,
    ntest=1000,
    p=1,
)

simulation_setting_4 = dict(
    ntrain=6000,
    ntest=1000,
    p=10,
)

simulation_settings = [
    (data_simulator1, simulation_setting_1), 
    (data_simulator2, simulation_setting_2),
    (data_simulator3, simulation_setting_3),
    (data_simulator4, simulation_setting_4)
]


for data_simulator, simulator_setting in simulation_settings:
    setting_str = unpack_setting_to_str(simulator_setting)
    filename = f'{data_simulator.__name__}_{setting_str}.csv'
    csv_file_path = os.path.join(target_dir, filename)
    for i, seeding in enumerate(seedlist):
        if check_seed_exist(csv_file_path, 'Random Seed', seeding):
            continue 

        simulator_setting.update(seeding=seeding)
        TrainX, TrainY, TestX, TestY = data_simulator(**simulator_setting)

        # Start of dcdr
        for num_cut in num_cut_list:
            for histogram_bin in histogram_bin_list:
                for loss_model in loss_model_list:
                    dcdr_st = default_timer()
                    dcdr_model = Binning_CDF(num_cut=num_cut, hidden_list=hidden_list,
                                             histogram_bin=histogram_bin, dropout_list=dropout_list,
                                             seeding=seeding, loss_model=loss_model, 
                                             niter=niter)

                    dcdr_model.fit_cdf(TrainX, TrainY, batch_size=32, merge_empty_bin=True)

                    dcdr_crps = dcdr_model.evaluate(TestX, TestY, mode='CRPS')

                    dcdr_aqtl = dcdr_model.evaluate(TestX, TestY, quantiles=quantiles, 
                                                    mode='QuantileLoss')  

                    dcdr_cover = [[] for _ in coverage_list]

                    for ncov, interval in enumerate(coverage_list):
                        cover_percent = dcdr_model.evaluate(TestX, TestY,interval=interval, mode='Coverage')
                        dcdr_cover[ncov].append(cover_percent)

                    dcdr_model.clear_model_memory()
                    
                    del dcdr_model
                    backend.clear_session()
                    gc.collect()
                    gc.collect()
                    
                    dcdr_elapsed = default_timer()-dcdr_st

                    model_name = '_'.join(['dcdr', loss_model, histogram_bin])
                    
                    if histogram_bin == 'random':
                        model_name += f'_{niter}'

                    header = ['model', 'result', 'eval_metric', 'ncut_points', 'Random Seed', 'time_elapsed']

                    dcdr_crps_results = [model_name, dcdr_crps, 'CRPS', num_cut, seeding, dcdr_elapsed]

                    dcdr_aqtl_results = [model_name, dcdr_aqtl, 'QuantileLoss', num_cut, seeding, dcdr_elapsed]

                    dcdr_cover_results = [[] for _ in range(len(coverage_list))]
                    for i in range(len(coverage_list)):
                        dcdr_cover_results[i].extend([model_name, dcdr_cover[i][0], 
                                                      coverage_list[i], num_cut, seeding, dcdr_elapsed])

                    write_to_csv(csv_file_path, header, dcdr_crps_results, dcdr_aqtl_results, dcdr_cover_results)
            
            # Start of logistic regression
            lr_st = default_timer()
            lr_model = LR(num_cut=num_cut, seeding=seeding)

            lr_model.fit_cdf(TrainX, TrainY)

            lr_crps = lr_model.evaluate(TestX, TestY, mode='CRPS')

            lr_aqtl = lr_model.evaluate(TestX, TestY, quantiles=quantiles, 
                                            mode='QuantileLoss')  

            lr_cover = [[] for _ in coverage_list]

            for ncov, interval in enumerate(coverage_list):
                cover_percent = lr_model.evaluate(TestX, TestY, interval=interval, mode='Coverage')
                lr_cover[ncov].append(cover_percent)  
                
            del lr_model
            gc.collect()
            gc.collect()

            lr_elapsed = default_timer()-lr_st

            lr_crps_results = ['lr', lr_crps, 'CRPS', num_cut, seeding, lr_elapsed]

            lr_aqtl_results = ['lr', lr_aqtl, 'QuantileLoss', num_cut, seeding, lr_elapsed]

            lr_cover_results = [[] for _ in range(len(coverage_list))]
            for i in range(len(coverage_list)):
                lr_cover_results[i].extend(['lr', lr_cover[i][0], coverage_list[i], num_cut, seeding, lr_elapsed])

            write_to_csv(csv_file_path, header, lr_crps_results, lr_aqtl_results, lr_cover_results)

        # Start of QRF
        qrf_st = default_timer()
        qrf_model = QRFCDF(random_state=seeding, min_samples_split=10, n_estimators=500)

        qrf_model.fit_cdf(TrainX, TrainY)

        qrf_crps = qrf_model.evaluate(TestX, TestY, mode='CRPS')

        qrf_aqtl = qrf_model.evaluate(TestX, TestY, quantiles=quantiles, 
                                        mode='QuantileLoss')  

        qrf_cover = [[] for _ in coverage_list]

        for ncov, interval in enumerate(coverage_list):
            cover_percent = qrf_model.evaluate(TestX, TestY, interval=interval,mode='Coverage')
            qrf_cover[ncov].append(cover_percent)
            
        del qrf_model
        gc.collect()
        gc.collect()

        qrf_elapsed = default_timer()-qrf_st

        qrf_crps_results = ['qrf',qrf_crps, 'CRPS', num_cut, seeding, qrf_elapsed]

        qrf_aqtl_results = ['qrf', qrf_aqtl, 'QuantileLoss', num_cut, seeding, qrf_elapsed]

        qrf_cover_results = [[] for _ in range(len(coverage_list))]
        for i in range(len(coverage_list)):
            qrf_cover_results[i].extend(['qrf', qrf_cover[i][0], coverage_list[i], num_cut, seeding, qrf_elapsed])

        write_to_csv(csv_file_path, header, qrf_crps_results, qrf_aqtl_results, qrf_cover_results)
        
        gc.collect()
        gc.collect()


