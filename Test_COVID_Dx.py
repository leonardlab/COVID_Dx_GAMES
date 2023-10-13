#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:34:00 2020

@author: kate
"""
#Package imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
      
#GAMES imports
import Settings_COVID_Dx
from Solvers_COVID_Dx import calcRsq
from Run_COVID_Dx import solveAll
from Saving_COVID_Dx import createFolder
from Analysis_Plots import plotModelingObjectives123, plotModelingObjectives456, parityPlot
from Analysis_Plots import plotModelingObjectives123, plotModelingObjectives456, parityPlot

#Define settings
conditions_dictionary, initial_params_dictionary, data_dictionary = Settings_COVID_Dx.init()
full_path = conditions_dictionary["directory"]
model = conditions_dictionary["model"]
data = conditions_dictionary["data"]
error = data_dictionary["error"]
exp_data = data_dictionary["exp_data"]
timecourses_err = data_dictionary["timecourses_err"]
timecourses_exp = data_dictionary["timecourses"]
x = data_dictionary["x_vals"]
problem_free = conditions_dictionary["problem"]
bounds = problem_free['bounds']

#Note that this path needs to be updated before running
plt.style.use('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/paper.mplstyle.py')
    

def testSingleSet(p: list) -> None:
    
    ''' 
    Simulates the entire dataset for a single parameter set
        
    Args: 
        p: a list of floats defining the parameter values for all 
            potentially free parameters (Settings_COVID_Dx.py
            conditions_dictionary["p_all"])
           
    Returns: None
    
    Figures: 
        'FIT TO TRAINING DATA.svg':
            parity plot showing agreement between xperimental and
                simulated data

        'Modeling objectives 123 sim.svg':
            histograms of summary metrics for simulations run with
            parameter set p

        'MODELING OBJECTIVE 4.svg': plot showing readout dynamics
            for the experimental and simulated data for the data set
            involved with modeling objective 4 for parameter set p

        'MODELING OBJECTIVE 5.svg': plot showing readout dynamics for
            the experimental and simulated data for the data set 
            involved with modeling objective 5 for parameter set p

        'MODELING OBJECTIVE 6.svg': plot showing readout dynamics for
            the experimental and simulated data for the data set 
            involved with modeling objective 6 for parameter set p
    '''
    
    os.chdir(full_path)
    sub_folder_name = './TEST'
    createFolder('./' + sub_folder_name)
    os.chdir('./' + sub_folder_name)
    start_time = datetime.datetime.now()
    doses, solutions, chi2, df_sim, df_all_states = solveAll(p, exp_data, 'all states')
    stop_time = datetime.datetime.now()
    elapsed_time = stop_time - start_time
    elapsed_time_total = round(elapsed_time.total_seconds(), 1)
    print(round(elapsed_time_total, 3))

    R_sq = calcRsq(solutions, exp_data)  
    is_negative, full_solutions = solveAll(p, exp_data, 'check negative')
    print(is_negative)
    negative_vals = full_solutions[np.where(full_solutions < 0)]
    print(negative_vals)
    parityPlot(solutions, exp_data, data)

    #Plot modeling objectives
    plotModelingObjectives123(solutions)
    plotModelingObjectives456(df_sim)
    
    print('*******')
    print('R2: ' + str(np.round(R_sq, 5)))
    print('chi2: ' + str(np.round(chi2, 5)))
    print('*******')

############################################
#model A rep 1 best fit high tol
# p = [0.001897298, 7712.751409, 0.036062664, 0.823477247, 0.01953665, 0, 0]
#model A rep 1 best fit low tol
# p = [0.466701312, 10299.00626, 0.036257637, 20.40802615, 0.013807927, 0, 0]

#model A rep 2 best fit high tol
# p = [0.694384997, 3160.518228, 0.037335763, 9.732622387, 0.008703705, 0, 0]
#model A rep 2 best fit low tol
# p = [0.000904594, 2514.267688, 0.036000003, 7.528500033, 0.017006437, 0, 0]

#model A rep 3 best fit high tol
# p =[0.013285913, 12297.96647, 0.314823239, 0.003216157, 1.330392949, 0, 0]
#model A rep 3 best fit low tol
# p = [1.264748656, 3407.227184, 0.036028742, 0.017441602, 0.976749751, 0, 0]

############################################
#model B rep 1 best fit high tol
# p = [0.000444994, 14762.07943, 0.660385544, 0.083092643, 0.099344848, 34.7484005, 17.54455931]
#model B rep 1 best fit low tol
# p = [0.000536105, 7115.863562, 0.892292063, 0.066547512, 0.114401689, 42.6583508, 16.62590786]

#model B rep 2 best fit high tol
# p = [0.027556769, 331.3078072, 1.398234485, 0.009171153, 2677.166783, 5.278146074, 61.79702582]
#model B rep 2 best fit low tol
# p = [0.001723086, 128.9759141, 0.251419297, 0.022706194, 2690.368918, 36.30447526, 39.24078673]

#model B rep 3 best fit high tol
# p = [0.030431954, 510.264659, 72.24421616, 0.000831638, 59.86896595, 79.88669589, 75.55711055]
#model B rep 3 best fit low tol
# p = [1.351510954, 13705.45827, 0.093054739, 0.006773398, 3.607886549, 1.50686083, 148.2506733]

############################################
#model C rep 1 best fit run 1 high tol
# p = [0.000900988, 3950.759594, 45.42883694, 0.039433974, 0.387098088, 60.51203729, 15.15736952]
#model C rep 1 best fit run 1 low  tol 
# p = [0.000290052, 56.67368127, 1282.861975, 0.092615409, 3933.113423, 53.19481399, 6.448545639]

#model C rep 1 best fit run 2 high tol
# p = [0.00031178, 78.91898242, 1194.417613, 0.074241899, 7.909074516, 55.58341311, 6.745005326]
#model C rep 1 best fit run 2 low tol 
# p = [0.000384312, 75.38073202, 1145.888736, 0.062734604, 7.498278516, 55.47333899, 6.805959386]

#model C rep 3 best fit low tol
# p = [8.676e-05, 20630.70109358, 45.58509923, 0.1057036, 220.1036333, 44.4359737, 17.92897678]

#model C rep 2 best fit low tol
# p = [0.000229349, 2907.171643, 1.661924708, 315.1002541, 1.006983458, 114.912718, 26.30885998]

############################################
#model D rep 1 best fit high tol
# p = [0.00063618, 239.9315589, 858.2136969, 0.027772651, 1.753931699, 10.64957523, 42.38269934, 61.82812743, 8.241330405]
#model D rep 1 best fit low tol
# p = [0.000224063, 136.0589787, 1151.286829, 0.09603252, 1.659857597, 12.48387036, 55.71209137, 56.73099109, 6.906538862]

#model D rep 2 best fit high tol
# p = [1.51719E-05, 12223.96888, 315.8195865, 0.381765754, 2.197307165, 29.34473237, 68.69366218, 101.5269246, 17.96639629]
#model D rep 2 best fit low tol
p = [2.22994E-05, 8940.243435, 226.2897324, 232.9366873, 1.749944885, 22.66728787, 4.675577757, 97.309157, 19.21663556]

#model D rep 3 best fit high tol
# p = [0.00012593, 30452.9863669, 41.57403523, 0.07926811, 1.07134915, 14.7113393, 0.62647773, 46.40684061, 18.541746]
#model D rep 3 best fit low tol
# p = [6.38185E-05, 28934.65508, 119.187498, 0.077888022, 1.627413157, 23.55089534, 22.28442186, 50.03161646, 17.06654392]

# p = [0.001723086, 128.9759141, 0.251419297, 0.022706194, 2690.368918, 36.30447526, 39.24078673]

#model B rep 2 with rep 1 opt LM
# p = [0.00031625, 771.62538369, 0.04050095, 1.0522019, 0.22352758, 24.85053933, 65.76394723]

#model A rep 2 with rep 1 opt LM
# p = [0.39922319, 4836.19487492, 0.03600005, 110.2633939, 0.00818332, 0.0, 0.0]

#model A rep 3 with rep 1 opt LM
p = [0.56054578, 7996.04055801, 0.03600003, 27.68577631, 0.00786939, 0.0, 0.0]

testSingleSet(p)   


   


