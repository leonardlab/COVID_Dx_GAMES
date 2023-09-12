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
from Analysis_Plots import plotModelingObjectives123, plotModelingObjectives456, parityPlot, plot_all_states, plot_states_RHS

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
# df_data = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED DATA EXP.pkl')
# df_error = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED DATA ERR.pkl')
plt.style.use('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/paper.mplstyle.py')
    

def testSingleSet(p):
    ''' 
    Purpose: Simulate the entire dataset for a single parameter set
        
    Inputs: 
        p: list of floats, each float corresponds to a parameter value 
            (length = # parameters in p_all). 
            Parameter labels and order defined in init() (in Settings.py)   
           
    Output: None
    
    Figures: FIT TO TRAINING DATA.svg - parity plot showing agreement between xperimental and simulated data
             Modeling objectives 123 sim.svg - histograms of summary metrics for simulations run with parameter set, p
             MODELING OBJECTIVE 4.svg - plot showing readout dynamics for the experimental and simulated data 
                 for the data set involved with modeling objective 4 (run with parameter set, p)
             MODELING OBJECTIVE 5.svg - plot showing readout dynamics for the experimental and simulated data 
                 for the data set involved with modeling objective 5 (run with parameter set, p)
             MODELING OBJECTIVE 6.svg - plot showing readout dynamics for the experimental and simulated data 
                 for the data set involved with modeling objective 6 (run with parameter set, p)
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
    
    # filename = 'All_states_solutions_check_negative'
    # with pd.ExcelWriter(filename + '.xlsx') as writer:  # doctest: +SKIP
    #     df_all_states.to_excel(writer, sheet_name=' ')
    # df_all_states.to_pickle(filename + '.pkl')

    R_sq = calcRsq(solutions, exp_data)  
    is_negative, full_solutions = solveAll(p, exp_data, 'check negative')
    print(is_negative)
    negative_vals = full_solutions[np.where(full_solutions < 0)]
    print(negative_vals)
    # parityPlot(solutions, exp_data, data)

    #Plot modeling objectives
    plotModelingObjectives123(solutions)
    plotModelingObjectives456(df_sim)
    # plot_all_states(df_all_states, [5.0, 2.5, 0.005, 1, 90], '', '')

    # for doses in x:
    #     plot_all_states(df_all_states, doses, '', '')

    ####no longer work due to updated Analysis_Plots.py
    #plot all model states for high RNase H dose (- kRHA)
    # plot_all_states(df_all_states, 'high RNase H', 'rep2 slice', '')
    
    #Plot all model states ('ensemble' or 'slice')
    # plot_all_states(df_all_states, 'mid', 'slice', ' ')
    # plot_all_states(df_all_states, 'opt', 'slice', '2 hours')

    #Plot all states ODE RHS
    # plot_states_RHS(df_all_states, 'mid', 'slice', p, '2 hours')
    # plot_states_RHS(df_all_states, 'opt', 'slice', p, '2 hours')
    
    print('*******')
    print('R2: ' + str(np.round(R_sq, 5)))
    print('chi2: ' + str(np.round(chi2, 5)))
    print('*******')
    # print(df_all_states.at['target aCas13a-gRNA', str([5.0, 2.5, 0.005, 1, 90])])
    

#p = [8.73073E-05,689.9907897,1074.669737,0.190102635,69522.58812,51.10418826,1.377712318,1391.322358,9.030497548, 0]
#p = [2.754411, 27355.04, 5.553618, 0.024552, 0.198738, 47.82998, 5.915647, 3.902321, 24.87791]
#p =  [3.2708473275180174e-05, 2677.071809398809, 2256.0052576508065, 0.2960119850108069, 20702.650674976157, 52.36016435572264, 7.17766539510405, 62.5023586929695, 10.645921202330825]
#p = [.1,	2.504581573,	30146.51304,	0.023584429,	24304579.82,	100,	0.001502667,	0.333186753,	10,	1e-7]
#p = [0.003275643,	49.41033053,	1673.043616,	0.023119102,	64379.62058,	44.18489193,	1.373374696,	0.636375368,	8.236485532]
#p = [0.38472262929271894, 10326.170610983589, 0.03756947455958354, 0.030842009783112367, 0.03777388939379017, 0, 0, 0, 0]
#p = [0.002769118,	600.876134,	1735.266426,	0.024978384,	2164.169562,	15.56007171,	4.030368088,	0.732126987,	9.285178148]
#p = [1.240549087,	36.1361859,	994.7190046,	0.013171187,	2.56898404,	89.54060729,	0,	0,	72.1529056]
#p = [0.00063308,	2327.274696,	100,	0.059341649,	0.419926497,	67.13961241,	0,	0,	12.30863445]
#p = [0.000391575,	17890.64388,	1392.991394,	0.088281649,	79.99260343,	2.113431759,	6.742071329]
# p = [5.98681E-05,	721.1529526,	1360.727836,	0.385250686,	2.580973544,	58.85708085,	7.876468573]
# p = [0.000168421,	47523.25047,	1555.791395,	0.177256736,	14.59213848,	1.196304863,	6.927381021]
# p = [0.002600907,	7196.142726,	0.036236089,	0.770246558,	0.020866175, 0, 0]
# p = [0.001186029,	10898.89966,	0.478019244,	0.050449563,	0.118303588,	35.14967272,	18.85547134]


#ensemble model params
###THESE are used for supp fig sim/exp comparisons
# p = [0.00039, 17890.64388, 1392.99139, 0.08828, 79.9926, 2.11343, 6.74207] #use for CV       

#params from fitting to slice
# p = [5.98681E-05,	721.1529526,	1360.727836,	0.385250686,	2.580973544,	58.85708085,	7.876468573]

#test my best fit for model C params (230731_ModelC_PEM_rep1_slice_nofilter_redo_run2)
# p = [0.00031178, 78.91898242, 1194.417613, 0.074241899, 7.909074516, 55.58341311, 6.745005326]

#model C low tol best fit run 2
# p = [0.000384312, 75.38073202, 1145.888736, 0.062734604, 7.498278516, 55.47333899, 6.805959386]

#high tol best fit run 2
# p = [0.00031178, 78.91898242, 1194.417613, 0.074241899, 7.909074516, 55.58341311, 6.745005326]

#model D rep 2 best fit high tol
# p = [1.51719E-05, 12223.96888, 315.8195865, 0.381765754, 2.197307165, 29.34473237, 68.69366218, 101.5269246, 17.96639629]

#model D rep 2 best fit low tol
# p = [2.22994E-05, 8940.243435, 226.2897324, 232.9366873, 1.749944885, 22.66728787, 4.675577757, 97.309157, 19.21663556]

#model D rep 3 best fit high tol
# p = [0.00012593, 30452.9863669, 41.57403523, 0.07926811, 1.07134915, 14.7113393, 0.62647773, 46.40684061, 18.541746]

#model D rep 1 best fit high tol
# p = [0.00063618, 239.9315589, 858.2136969, 0.027772651, 1.753931699, 10.64957523, 42.38269934, 61.82812743, 8.241330405]

#model D rep 1 best fit low tol
p = [0.000224063, 136.0589787, 1151.286829, 0.09603252, 1.659857597, 12.48387036, 55.71209137, 56.73099109, 6.906538862]

testSingleSet(p)   


   


