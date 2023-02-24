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
df_data = pd.read_pickle('./PROCESSED DATA EXP.pkl')
df_error = pd.read_pickle('./PROCESSED DATA ERR.pkl')  
plt.style.use('/Users/kdreyer/Desktop/Github/COVID_Dx_GAMES/paper.mplstyle.py')
    

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
    
    doses, solutions, chi2, df_sim, df_all_states = solveAll(p, exp_data, 'all states')
    R_sq = calcRsq(solutions, exp_data)  
    # parityPlot(solutions, exp_data, data)
    
    #Plot modeling objectives
    # plotModelingObjectives123(solutions)
    # plotModelingObjectives456(df_sim)

    #Plot all model states ('ensemble' or 'slice')
    plot_all_states(df_all_states, 'mid', 'slice')
    plot_all_states(df_all_states, 'opt', 'slice')

    #Plot all states ODE RHS
    plot_states_RHS(df_all_states, 'mid', 'slice', p)
    plot_states_RHS(df_all_states, 'opt', 'slice', p)
    
    print('*******')
    print('R2: ' + str(np.round(R_sq, 3)))
    print('chi2: ' + str(np.round(chi2, 3)))
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
#p = [5.98681E-05,	721.1529526,	1360.727836,	0.385250686,	2.580973544,	58.85708085,	7.876468573]
#p = [0.000168421,	47523.25047,	1555.791395,	0.177256736,	14.59213848,	1.196304863,	6.927381021]
# p = [0.002600907,	7196.142726,	0.036236089,	0.770246558,	0.020866175, 0, 0]
# p = [0.001186029,	10898.89966,	0.478019244,	0.050449563,	0.118303588,	35.14967272,	18.85547134]


#ensemble model params
# p = [0.00039, 17890.64388, 1392.99139, 0.08828, 79.9926, 2.11343, 6.74207] #use for CV       

#params from fitting to slice
p = [5.98681E-05,	721.1529526,	1360.727836,	0.385250686,	2.580973544,	58.85708085,	7.876468573]

testSingleSet(p)   


   


