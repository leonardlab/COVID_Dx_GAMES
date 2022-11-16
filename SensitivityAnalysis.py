#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:48:28 2022

@author: kate
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log10
import math
import seaborn as sns
import itertools
from lmfit import Parameters, minimize
from gen_mechanism import *
      
#GAMES imports
import Settings_COVID_Dx
from Solvers_COVID_Dx import calcRsq, calcChi2
from DefineExpData_COVID_Dx import defineExp
# from Run_COVID_Dx import solveAll
# from Saving_COVID_Dx import createFolder
# from Analysis_Plots import plotModelingObjectives123, plotModelingObjectives456, parityPlot
# from Test import solveSingleSet

#Define settings
conditions_dictionary, initial_params_dictionary, data_dictionary = Settings_COVID_Dx.init()
full_path = conditions_dictionary["directory"]
model = conditions_dictionary["model"]
data = conditions_dictionary["data"]
error = data_dictionary["error"]
exp_data = data_dictionary["exp_data"]
timecourses_err = data_dictionary["timecourses_err"]
timecourses_exp = data_dictionary["timecourses"]
problem_free = conditions_dictionary["problem"]
bounds = problem_free['bounds']
plt.style.use('/Users/kate/Documents/GitHub/GAMES_COVID_Dx/paper.mplstyle.py')

def solveAll(p, exp_data):
    
    ''''
    Purpose: Solve ODEs for the entire dataset using parameters defined in p 
        p = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA', 'k_loc_deactivation', 'k_scale_deactivation']
    
    Inputs: 
        p: a list of floats containing the parameter set (order of parameter defined in 
            Settings.py and in the ODE defintion function in Solvers.py)
        exp_data: a list of floats containing the experimental data (length = # data points)
        x:  a list of lists containing the component dose conditions
            
    Outputs: 
        x: list of lists containing the conditions
        solutions_norm: list of floats containing the normalized simulation values
        mse: float defining the chi2/number of data points
        dfSimResults: df containing the normalized simulation values'''
   
    dfSimResults = pd.DataFrame()
    solutions = []
   
    for doses in x: #For each condition (dose combination) in x 

        t, solutions_all, reporter_timecourse = solveSingle(doses, p, p_fixed, model)
     
        if len(reporter_timecourse) == len(t):
            reporter_timecourse = reporter_timecourse
            
        else:
            reporter_timecourse = [0] * len(t)
        
        for i in reporter_timecourse:
            solutions.append(float(i))
            
        dfSimResults[str(doses)] = reporter_timecourse
        
   
    #Normalize solutions
    if max(solutions) == 0:
        solutions_norm = [0] * len(solutions)
    else:    
        solutions_norm = [i/max(solutions) for i in solutions]  

    #Normalize df solutions
    for column in dfSimResults:
        vals = list(dfSimResults[column])
        if max(solutions) == 0:
            dfSimResults[column] = vals
        else:
            dfSimResults[column] = [i/max(solutions) for i in vals]
   
    #dfSimResults.to_excel('./SIMULATION RESULTS.xlsx')
    
    #Check for Nan
    i = 0
    for item in solutions_norm:
       
        if math.isnan(item) == True:
     
            print('Nan in solutions')
            chi2 = 1e10
            return x, solutions_norm, chi2, dfSimResults

    chi2 = calcChi2(exp_data, solutions_norm)
    mse = chi2/len(solutions_norm)
    print('mse: ' + str(np.round(mse, 4)))
    
    #max val filter
    if max(solutions) < 2000:
        print('Failed filter 1')
        print('The maximum value of x_f is: ' + str(max(solutions)))
        mse = 1
        
    #low iCas13 filter
    else:
        doses = [5.0, 0.5, 0.005, 1, 4.5]
        t, solutions_all, reporter_timecourse = solveSingle(doses, p, p_fixed, model)
        final_timepoint_iCas13 = reporter_timecourse[-1] #no norm
        max_high_iCas13 = max(solutions) #no norm
        ratio_2 = final_timepoint_iCas13/max_high_iCas13
        if ratio_2 > 0.10:
            print('Failed filter 2')
            print('ratio2: The max lowiCas13/max high iCas13 ratio is: ' + str(ratio_2))
            mse = 2

    return x, solutions_norm, mse, dfSimResults



def solveSingle(doses, p, p_fixed, model): 
    '''Purpose: 
        Solve equations for a single set of conditions (component doses)
        
    Inputs: 
        doses: list of floats containing the component doses 
        p: list of floats containing parameter values
        model: string defining the model identity
           
    Output: 
        t: list of floats containing the time points
        solution: array containing the solutions for all model states
        timecourse_readout: list of floats containing the value of the readout at each time point'''
    
    solver = ODE_solver()
    x_init = np.zeros(33)
   
    
    C_scale = 10 ** 6
    x_init[0] = doses[3] * .000001  # x_v
    x_init[0] = x_init[0] * C_scale #x_v' sent into ODEs (x_v' = x_v * 10^6)   
    x_init[1] = 250 # x_p1
    x_init[2] = 250 # x_p2
    x_init[7] = doses[1] * 139.1 # x_RT
    x_init[8] = doses[2] * 6060 # x_RNase
    x_init[23] = doses[0] * 16.16 # x_T7
    x_init[27] = doses[4]/2 # x_iCas13
    x_init[30] = 2500 # x_qRf
    solver.set_initial_condition(np.array(x_init))
  
    #Parameters
    k_cas13  = p[0] #nM-1 min-1
    k_degv = p[1] #nM-1 min-1
    k_txn = p[2] #min-1
    k_FSS = p[3] #min-1
    k_RHA = p[4] #min-1
    k_bds = k_cas13 #nM-1 min-1
    k_RTon = p_fixed[0] #nM-1 min-1
    k_RToff = p_fixed[1] #min-1
    k_T7on = p_fixed[2] #nM-1 min-1
    k_T7off = p_fixed[3] #min-1
    k_SSS = k_FSS #min-1
    k_degRrep = k_degv  #nM-1 min-1
    k_RNaseon = p_fixed[4] #nM-1 min-1
    k_RNaseoff = p_fixed[5] #min-1
    the_rates = np.array([k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, k_FSS, k_RHA, k_SSS, k_txn, k_cas13, k_degRrep]).astype(float)
    solver.set_rates(the_rates)
    solver.abs_tol = 1e-16
    solver.complete_output = 0
    solver.conservation_form = True
    solver.dist_type = 'expon'
  
    #Time-stepping
    timesteps = (240 * 100) + 1
    final_time = 240
    tspace = np.linspace(0, final_time, timesteps)
    
    #Set solver type and algorithm
    solver.solver_type = 'solve_ivp'
    solver.solver_alg = 'LSODA'
    solver.k_loc_deactivation = p[5]
    solver.k_scale_deactivation = p[6]
    
    if model == 'model A':
        solver.mechanism_B = 'no'
        solver.mechanism_C = 'no'
       
    elif model == 'model B':
        solver.mechanism_B = 'yes'
        solver.mechanism_C = 'no'
      
    elif model == 'model C' or model == 'model D':
        solver.mechanism_B = 'yes'
        solver.mechanism_C = 'yes'
        solver.txn_poisoning = 'no'
        
    elif model == 'w/txn poisoning':
        
        solver.mechanism_B = 'yes'
        solver.mechanism_C = 'yes'
        solver.txn_poisoning = 'yes'
    
        solver.k_loc_deactivation = p[5]
        solver.k_Mg = p[6]
        solver.n_Mg = p[7]
        solver.k_scale_deactivation = p[8]
        
    
    
    #Solve equations
    solution, t = solver.solve(tspace)
    
    #Round results
    solution = np.around(solution, decimals = 10)
      
    #Define the time course of the readout
    timecourse_readout = solution[:, -1]   
  
    #Unscale vRNA to original units
    vRNA_unscaled = [i/C_scale for i in solution[:,0]] #vRNA = vRNA' / 1000000
    solution[:,0] = vRNA_unscaled
    
    #Restructure t and readout time course to match exp data sampling
    t = t[::400]
    timecourse_readout = timecourse_readout[::400]
    
    return t, solution, timecourse_readout
         

def plotPrediction():
    p = [5.98681E-05,	721.1529526,	1360.727836,	0.385250686,	2.580973544,	58.85708085,	7.876468573]
    t, solution, timecourse_readout = solveSingle(doses, p, p_fixed, 'model C')
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (6.5, 3))
    ax1.plot(t, timecourse_readout, marker = 'None', linestyle= 'dotted', color='dimgrey')
    ax1.set_title('Timecourse', fontsize = 8)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Readout (au)')  
    ax1.set_ylim([0, 2500])
    
   
    input_RNA_dose_response_readouts = []
    for i, input_RNA_dose in enumerate(input_RNA_doses):
        doses[3] = input_RNA_dose
        t, solution, timecourse_readout = solveSingle(doses, p, p_fixed, 'model C')
        input_RNA_dose_response_readouts.append(timecourse_readout[-1])
    
    ax2.plot(input_RNA_doses, input_RNA_dose_response_readouts,  marker = 'None', linestyle= 'dashed', color='dimgrey')
    ax2.set_title('Prediction: input RNA dose response', fontsize = 8)
    ax2.set_xlabel('input RNA (fM)')
    ax2.set_ylabel('Readout (au) at 240 min.')    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.savefig('./input RNA dose response prediction for optimal conditions.svg', dpi = 600)

def fitHill(y_exp):
    if data_type == 'training':
        x = list(np.linspace(0, 240, 61)) #time (min)
    elif data_type == 'prediction':
        x = input_RNA_doses
    #Set v max to the final value of the time course
    fmax = y_exp[-1]
   
    #Set v0 to the intiial value of the time course
    f0 = y_exp[0]

    #Define a function to calculate the residual between the input simulation value (sim) and the Hill fit (model)
    def residual(p, x, y_exp):
        km = p['km'].value
        n = p['n'].value
        model = (((fmax - f0) * (x ** n)) / (km ** n + x ** n)) + f0
        return (y_exp - model)
    
    #Define parameters to be fit, along with initial guesses
    p = Parameters()
    p.add('km', value=10e-2, min=0, vary = True)
    p.add('n', value=2, min=0, max = 5, vary = True)
    
    #Perform the fit
    out = minimize(residual, p, args=(x, y_exp))
    bestFitParams = out.params.valuesdict()
    bestFitParamsList= bestFitParams.values()
    
    #Define fit parameters
    fit = []
    for value in bestFitParamsList:
        fit.append(value)
    km = fit[0]
    n = fit[1]
    
    #Simulate the Hill fit
    y_Hill = []
    for item in x:
        value = (((fmax - f0) * (item ** n)) / (km ** n + item ** n)) + f0
        y_Hill.append(value)
    
    #Calculate the R2 between the data and the Hill fit
    R_sq = calcRsq(y_exp, y_Hill)

    
    return f0, fmax, km, n, R_sq
    
def singleParamSweep(param_index):
    p = [5.98681E-05,	721.1529526,	1360.727836,	0.385250686,	2.580973544,	58.85708085,	7.876468573]
    
    param_log = log10(p[param_index])
    min_bound_log = param_log - 1
    max_bound_log = param_log + 1
    params_for_sweep = np.logspace(min_bound_log, max_bound_log, 10)
    
    #Plot timecourse for each parameter value
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (6.5, 3))
    palette = itertools.cycle(sns.color_palette("Blues", 10))
    for i, param_value in enumerate(params_for_sweep):
        p = [5.98681E-05,	721.1529526,	1360.727836,	0.385250686,	2.580973544,	58.85708085,	7.876468573]
        p[param_index] = param_value
        t, solution, timecourse_readout = solveSingle(doses, p, 'model C')
        ax1.plot(t, timecourse_readout, marker = 'None', linestyle= 'dotted', color=next(palette), label = str(np.round(log10(params_for_sweep[i]), 2)))
    #ax1.legend(title = 'log(' + p_labels[param_index] + ')', loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_title('Timecourse', fontsize = 8)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Readout (au)')    
    ax1.set_ylim([0, 2500])
        
    #Plot input RNA dose response for each parameter value
   
    palette = itertools.cycle(sns.color_palette("Blues", 10))
    for j, param_value in enumerate(params_for_sweep):
        p = [5.98681E-05,	721.1529526,	1360.727836,	0.385250686,	2.580973544,	58.85708085,	7.876468573]
        p[param_index] = param_value
        input_RNA_dose_response_readouts = []
        for i, input_RNA_dose in enumerate(input_RNA_doses):
           doses[3] = input_RNA_dose
           t, solution, timecourse_readout = solveSingle(doses, p, 'model C')
           input_RNA_dose_response_readouts.append(timecourse_readout[-1])
       
        ax2.plot(input_RNA_doses, input_RNA_dose_response_readouts,  marker = 'None', linestyle= 'dashed', color=next(palette), label = str(np.round(log10(params_for_sweep[j]), 2)))
    ax2.legend(title = 'log(' + p_labels[param_index] + ')', loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_title('Prediction: input RNA dose response', fontsize = 8)
    ax2.set_xlabel('input RNA (fM)')
    ax2.set_ylabel('Readout (au) at 240 min.')    
    ax2.set_xscale('log')
    
    plt.savefig('./sensitivity analysis ' + p_labels[param_index] + '.svg', dpi = 600)
    
    
def singleParamSweep_10percent(param_index, p_type, data_type):
    p = [5.98681E-05,	721.1529526,	1360.727836,	0.385250686,	2.580973544,	58.85708085,	7.876468573]
    p_fixed = [.024, 2.4, 3.36, 12, .024, 2.4]
    
    if p_type == 'free':
        p_low = p[param_index] * .9
        p_high = p[param_index] * 1.1
        p[param_index] = p_low
     
    elif p_type == 'fixed':
        p_fixed_low = p_fixed[param_index] * .9
        p_fixed_high = p_fixed[param_index] * 1.1
        p_fixed[param_index] = p_fixed_low
        
    if data_type == 'training':
        
        t, solution, timecourse_readout = solveSingle(doses, p, p_fixed, 'model C')
        t, solution, timecourse_readout_norm_condition = solveSingle(doses_norm, p, p_fixed, 'model C')
        timecourse_readout_norm = [i/max(timecourse_readout_norm_condition) for i in timecourse_readout]
        f0_low, fmax_low, km_low, n_low, R_sq_low = fitHill(timecourse_readout_norm)
        
        if p_type == 'free':
            p[param_index] = p_high
         
        elif p_type == 'fixed':
            p_fixed[param_index] = p_fixed_high
            
        t, solution, timecourse_readout = solveSingle(doses, p, p_fixed, 'model C')
        t, solution, timecourse_readout_norm_condition = solveSingle(doses_norm, p, p_fixed, 'model C')
        timecourse_readout_norm = [i/max(timecourse_readout_norm_condition) for i in timecourse_readout]
        f0_high, fmax_high, km_high, n_high, R_sq_high = fitHill(timecourse_readout_norm)
        
    elif data_type == 'prediction':
        
        #low
        
        input_RNA_dose_response_readouts = []
        for i, input_RNA_dose in enumerate(input_RNA_doses):
            doses[3] = input_RNA_dose
            t, solution, timecourse_readout = solveSingle(doses, p, p_fixed, 'model C')
            input_RNA_dose_response_readouts.append(timecourse_readout[-1])
        f0_low, fmax_low, km_low, n_low, R_sq_low = fitHill(input_RNA_dose_response_readouts)
        
        
        #high
        if p_type == 'free':
            p[param_index] = p_high
         
        elif p_type == 'fixed':
            p_fixed[param_index] = p_fixed_high
            
       
        input_RNA_dose_response_readouts = []
        for i, input_RNA_dose in enumerate(input_RNA_doses):
            doses[3] = input_RNA_dose
            t, solution, timecourse_readout = solveSingle(doses, p, p_fixed, 'model C')
            input_RNA_dose_response_readouts.append(timecourse_readout[-1])
        f0_high, fmax_high, km_high, n_high, R_sq_high = fitHill(input_RNA_dose_response_readouts)
        
    
    return fmax_low, km_low, fmax_high, km_high
    
   
def tornadoPlot(metric, low_vals, high_vals, p_type):
     
    if p_type == 'free':
        num_params = len(p_labels)
        labels = p_labels
    elif p_type == 'fixed':
        num_params = len(p_fixed_labels)
        labels = p_fixed_labels
    pos = np.arange(num_params) + .5 # bars centered on the y axis
    
    fig, (ax_left, ax_right) = plt.subplots(ncols=2)
    ax_left.set_title('Change in metric from mid to low', fontsize = 8)
    ax_right.set_title('Change in metric from mid to high', fontsize = 8)
    ax_left.barh(pos, low_vals, align='center', facecolor='lightcoral')
    ax_right.set_yticks([])
    ax_left.set_xlabel('% change in metric')
    ax_right.barh(pos, high_vals, align='center', facecolor='mediumaquamarine')
    ax_left.set_yticks(pos)
    ax_left.set_yticklabels(labels, ha='center', x=-.1)
    ax_right.set_xlabel('% change in metric')

    plt.savefig('./tornado plot ' + metric + ' ' + p_type + ' ' + data_type + '.svg', dpi = 600)
   
def plot_tradeoff(x, fmax, t_half, k_CV, fmax_sd_list, t_half_sd_list):
    fig, axes = plt.subplots(ncols=3,nrows=1, sharex=True, sharey=True, figsize = (9,3))
    
    for color_by, ax in zip(['T7', 'RT', 'RNAse'], axes.flat):
        
        if color_by == 'T7':
            vals = [1.0, 5.0, 20.0]
            colors = ['lightgreen', 'mediumseagreen', 'darkgreen']    
            varyIndex = 0
        elif color_by == 'RT':
            vals = [0.5, 2.5, 10.0]
            colors = ['lightsteelblue', 'royalblue', 'midnightblue']
            varyIndex = 1
        elif color_by == 'RNAse':
            vals = [0.001, 0.005, 0.02]
            colors = ['lightcoral', 'red', 'maroon']
            varyIndex = 2
            
        for i, enzyme_condition in enumerate(x):
            if enzyme_condition[varyIndex] == vals[0]:
                color_ = colors[0]
            elif enzyme_condition[varyIndex] == vals[1]:
                color_ = colors[1]
            elif enzyme_condition[varyIndex] == vals[2]:
                color_ = colors[2]     

            if enzyme_condition[3] == 1 and enzyme_condition[4] == 90:
                ax.errorbar(t_half[i], fmax[i],  marker = 'o', xerr = t_half_sd_list[i], yerr = fmax_sd_list[i], linestyle = 'None', color = color_)

            if enzyme_condition == [5.0, 2.5, 0.005, 1, 90]:
                ax.errorbar(t_half[i], fmax[i],  marker = 'o', xerr = t_half_sd_list[i], yerr = fmax_sd_list[i], linestyle = 'None', color = 'dimgrey')
                
            if enzyme_condition == [5.0, 10.0, 0.02, 1, 90]:
                ax.errorbar(t_half[i], fmax[i],  marker = 'o', xerr = t_half_sd_list[i], yerr = fmax_sd_list[i], linestyle = 'None', color = 'black')
                
                
            if color_by == 'RT':
                ax.set_xlabel('t 1/2 (min.)')
            if color_by == 'T7':
                ax.set_ylabel('f max (normalized)')
            ax.set_ylim([0, 1])
            #ax.set_xlim([60, 150])
        
    plt.savefig('./Results/Results tuning/experimental performance metric tradeoff test' + '.svg')
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def plotPerformanceMetricTradeoffs(solutions_norm, k_CV):
    chunked_data = list(chunks(solutions_norm, 61))

    
    fmax_list = []
    t_half_list = []
 
    for data_list in chunked_data:
        f0, fmax, km, n, R_sq = fitHill(data_list)
        fmax_list.append(fmax)
        t_half_list.append(km)
    
    fmax_sd_list = [0] * len(fmax_list)
    t_half_sd_list = [0] * len(fmax_list)
    plot_tradeoff(x, fmax_list, t_half_list, k_CV, fmax_sd_list, t_half_sd_list)
    
    return fmax_list, t_half_list
     
   
# =============================================================================
# manual parameter tuning - fitting on entire dataset
# =============================================================================
data_ = 'all echo without low iCas13 or 0 vRNA and drop high error'
x, exp_data, error, timecourses, timecourses_err = defineExp(data_,'', '','')    
k_CV = 'na'
data_type = 'training'

p_fixed_labels = ['k_RTon', 'k_RToff', 'k_T7on', 'k_T7off', 'k_RNaseon', 'k_RNaseoff']
p_fixed = [.024, 2.4, 3.36, 12, .024, 2.4]
p = [0.000467243,	23034.47396,	1174.784961,	0.006634409,	1.083733895,	25.37653242,	16.83162789]

x, solutions_norm, mse, dfSimResults = solveAll(p, exp_data)
plotPerformanceMetricTradeoffs(solutions_norm, k_CV)
                         
    
  
   
# =============================================================================
# sens. anaylsis
# =============================================================================
 
#Define doses on which to perform sensitivity analysis 
doses_norm = [1.0, 2.5, 0.005, 10, 90]

p_fixed_labels = ['k_RTon', 'k_RToff', 'k_T7on', 'k_T7off', 'k_RNaseon', 'k_RNaseoff']
p_fixed = [.024, 2.4, 3.36, 12, .024, 2.4]


input_RNA_doses = np.logspace(-1, 1, 10) #100 aM (.1fM) to 10 fM (10000 aM)

#use parameters from fitting to slice
p = [5.98681E-05,	721.1529526,	1360.727836,	0.385250686,	2.580973544,	58.85708085,	7.876468573]
p_labels = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA', 'k_loc', 'k_scale']

data_type = 'prediction'

if data_type == 'training':
    t, solution, timecourse_readout = solveSingle(doses, p, p_fixed, 'model C')
    t, solution, timecourse_readout_norm_condition = solveSingle(doses_norm, p, p_fixed, 'model C')
    timecourse_readout_norm = [i/max(timecourse_readout_norm_condition) for i in timecourse_readout]
    f0_mid, fmax_mid, km_mid, n_mid, R_sq_mid = fitHill(timecourse_readout_norm)


elif data_type == 'prediction':
    doses_optimal = [5.0, 10.0, 0.02, 1, 90]
    doses_original =  [5.0, 2.5, 0.005, 1, 90]
    colors = ['forestgreen', 'dimgrey']
    fig = plt.figure(figsize = (4.5, 3))
    
    for j, doses in enumerate([doses_optimal, doses_original]):
      input_RNA_dose_response_readouts = []
        for i, input_RNA_dose in enumerate(input_RNA_doses):
            doses[3] = input_RNA_dose
            t, solution, timecourse_readout = solveSingle(doses, p, p_fixed, 'model C')
            input_RNA_dose_response_readouts.append(timecourse_readout[-1])
        f0_mid, fmax_mid, km_mid, n_mid, R_sq_mid = fitHill(input_RNA_dose_response_readouts)
        
       
        plt.plot(input_RNA_doses, input_RNA_dose_response_readouts,  marker = 'None', linestyle= 'dashed', color=colors[j])
        plt.title('Prediction: input RNA dose response', fontsize = 8)
        plt.xlabel('input RNA (fM)')
        plt.ylabel('Readout (au) at 240 min.')    
        plt.xscale('log')
        plt.ylim([0,2500])
        plt.legend(['optimal', 'original'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./input RNA dose response prediction for original and optimal conditions updated.svg', dpi = 600)
        

fmax_low_list = []
thalf_low_list = []
fmax_high_list = []
thalf_high_list = []
for param_index in range(0, len(p_labels)):
    fmax_low, km_low, fmax_high, km_high = singleParamSweep_10percent(param_index, 'free',data_type)
    percent_fmax_low = -100 * (fmax_mid - fmax_low)/fmax_mid
    percent_fmax_high= -100 * (fmax_mid - fmax_high)/fmax_mid
    percent_km_low = -100 * (km_mid - km_low)/km_mid
    percent_km_high= -100 * (km_mid - km_high)/km_mid
    
    fmax_low_list.append(percent_fmax_low)
    thalf_low_list.append(percent_km_low)
    fmax_high_list.append(percent_fmax_high)
    thalf_high_list.append(percent_km_high)
    
tornadoPlot('fmax', fmax_low_list, fmax_high_list, 'free')
tornadoPlot('thalf', thalf_low_list, thalf_high_list, 'free')

fmax_low_list = []
thalf_low_list = []
fmax_high_list = []
thalf_high_list = []
for param_index in range(0, len(p_fixed_labels)):
    fmax_low, km_low, fmax_high, km_high = singleParamSweep_10percent(param_index, 'fixed', data_type)
    percent_fmax_low = -100 * (fmax_mid - fmax_low)/fmax_mid
    percent_fmax_high= -100 * (fmax_mid - fmax_high)/fmax_mid
    percent_km_low = -100 * (km_mid - km_low)/km_mid
    percent_km_high= -100 * (km_mid - km_high)/km_mid
    
    fmax_low_list.append(percent_fmax_low)
    thalf_low_list.append(percent_km_low)
    fmax_high_list.append(percent_fmax_high)
    thalf_high_list.append(percent_km_high)
    
tornadoPlot('fmax', fmax_low_list, fmax_high_list, 'fixed')
tornadoPlot('thalf', thalf_low_list, thalf_high_list, 'fixed')
