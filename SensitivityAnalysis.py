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
from copy import deepcopy
from lmfit import Parameters, minimize
from gen_mechanism import *
      
#GAMES imports
import Settings_COVID_Dx
from Solvers_COVID_Dx import calcRsq, calc_chi_sq
from DefineExpData_COVID_Dx import defineExp


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
p_all = conditions_dictionary["p_all"] 
real_param_labels_all = conditions_dictionary["real_param_labels_all"] 
plt.style.use('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/paper.mplstyle.py')

def solveSingle(
        doses: list, p: list, p_fixed: list
 ) -> Tuple[list, list, list]: 
    '''
    Solves equations for a single set of conditions (component doses)

    Args:
        doses: a list of floats containing the component doses 
        p: a list of floats defining the parameter values for all 
            potentially free parameters (Settings_COVID_Dx.py
            conditions_dictionary["p_all"])

        p_fixed: a list of floats defining the fixed parameters
           
    Returns: 
        t: a list of floats containing the time points

        solution: a numpy array containing the solutions for all model states

        timecourse_readout: a list of floats containing the value of the 
            readout at each time point
    '''
    
    solver = ODE_solver_D()
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
    a_RHA = p[4]
    b_RHA = p[5]
    c_RHA = p[6]
    k_bds = k_cas13 #nM-1 min-1

    k_RTon = p_fixed[0] #nM-1 min-1
    k_RToff = p_fixed[1] #min-1
    k_T7on = p_fixed[2] #nM-1 min-1
    k_T7off = p_fixed[3] #min-1
    k_SSS = k_FSS #min-1
    k_degRrep = k_degv  #nM-1 min-1
     
    
    k_RNaseon = p_fixed[4] #nM-1 min-1
    k_RNaseoff = p_fixed[5] #min-1
    the_rates = np.array([k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, 
                          k_T7on, k_T7off, k_FSS, a_RHA, b_RHA, c_RHA, k_SSS, 
                          k_txn, k_cas13, k_degRrep]).astype(float)
    solver.set_rates(the_rates)
    solver.abs_tol = 1e-13
    solver.rel_tol = 1e-10
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
    solver.k_loc_deactivation = p[7]
    solver.k_scale_deactivation = p[8]
    
    solver.mechanism_B = 'yes'
    solver.mechanism_C = 'yes'
    solver.txn_poisoning = 'no'
    
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


def fitHill(y_exp: list) -> Tuple[float, float, float, float, float]:

    """
    Fits data to a hill function for a single set of
        conditions (component doses).

    Args: 
        y_exp: a list floats defining the normalized simulation 
            values for a single set of conditions

    Returns:
        f0: a float defining the initial value in the dataset

        fmax: a float defining the final value in the dataset

        km: a float defining the optimized parameter for t1/2

        n: a float defining the optimized parameter for the 
            Hill coefficient

        R_sq: a float defining the R squared value between the 
            data and the Hill fit
    """

    x = list(np.linspace(0, 240, 61)) #time (min)

    #Set v max to the final value of the time course
    fmax = y_exp[-1]
   
    #Set v0 to the intiial value of the time course
    f0 = y_exp[0]

    #Define a function to calculate the residual between the input simulation value (sim) and the Hill fit (model)
    def residual(p: list, x: list, y_exp: list) -> float:

        """
        Calculates the residual between the input simulation
            values and the Hill fit. Used in the minimization
            function as the cost function to be minimized
            between the simulation and the Hill fit.

        Args:
            p: a list of floats defining the parameters for
                the hill function

            x: a list of floats defining the time values for
                the simulation

            y_exp: a list of floats defining the simulation
                values

        Returns: 
            (y_exp - model): a float defining the residual 
            between the input simulation values and the Hill 
            fit
        """

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
    

def singleParamSweep_10percent(
        doses: list, p: list , p_fixed: list,
        param_index: int, p_type: str
) -> Tuple[float, float, float, float]:

    """
    Solves model ODEs for a given set of component doses for two 
        cases of changing parameter at param_index: increase by 10%
        and decrease by 10%

    Args:
        doses: a list of floats defining the component doses for 
            which the ODEs are solved

        p: a list of floats defining the parameter values for all 
            potentially free parameters (Settings_COVID_Dx.py
            conditions_dictionary["p_all"])

        p_fixed: a list of floats defining the fixed parameters

        param_index: an integer defining the index of the parameter
            for the sweep

        p_type: a string defining the parameter type for the
            sweep- 'free' or 'fixed'

    Returns:
        fmax_low: a float defining the maximum value for the 
            time course resulting from the 10% decrease in
            the parameter

        km_low: a float defining the optimal t1/2 from the Hill
            fit resulting from the 10% decrease in the parameter

        fmax_high: a float defining the maximum value for the 
            time course resulting from the 10% increase in
            the parameter
        
        km_high: a float defining the optimal t1/2 from the Hill
            fit resulting from the 10% increase in the parameter
    """

    p_vals = deepcopy(p)
    p_vals_fixed = deepcopy(p_fixed)

    if p_type == 'free':
        p_low = p_vals[param_index] * .9
        p_high = p_vals[param_index] * 1.1
        p_vals[param_index] = p_low
     
    elif p_type == 'fixed':
        p_fixed_low = p_vals_fixed[param_index] * .9
        p_fixed_high = p_vals_fixed[param_index] * 1.1
        p_vals_fixed[param_index] = p_fixed_low
        
        
    t, solution, timecourse_readout = solveSingle(doses, p_vals, p_vals_fixed)
    t, solution, timecourse_readout_norm_condition = solveSingle(doses_norm, p_vals, p_vals_fixed)
    timecourse_readout_norm = [i/max(timecourse_readout_norm_condition) for i in timecourse_readout]
    f0_low, fmax_low, km_low, n_low, R_sq_low = fitHill(timecourse_readout_norm)
    
    if p_type == 'free':
        p_vals[param_index] = p_high
        
    elif p_type == 'fixed':
        p_vals_fixed[param_index] = p_fixed_high
        
    t, solution, timecourse_readout = solveSingle(doses, p_vals, p_vals_fixed)
    t, solution, timecourse_readout_norm_condition = solveSingle(doses_norm, p_vals, p_vals_fixed)
    timecourse_readout_norm = [i/max(timecourse_readout_norm_condition) for i in timecourse_readout]
    f0_high, fmax_high, km_high, n_high, R_sq_high = fitHill(timecourse_readout_norm)

    return fmax_low, km_low, fmax_high, km_high
    
def calc_percent_change(metric_mid: float, metric_new: float) -> float:
    """
    Calculates the percent change between the given metric at original
        and new value

    Args:
        metric_mid: a float defining the mse for the original parameter set

        metric_new: a float defining the mse for the parameter set with increased
            or decreased parameter value

    Returns:
        100 * (mse_new-mse_mid)/mse_mid: a float defining percent change
    """

    return 100 * (metric_new-metric_mid)/metric_mid

def all_param_sweeps_10pct(
        doses: list, doses_norm: list, p: list,
        p_fixed: list, p_type: str, p_labels: list
) -> Tuple[list, list, list, list]:

    """
    Performs all parameter sweeps for increasing or decreasing 
        each parameter value by 10%

    Args:
        doses: a list of floats defining the component doses for 
            which the ODEs are solved

        doses_norm: a list of floats defining the component doses
            for which the ODEs are solved for the normalization

        p: a list of floats defining the parameter values for all 
            potentially free parameters (Settings_COVID_Dx.py
            conditions_dictionary["p_all"])

        p_fixed: a list of floats defining the fixed parameters

        p_type: a string defining the parameter type for the
            sweep- 'free' or 'fixed'

        p_labels: a list of strings defining the parameter labels
            corresponding to the type of sweep

    Returns:
        fmax_low_list: a list of floats defining the percent change
            in fmax for each decrease in parameter by 10%
        
        fmax_high_list: a list of floats defining the percent change
            in fmax for each increase in parameter by 10%
        
        thalf_low_list: a list of floats defining the percent change
            in t_1/2 for each decrease in parameter by 10%
        
        thalf_high_list: a list of floats defining the percent change
            in t_1/2 for each increase in parameter by 10%
    """

    t, solution, timecourse_readout = solveSingle(doses, p, p_fixed)
    t, solution, timecourse_readout_norm_condition = solveSingle(doses_norm, p, p_fixed)
    timecourse_readout_norm = [i/max(timecourse_readout_norm_condition) for i in timecourse_readout]
    f0_mid, fmax_mid, km_mid, n_mid, R_sq_mid = fitHill(timecourse_readout_norm)

    fmax_low_list = []
    thalf_low_list = []
    fmax_high_list = []
    thalf_high_list = []

    for param_index in range(0, len(p_labels)):
        (fmax_low,
         km_low, 
         fmax_high, 
         km_high) = singleParamSweep_10percent(doses, p, p_fixed, param_index, p_type)
        percent_fmax_low = calc_percent_change(fmax_mid, fmax_low)
        percent_fmax_high= calc_percent_change(fmax_mid, fmax_high)
        percent_km_low = calc_percent_change(km_mid, km_low)
        percent_km_high= calc_percent_change(km_mid, km_high)
        
        fmax_low_list.append(percent_fmax_low)
        thalf_low_list.append(percent_km_low)
        fmax_high_list.append(percent_fmax_high)
        thalf_high_list.append(percent_km_high)
    
    return fmax_low_list, fmax_high_list, thalf_low_list, thalf_high_list

   
def tornado_plot(
        metric: str, low_vals: list, high_vals: list,
        p_labels: list, p_type: str, rep: str
) -> None:

    """
    Creates a tornado plot for the sensitivity analysis

    Args:
        metric: a string defining the metric to be plotted

        low_vals: a list of floats defining the percent changes 
            for decreasing each parameter by 10%

        high_vals: a list of floats defining the percent changes 
            for increasing each parameter by 10% 
        
        p_labels: a list of strings defining the parameter
            labels for the plot (Settings_COVID_Dx.py
            conditions_dictionary["real_param_labels_all"]) for
            free parameters or given list for fixed parameters

        p_type: a string defining the parameter type for the
            sweep- 'free' or 'fixed'

        rep: a string defining the data set used in the
            sensitivity analysis
    
    Returns: none

    Figures:
        './tornado plot ' + metric + ' ' + p_type + ' ' + rep + '.svg':
            tornado plot for the sensitivity analysis
    """

    num_params = len(p_labels)

    pos = np.arange(num_params) + .5 # bars centered on the y axis
    
    fig, (ax_left, ax_right) = plt.subplots(ncols=2)
    ax_left.set_title('Change in metric from mid to low', fontsize = 8)
    ax_right.set_title('Change in metric from mid to high', fontsize = 8)
    bars_left = ax_left.barh(pos, low_vals, align='center', facecolor='dimgrey')
    ax_left.bar_label(bars_left)
    ax_right.set_yticks([])
    ax_left.set_xlabel('% change in metric')
    bars_right = ax_right.barh(pos, high_vals, align='center', facecolor='dimgrey')
    ax_left.set_yticks(pos)
    ax_right.bar_label(bars_right)
    ax_left.set_yticklabels(p_labels, ha='center', x=-.1)
    ax_right.set_xlabel('% change in metric')
    # plt.show()
    os.chdir(full_path)
    plt.savefig('./tornado plot ' + metric + ' ' + p_type + ' ' + rep + '.svg', dpi = 600)


def run_sensitivity_analysis(
        doses: list, doses_norm: list, p:list, p_fixed: list,
        p_type: str, p_labels: list, rep: str
) -> None:

    """
    Runs the sensitivity analysis by calling the above functions

    Args:
        doses: a list of floats defining the component doses for 
            which the ODEs are solved

        doses_norm: a list of floats defining the component doses
            for which the ODEs are solved for the normalization

        p: a list of floats defining the parameter values for all 
            potentially free parameters (Settings_COVID_Dx.py
            conditions_dictionary["p_all"])

        p_fixed: a list of floats defining the fixed parameters

        p_type: a string defining the parameter type for the
            sweep- 'free' or 'fixed'

        p_labels: a list of strings defining the parameter labels
            corresponding to the type of sweep

        rep: a string defining the data set used in the
            sensitivity analysis

    Returns: none
    """

    os.chdir(full_path)

    (fmax_low_list, 
     fmax_high_list, 
     thalf_low_list, 
     thalf_high_list) = all_param_sweeps_10pct(doses, doses_norm, p, p_fixed, p_type, p_labels)
    
    tornado_plot('f_max', fmax_low_list, fmax_high_list, p_labels, p_type, rep)
    tornado_plot('t_half', thalf_low_list, thalf_high_list, p_labels, p_type, rep)
   
# =============================================================================
# sens. anaylsis
# =============================================================================
 
#Define doses on which to perform sensitivity analysis 
doses_norm = [1.0, 2.5, 0.005, 10, 90] #(condition for norm- highest readout)
doses_mid =  [5.0, 2.5, 0.005, 1, 90]

p_fixed_labels = ['k_RTon', 'k_RToff', 'k_T7on', 'k_T7off', 'k_RNaseon', 'k_RNaseoff']
p_fixed = [0.024, 2.4, 3.36, 12, 0.024, 2.4]

run_sensitivity_analysis(doses_mid, doses_norm, p_all, p_fixed, 'free', real_param_labels_all, 'rep1')