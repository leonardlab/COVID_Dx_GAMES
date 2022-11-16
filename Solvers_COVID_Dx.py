#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 08:39:09 2020

@author: kate
"""

#Package imports
import numpy as np
from sklearn.linear_model import LinearRegression
from gen_mechanism import *

# =============================================================================
# CODE TO CALCULATE COST FUNCTIONS
# ============================================================================= 
def calcRsq(data_x, data_y):
    ''' 
    Purpose: Calculate correlation coefficient, Rsq, between 2 datasets
        
    Inputs: 2 lists of floats (of the same length), dataX and dataY 
           
    Output: Rsq value (float) 
    
    '''
    #Restructure the data
    x = np.array(data_x)
    y = np.array(data_y)
    x = x.reshape((-1, 1))
    
    #Perform linear regression
    model = LinearRegression()
    model.fit(x,y)
    
    #Calculate Rsq
    Rsq = model.score(x, y)
   
    return Rsq

def calc_chi_sq(exp, sim):
    ''' 
    Purpose: 
        Calculate chi2 between 2 datasets 
        
    Inputs: 
        exp: experimental data (list of floats, length = # datapoints)
        sim: simulated data (list of floats, length = # datapoints)\
           
    Output: 
        chi2: chi2 value (float) 
    
    '''
   
    #Initialize chi2
    chi2 = 0
    
    #Calculate chi2
    for i, sim_val in enumerate(sim): #for each datapoint
        err = ((exp[i] - sim_val) ** 2)
        chi2 = chi2 + err
        
    return chi2

# =============================================================================
# CODE TO SOLVE MODEL FOR A SINGLE CONDITION
# ============================================================================= 
def solveSingle(doses, p, model): 
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
    k_RTon = .024 #nM-1 min-1
    k_RToff = 2.4 #min-1
    k_T7on = 3.36 #nM-1 min-1
    k_T7off = 12 #min-1
    k_SSS = k_FSS #min-1
    k_degRrep = k_degv  #nM-1 min-1
    k_RNaseon = .024 #nM-1 min-1
    k_RNaseoff = 2.4 #min-1
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
        solver.txn_poisoning = 'no'
       
    elif model == 'model B':
        solver.mechanism_B = 'yes'
        solver.mechanism_C = 'no'
        solver.txn_poisoning = 'no'
      
    elif model == 'model C' or model == 'model D':
        solver.mechanism_B = 'yes'
        solver.mechanism_C = 'yes'
        solver.txn_poisoning = 'no'
        
    # elif model == 'w/txn poisoning':
        
    #     solver.mechanism_B = 'yes'
    #     solver.mechanism_C = 'yes'
    #     solver.txn_poisoning = 'yes'
    
    #     solver.k_loc_deactivation = p[5]
    #     solver.k_Mg = p[6]
    #     solver.n_Mg = p[7]
    #     solver.k_scale_deactivation = p[8]
        
    
    
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
         
         
   