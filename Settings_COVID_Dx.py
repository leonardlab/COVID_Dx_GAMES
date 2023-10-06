#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:12:12 2020

@author: kate
"""
#Package imports
from math import log10
from typing import Tuple

#GAMES imports
from Saving_COVID_Dx import makeMainDir
from DefineExpData_COVID_Dx import defineExp


def init() -> Tuple[dict, dict, dict]:
    '''
    Defines conditions for simulations, called by other functions to set conditions
    
    Args: none
    
    Returns:
        conditions_dictionary: a dictionary holding simulation conditions

        initial_params_dictionary: a dictionary holding initial parameters
            and related information

        data_dictionary: a dictionary holding experimental data and related
            information
    '''

    # =============================================================================
    # 1. Define and create folder for saving results
    # =============================================================================
    #This will be the name of the run-specific results folder. 
    folder_name = 'Test_run'

    # =============================================================================
    # 2. Define modelID, free parameters, and bounds
    # =============================================================================
    modelID = 'model D'
    
    real_param_labels_all = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA',
                            'k_loc_deactivation', 'k_scale_deactivation'] #real labels for p_all
    
    if modelID == 'model A':
        real_param_labels_free = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA']
        p_all = [0.00198, 30.6, 36, 0.6, 7.8, 0, 0]
        
    elif modelID == 'model B':
        real_param_labels_free = real_param_labels_all
        p_all = [0.00198, 30.6, 36, 0.6, 7.8, 1, 1] 
        
    elif modelID == 'model C':
        real_param_labels_free = real_param_labels_all
        p_all = [0.00198, 30.6, 36, 0.6, 7.8, 1, 1] #use for 1st round

    elif modelID == 'model D':
        p_all = [0.00198, 30.6, 36, 0.6, 1.0, 1.0, 1.0, 1, 1]

        # p_all = [0.00063618, 239.9315589, 858.2136969, 0.027772651, 1.753931699, 10.64957523, 42.38269934, 61.82812743, 8.241330405] #rep1 best fit high tol
        # p_all = [0.000224063, 136.0589787, 1151.286829, 0.09603252, 1.659857597, 12.48387036, 55.71209137, 56.73099109, 6.906538862] #rep1 best fit low tol

        # p_all = [1.51719E-05, 12223.96888, 315.8195865, 0.381765754, 2.197307165, 29.34473237, 68.69366218, 101.5269246, 17.96639629] #rep2 best fit high tol
        # p_all = [2.22994E-05, 8940.243435, 226.2897324, 232.9366873, 1.749944885, 22.66728787, 4.675577757, 97.309157, 19.21663556] #rep2 best fit low tol 

        # p_all = [0.00012593, 30452.9863669, 41.57403523, 0.07926811, 1.07134915, 14.7113393, 0.62647773, 46.40684061, 18.541746] #rep3 best fit high tol
        # p_all = [6.38185E-05, 28934.65508, 119.1874985, 0.077888022, 1.627413157, 23.55089534, 22.28442186, 50.03161646, 17.06654392] #rep3 best fit low tol

        real_param_labels_free = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'a_RHA', 'b_RHA', 'c_RHA', 'k_loc_deactivation', 'k_scale_deactivation']
        real_param_labels_all = real_param_labels_free
        p_labels_all = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']
        
    #Change param labels to generalizable param labels
    # p_labels_all = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'] 
    num_free_params = len(real_param_labels_free)
    initial_params_dictionary = {}
    params = []
    p_labels_free = []
    for i, value in enumerate(p_all):
        label = real_param_labels_all[i]
        if label in real_param_labels_free:
            initial_params_dictionary[label] = value
            p_labels_free.append(p_labels_all[i])
            params.append(value)
    
    #Set bounds for parameter estimation
    bounds_log = []
    for i in range(0, num_free_params):
        
        if modelID == 'model A':
            minBound = log10(p_all[i]) - 3
            maxBound = log10(p_all[i]) + 3
            bounds_log.append([minBound, maxBound])
     
        elif modelID == 'model B' or modelID == 'model C':
            #k_loc_deactivation
            if i == 5: 
                minBound = 0
                maxBound = 2.38
            
            #k_scale_deactivation
            elif i == 6:
                minBound = 0
                maxBound = 2.38
                
            else: 
                minBound = log10(p_all[i]) - 3
                maxBound = log10(p_all[i]) + 3
            bounds_log.append([minBound, maxBound])

        elif modelID == 'model D':
            #a_RHA
            if i == 4:
                minBound = 0
                maxBound = 1

            #b_RHA
            elif i == 5: 
                minBound = 0
                maxBound = 2

            #k_loc_deactivation   
            elif i == 7:
                minBound = 0
                maxBound = 2.38

            #k_scale_deactivation   
            elif i == 8:
                minBound = 0
                maxBound = 2.38
                
            else: 
                minBound = log10(p_all[i]) - 3
                maxBound = log10(p_all[i]) + 3
            bounds_log.append([minBound, maxBound])

    
    #Define the parameter estimation problem (free parameters for this run only)
    problem = {'num_vars': num_free_params,  #set free parameters and bounds
               'names': p_labels_free, 
               'bounds': bounds_log} #bounds are in log scale
    
    # =============================================================================
    # 3. Define conditions dictionary
    # =============================================================================
    #Initialize conditions dictionary
    conditions_dictionary = {}
    conditions_dictionary["model"] = modelID
    #data set 1: 'slice drop high error' 
    #data set 2: 'rep2 slice drop high error'
    #data set 3: 'rep3 slice drop high error'
    #PEM evaluation data: 'PEM evaluation'
    conditions_dictionary["data"] = 'rep2 slice drop high error'
    #'parameter estimation', #'generate PEM evaluation data', or ' ' for test
    conditions_dictionary["run_type"] = ' '  
    conditions_dictionary["n_search"] = 5000
    conditions_dictionary["n_initial_guesses"] = 24
    #PEM evaluation data set to use for PEM: starts at 1, not 0. 
    #Only relevant if data == 'PEM evaluation'
    conditions_dictionary['k_PEM_evaluation'] = 1
    conditions_dictionary["num_cores"] = 8
    conditions_dictionary["num_datasets_pem_eval"] = 3
    full_path = makeMainDir(folder_name)
    conditions_dictionary["real_param_labels_free"] = real_param_labels_free
    problem_all_params = {'num_vars': len(p_labels_all),  
                          'names': p_labels_all, 
                          'bounds': [[]]} 
    conditions_dictionary["real_param_labels_all"] = real_param_labels_all
    conditions_dictionary["p_all"] = p_all
    conditions_dictionary["directory"] = full_path
    conditions_dictionary["problem"] = problem
    conditions_dictionary["problem_all_params"] = problem_all_params
    conditions_dictionary["parallelization"] = 'yes'
    model_states = [
        'vRNA (input)',
        'ssDNA p1',
        'ssDNA p2',
        'ssDNA p1:vRNA',
        'ssDNA p2:tRNA',
        'ssDNA p1:cvRNA',
        'ssDNA p2:ctRNA',
        'RT',
        'RNase H',
        'RT-ssDNA p1:vRNA',
        'RT-ssDNA p2:tRNA',
        'RT-ssDNA p1:cvRNA',
        'RT-ssDNA p2:ctRNA',
        'cDNA1:vRNA',
        'cDNA2:tRNA',
        'cDNA1:vRNA: RNase H',
        'cDNA2:tRNA: RNase H',
        'cDNA1:vRNA frag',
        'cDNA2:tRNA frag',
        'cDNA1:ssDNA p2',
        'cDNA2:ssDNA p1',
        'cDNA1:ssDNA p2:RT',
        'cDNA2:ssDNA p1:RT',
        'T7 RNAP',
        'dsDNA T7 target',
        'T7: dsDNA T7 target',
        'tRNA (target)',
        'target iCas13a-gRNA',
        'target aCas13a-gRNA',
        'dsRNA (input:target)',
        'quench-ssRNA-fluoro',
        'quencher',
        'fluorophore (output)'
    ]
    conditions_dictionary["model states"] = model_states
  
    
    # =============================================================================
    # 4. Define data dictionary
    # =============================================================================
    data_dictionary = {}
    x_vals, exp_data, error, timecourses, timecourses_err = defineExp(conditions_dictionary["data"], conditions_dictionary['k_PEM_evaluation'])
    data_dictionary["x_vals"] = x_vals
    data_dictionary["exp_data"] = exp_data
    
    if conditions_dictionary["run_type"] == 'generate PEM evaluation data' or conditions_dictionary["run_type"] == 'plot error':
        for i in range(0, len(error)):
            if error[i] == 0.0:
                error[i] = error[i-1]
    else:
        #use placeholder for error (not used)
        error = [1] * len(error)
        
    data_dictionary["error"] = error
    data_dictionary["timecourses_err"] = timecourses_err
    data_dictionary["timecourses"] = timecourses
    
    return conditions_dictionary, initial_params_dictionary, data_dictionary
