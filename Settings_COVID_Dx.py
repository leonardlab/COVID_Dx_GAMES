#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:12:12 2020

@author: kate
"""
#Package imports
from math import log10

#GAMES imports
from Saving_COVID_Dx import makeMainDir
from DefineExpData_COVID_Dx import defineExp


def init():
    '''
    Purpose: Define conditions for simulations, called by other functions to set conditions
    
    Input: None
    
    Outputs: 3 dictionaries - conditions_dictionary, initial_params_dictionary, data_dictionary
    
    '''
    # =============================================================================
    # 1. Define and create folder for saving results
    # =============================================================================
    #This will be the name of the run-specific results folder. 
    folder_name = '230727_PEM_eval_model_D_rep1'

    #model B parameter estimation with par 5000 + 24'
    #fix with 0 for txn poisoning mechs'

    # =============================================================================
    # 2. Define modelID, free parameters, and bounds
    # =============================================================================
    modelID = 'model D'
    
    real_param_labels_all = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA', 'k_loc_deactivation', 'k_scale_deactivation'] #real labels for p_all
    
    
    if modelID == 'model A':
        real_param_labels_free = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA']
        p_all = [0.00198, 30.6, 36, 0.6, 7.8, 0, 0]
        
    elif modelID == 'model B':
        real_param_labels_free = real_param_labels_all
        p_all = [0.00198, 30.6, 36, 0.6, 7.8, 1, 1] 
        
    elif modelID == 'model C':
        real_param_labels_free = real_param_labels_all
        p_all = [0.00198, 30.6, 36, 0.6, 7.8, 1, 1] #use for 1st round
        #p_all = [0.00063308,	2327.274696,	48.42183504,	0.059341649,	0.419926497,	67.13961241,	12.30863445] #use for 2nd round
        # p_all = [0.00039, 17890.64388, 1392.99139, 0.08828, 79.9926, 2.11343, 6.74207] #use for CV

    elif modelID == 'model D':
        p_all = [0.00198, 30.6, 36, 0.6, 0.0001, 0.01, 7.8, 1, 1]
        real_param_labels_free = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'a_RHA', 'b_RHA', 'c_RHA', 'k_loc_deactivation', 'k_scale_deactivation']
        real_param_labels_all = real_param_labels_free
        p_labels_all = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']

    # elif modelID == 'w/txn poisoning':
    #     p_all = [0.00198, 30.6, 36, 0.6, 7.8, 1, 0, 0, 1] #use for 1st round
    #     real_param_labels_free = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA', 'k_loc_deactivation', 'k_Mg', 'n_Mg', 'k_scale_deactivation']
    #     real_param_labels_all =real_param_labels_free
    #     p_labels_all = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']
        
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
            if i == 5: #k_loc_deactivation
                minBound = 0
                maxBound = 2.38
                
            elif i == 6: #k_scale_deactivation
                minBound = 0
                maxBound = 2.38
                
            else: 
                minBound = log10(p_all[i]) - 3
                maxBound = log10(p_all[i]) + 3
            bounds_log.append([minBound, maxBound])

        elif modelID == 'model D':
            if i == 7: #k_loc_deactivation
                minBound = 0
                maxBound = 2.38
                
            elif i == 8: #k_scale_deactivation
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
    conditions_dictionary["data"] = 'PEM evaluation' #'slice drop high error'
    conditions_dictionary["run_type"] = 'generate PEM evaluation data' #'parameter estimation'
    conditions_dictionary["n_search"] = 5000
    conditions_dictionary["n_initial_guesses"] = 24
    conditions_dictionary['k_CV'] = 13 #starts at 1, not 0. Only relevant if data == 'cross-validation train' or data == 'cross-validation test'
    conditions_dictionary['k_PEM_evaluation'] = 3 #starts at 1, not 0. Only relevant if data == 'PEM evaluation'
    
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
    x_vals, exp_data, error, timecourses, timecourses_err = defineExp(conditions_dictionary["data"], conditions_dictionary["model"], conditions_dictionary['k_CV'], conditions_dictionary['k_PEM_evaluation'])
    data_dictionary["x_vals"] = x_vals
    data_dictionary["exp_data"] = exp_data
    
    if conditions_dictionary["run_type"] == 'generate PEM evaluation data' or conditions_dictionary["run_type"] == 'plot error':
        for i in range(0, len(error)):
            if error[i] == 0.0:
                error[i] = error[i-1]
    else:
        error = [1] * len(error) #use placeholder for error (not used)
        
    data_dictionary["error"] = error
    data_dictionary["timecourses_err"] = timecourses_err
    data_dictionary["timecourses"] = timecourses
    
    return conditions_dictionary, initial_params_dictionary, data_dictionary
