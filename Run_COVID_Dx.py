#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 08:50:07 2020

@author: kate
"""

# =============================================================================
# IMPORTS
# =============================================================================
#Import external packages/functions
from lmfit import Model, Parameters
from openpyxl import load_workbook
from typing import Tuple
import datetime
import os
import multiprocessing as mp
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import random
import json
import signal

#Import GAMES functions
from Solvers_COVID_Dx import solveSingle, calcRsq, calc_chi_sq
from Saving_COVID_Dx import createFolder, saveConditions
from GlobalSearch import generateParams, filterGlobalSearch
import Settings_COVID_Dx
from Analysis_Plots import plotParamDistributions, plotParamBounds, plotModelingObjectives123, plotModelingObjectives456, plotLowCas13, parityPlot
from DefineExpData_COVID_Dx import defineExp

#Unpack conditions from Settings.py
conditions_dictionary, initial_params_dictionary, data_dictionary = Settings_COVID_Dx.init()
model = conditions_dictionary["model"] 
data = conditions_dictionary["data"]
num_cores = conditions_dictionary["num_cores"]
n_initial_guesses = conditions_dictionary["n_initial_guesses"]
problem_free = conditions_dictionary["problem"]
full_path = conditions_dictionary["directory"]
n_search = conditions_dictionary["n_search"]
n_search_parameter_estimation= conditions_dictionary["n_search"]
run_type = conditions_dictionary["run_type"] 
fit_params = problem_free['names']
bounds = problem_free['bounds']
num_vars = problem_free['num_vars']
p_all = conditions_dictionary["p_all"] 
real_param_labels_all = conditions_dictionary["real_param_labels_all"] 
num_datasets_pem_eval = conditions_dictionary["num_datasets_pem_eval"] 
problem_all_params = conditions_dictionary["problem_all_params"]
all_param_labels = problem_all_params['names']
param_labels = list(initial_params_dictionary.keys())
init_params = list(initial_params_dictionary.values())
real_param_labels_free = conditions_dictionary["real_param_labels_free"]
k_CV = conditions_dictionary['k_CV']
k_PEM_evaluation = conditions_dictionary['k_PEM_evaluation']
x = data_dictionary["x_vals"]
exp_data = data_dictionary["exp_data"]
error = data_dictionary["error"]
parallelization = conditions_dictionary["parallelization"]
model_states = conditions_dictionary["model states"]
save_internal_states_flag = False
data_type = 'experimental'

#COVID-DX specific
timecourses_err = data_dictionary["timecourses_err"]
timecourses =  data_dictionary["timecourses"] 
# df_data = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED DATA EXP.pkl')
# df_error = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED DATA ERR.pkl')


#Set style file
plt.style.use('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/paper.mplstyle.py')

#ignore ODEint warnings that clog up the console
warnings.filterwarnings("ignore")


# =============================================================================
# General parameter estimation/solver code (modules 1, 2, 3)
# =============================================================================    
save_internal_states_flag == False

def check_filters(solutions: list, mse: float , doses: list, p: str) -> float:
    """
    Checks whether simulation results associated with a given parameter set pass the cost function filters
        
    Parameters
    ----------
    solutions
        a list of floats containing the solutions associated with parameter set p 
       
        
    mse
        a float or int defining the original mse value before filtering 
        
    doses
        list of lists containing the conditions
    
    p
         a list of floats containing the parameter set (order of parameter defined in 
            Settings.py and in the ODE defintion function in Solvers.py)
         labels for p = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA', 'k_loc_deactivation', 'k_scale_deactivation']


    Returns
    -------
    mse
        a float or int defining the  mse value after filtering  """
        
    filter_code = 0
    #max val filter
    if max(solutions) < 2000:
        filter_code = 1
        
    # low iCas13 filter
    # else:
    #     doses = [5.0, 0.5, 0.005, 1, 4.5]
    #     t, solutions_all, reporter_timecourse = solveSingle(doses, p, model)
    #     final_timepoint_iCas13 = reporter_timecourse[-1] #no norm
    #     max_high_iCas13 = max(solutions) #no norm
    #     ratio_2 = final_timepoint_iCas13/max_high_iCas13
    #     if ratio_2 > 0.10:
    #         filter_code = 2
            
    mse = max(mse, filter_code)
            
    return float(mse)


def solveAll(p: list, exp_data: list, output: str) -> Tuple[list, list, float, pd.DataFrame]:
    """
    Solves ODEs for the entire dataset using parameters defined in p 
        
     
    Parameters
    ----------
    p
         a list of floats containing the parameter set (order of parameter defined in 
            Settings.py and in the ODE defintion function in Solvers.py)
         labels for p = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA', 'k_loc_deactivation', 'k_scale_deactivation']
        
    exp_data
        a list of floats containing the experimental data (length = # data points)

    output
        a string defining the desired output: '' or 'all states'

    Returns
    -------
    x
        list of lists containing the conditions
        
    solutions_norm
        list of floats containing the normalized simulation values
        
    mse
        float or int defining the chi_sq/number of data points
        
    dfSimResults
        df containing the normalized simulation values
        
    if output == 'all states': 
    df_all_states
        df containing all model states simulation values"""

    df_all_states = pd.DataFrame(
        index=model_states,
        columns = [str(i) for i in x],
        dtype=object
    )
    
    dfSimResults = pd.DataFrame()
    solutions = []
    full_solutions_all = []
    for doses in x: #For each condition (dose combination) in x 

        t, solutions_all, reporter_timecourse = solveSingle(doses, p, model)
        solutions_all_arr = np.concatenate(solutions_all) #save solutions for all states as a 1D array
        # print(np.shape(solutions_all))
        # print(np.shape(solutions_all_arr))
        full_solutions_all.append(solutions_all_arr) #add to list of all state solutions for full dose set

        if len(reporter_timecourse) == len(t):
            reporter_timecourse = reporter_timecourse
            
        else:
            reporter_timecourse = [0] * len(t)
        
        for i in reporter_timecourse:
            solutions.append(float(i))
            
        dfSimResults[str(doses)] = reporter_timecourse
        
        if output == 'all states':
            for i, state in enumerate(model_states):
                df_all_states.at[state, str(doses)] = solutions_all[i]
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
   
    #Check for Nan
    i = 0
    for item in solutions_norm:
        if math.isnan(item) == True:
            print('Nan in solutions')
            chi_sq = 1e10
            return x, solutions_norm, chi_sq, dfSimResults

    #Calculate cost function 
    chi_sq = calc_chi_sq(exp_data, solutions_norm)
    mse = chi_sq/len(solutions_norm)
    mse = check_filters(solutions, mse, doses, p)
   
    if output == 'all states':
        for index, row in df_all_states.iterrows():
            row_l = np.concatenate(row.values)
            df_all_states.loc[index, 'is_negative'] = np.any(row_l < 0.0)
            df_all_states.loc[index, 'min_val'] = min(row_l)

        return x, solutions_norm, mse, dfSimResults, df_all_states
    
    elif output == 'check negative':
        full_solutions_all_arr = np.concatenate(full_solutions_all) #save solutions for all states/ full dose set as a 1D array
        is_negative = np.any(full_solutions_all_arr < 0.0) #for full simulation all states, check if any vals are negative
        # print(is_negative)
        # print(np.shape(full_solutions_all))
        # print(np.shape(full_solutions_all_arr))
        return is_negative, full_solutions_all_arr
        
    else:
        return x, solutions_norm, mse, dfSimResults

    
def calculate_mse_k_PEM_evaluation(k_PEM_evaluation: int, df: pd.DataFrame) -> list:
    """Calculates the mse with respect to the given PEM evaluation dataset
     
    Parameters
    ----------
    k_PEM_evaluation
        an integer defining the identity of the PEM evaluation data set
        
    df
        a data frame containing the parameter sets and simulated results from a global search

    Returns
    -------
    mse_list_PEM_evaluation
         a list of floats containing the mse values for each parameter set in df
    """
    
    mse_list_PEM_evaluation = []
    mse_original_data = list(df['chi_sq'])
    for i, solutions in enumerate(list(df['normalized solutions'])):
        for item in solutions:
            if math.isnan(item) == True:
                print('Nan in solutions')
                chi_sq = 1e10
                
        if mse_original_data[i] >= 1:
            mse = mse_original_data[i]
        else:
            chi_sq = calc_chi_sq(exp_data, solutions)
            mse = chi_sq/len(solutions)
       
        mse_list_PEM_evaluation.append(mse)
      
    return mse_list_PEM_evaluation


def solvePar(row: tuple):
    """Solve sODEs for the parameters defined in row 
    (can be called directly by multiprocessing function)
     
    Parameters
    ----------
    row
        a tuple containing the row of the dataframe containing the parameters

    Returns
    -------
    mse
        a float containing the mean squared error for the given parameter set
    
    norm_solutions
        a list of floats containing the normalized simulated solutions
       
    """
   
    #Define parameters and solve ODEs
    p = []
    for i in range(1, len(p_all) + 1):
        p.append(row[i])
    
    print('p: ' + str(p))
    x, norm_solutions, mse, df_sim = solveAll(p, exp_data, '')

    print('mse: ' + str(round(mse, 6)))
    print('**************')
    
    if run_type == 'generate PEM evaluation data':
        return mse, norm_solutions
    else: 
        return mse


def optPar(row: tuple) -> Tuple[list, list]: 
    """Performs optimization with initial guesses as defined in row 
     (can be called directly by multiprocessing function)
     
    Parameters
    ----------
    row
        a tuple containing the row of the dataframe containing the optimization conditions

    Returns
    -------
    results_row
        a list of floats, strings, and lists containing the optimization results
    
    results_row_labels
        a list of stringe containing the labels to go along with resultsRow
       
    """
 
    #Unpack row
    count = row[0] + 1
    p = row[-3]
    exp_data = row[-1]
    fit_param_labels = row[-2]

    #Drop index 0 and 1 (count)
    row = row[2:]
    
    #Initialize list to keep track of CF at each function evaluation
    chi_sq_list = []

    def solveForOpt(x, p1, p2, p3, p4, p5, p6, p7):
        #This is the function that is solved at each step in the optimization algorithm
        #Solve ODEs for all data_sets
        p = [p1, p2, p3, p4, p5, p6, p7]
        doses, norm_solutions, mse, df_sim = solveAll(p, exp_data, '')
        print('eval #: ' + str(len(chi_sq_list)))
        print(p)
        print('mse: ' + str(mse))
        print('***')
        chi_sq_list.append(mse)
        return np.array(norm_solutions)
        
    #Set default values
    bound_min_list = [0] * (len(all_param_labels))
    bound_max_list = [np.inf] * (len(all_param_labels))
    vary_list = [False] * (len(all_param_labels))
   
    for param_index in range(0, len(all_param_labels)): #for each param in p_all
        for fit_param_index in range(0, len(fit_param_labels)): #for each fit param
        
            #if param is fit param, change vary to True and update bounds
            if all_param_labels[param_index] == fit_param_labels[fit_param_index]: 
                vary_list[param_index] = True
                bound_min_list[param_index] = 10 ** bounds[fit_param_index][0]
                bound_max_list[param_index] = 10 ** bounds[fit_param_index][1]
   
    #Add parameters to the parameters class
    params = Parameters()
    for index_param in range(0, len(all_param_labels)):
        params.add(all_param_labels[index_param], value=p[index_param], 
                   vary = vary_list[index_param], min = bound_min_list[index_param], 
                   max = bound_max_list[index_param])
    
    #Set conditions
    method = 'leastsq'
    model_ = Model(solveForOpt, nan_policy='propagate')
        
    #Perform fit and print results
    results = model_.fit(exp_data, params, method=method, x=x)
    print('Optimization round ' + str(count) + ' complete.')
   
    #add initial params to row for saving 
    result_row = p
    result_row_labels = real_param_labels_all
    
    #get best fit params    
    best_fit_params = results.params.valuesdict()
    best_fit_params_list = list(best_fit_params.values())
    
    #Solve with final optimized parameters and calculate chi_sq
    doses_ligand, norm_solutions, chi_sq, df_sim  = solveAll(best_fit_params_list, exp_data, '')
  
    result_row.append(chi_sq)
    result_row_labels.append('chi_sq')
    
    #append best fit params to results row for saving
    for index in range(0, len(best_fit_params_list)):
        result_row.append(best_fit_params_list[index])
        fit_param_label = real_param_labels_all[index] + '*'
        result_row_labels.append(fit_param_label)
    
    #Define other conditions and result metrics and add to result_row for saving
    items = [method,  results.success, results.ier,
             results.lmdif_message, model, chi_sq_list,
             norm_solutions]
    item_labels = ['method', 'success', 'integer flag',
                   'lmdif message', 'model', 'chi_sq_list', 
                   'Simulation results']
   
    # print('integer flag: ', results.ier)
    # print('lmdif message: ', results.lmdif_message)

    for i in range(0, len(items)):
        result_row.append(items[i])
        result_row_labels.append(item_labels[i])

    ### val = num params *2 + 8 (original code: val = num params *2 + 6)  #26 for modelD 
    result_row = result_row[:22]
    result_row_labels = result_row_labels[:22]
    return result_row, result_row_labels


def handler(signum, frame):
    """Registers handler for timeout function"""
    raise Exception("end of time")
signal.signal(signal.SIGALRM, handler)  
 
# =============================================================================
#  Module 2 - code for parameter estimation using training data
# =============================================================================  
def runParameterEstimation() -> Tuple[pd.DataFrame, list, list, pd.DataFrame]:
    """Runs PEM (global search, filter, and optimization)
    Parameters
    ----------
    None

    Returns
    -------
    df
        df containing results of the optimization simulations
        
    best_case_params
        list of floats defining the best parameter values following optimization
        
    norm_solutions_best_case
        list of floats containing the normalized simulated data generated with the best parameter values
        
    df_sim_best_case
        df containing the normalized simulated data generated with the best parameter values
        
    Files
    -------  
    'PARAM SWEEP.xlsx'
        dataframe containing the parameters used in the parameter sweep
        
    'GLOBAL SEARCH RESULTS.xlsx'
        dataframe containing results of the global search
        
    'INITIAL GUESSES.xlsx'
        dataframe containing the filtered results of the global search
        
    'OPT RESULTS.xlsx'
        dataframe containing results of the optimization algorithm'''
        
    """
     
    '''1. Global search'''   
    #use results from previous global search used to generate PEM evaluation data
    if data == 'PEM evaluation': 
        df_results = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230924_ModelA_PEM_rep3/GENERATE PEM EVALUATION DATA/' + 'GLOBAL SEARCH RESULTS ' + model + '.pkl')
        mse_values_PEM_evaluation_data = calculate_mse_k_PEM_evaluation(k_PEM_evaluation, df_results)
        label = 'chi_sq_' + str(k_PEM_evaluation)
        df_results[label] = mse_values_PEM_evaluation_data

    # elif data == 'slice drop high error' and model == 'model A':
    #     df_results = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230922_ModelA_PEM_rep1/GENERATE PEM EVALUATION DATA/' + 'GLOBAL SEARCH RESULTS ' + model + '.pkl')
    
    # elif data == 'rep2 slice drop high error' and model == 'model A':
    #     df_results = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230923_ModelA_PEM_rep2/GENERATE PEM EVALUATION DATA/' + 'GLOBAL SEARCH RESULTS ' + model + '.pkl')

    elif data == 'rep3 slice drop high error' and model == 'model A':
        df_results = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230924_ModelA_PEM_rep3/GENERATE PEM EVALUATION DATA/' + 'GLOBAL SEARCH RESULTS ' + model + '.pkl')

    #run global seach
    else:
        df_params = generateParams(problem_free, n_search, p_all, problem_all_params, model, data)
        
        print('Starting global search...')
        #set parallelization condition for GS
        if conditions_dictionary['parallelization'] == 'yes':
            parallelization_GS = 'yes'
        else:
            parallelization_GS = 'no'
        if model == 'model A' or model == 'model C' or model == 'model D':
            parallelization_GS = 'no'
            
        #perform GS without parallelization
        if parallelization_GS == 'no':
            output = []
            for row in df_params.itertuples(name = None):
                signal.alarm(100)
                try:
                    result = solvePar(row)
               
                except Exception:
                    print('timed out')
                    result = 3
                finally:
                    signal.alarm(0)
         
                output.append(result)
   
        
        #perform GS with parallelization
        elif parallelization_GS == 'yes':  ###with multiprocessing###
            with mp.Pool(conditions_dictionary["num_cores"]) as pool:
              result = pool.imap(solvePar, df_params.itertuples(name = None))
              pool.close()
              pool.join()
              output = [[round(x,4)] for x in result]
                    
        #Restructure global search results
        chi_sq_list = []
        for pset in range(0, len(output)):
            chi_sq_list.append(output[pset])
 
        df_results = df_params
        df_results['chi_sq'] = chi_sq_list

        with pd.ExcelWriter('GLOBAL SEARCH RESULTS.xlsx') as writer:  # doctest: +SKIP
            df_results.to_excel(writer, sheet_name='GS results')
        print('Global search complete.')
        

    '''2. Filter'''
    #set column to sort by
    if data == 'PEM evaluation':
        sort_column = 'chi_sq_' + str(k_PEM_evaluation)
    else:
        sort_column = 'chi_sq'
    
    #filter global search data
    filtered_data, initial_guesses = filterGlobalSearch(df_results, n_initial_guesses, 
                                                        all_param_labels, sort_column)
    print('Filtering complete.')
    
    '''3. Optimization'''
    print('Starting optimization...')
    df = initial_guesses.copy(deep=True)
    df['exp_data'] = [exp_data] * len(df.index)
    # ig_df = initial_guesses.copy(deep=True)

    # is_negative_list = []
    # full_solutions_arr = []
    # for row in ig_df.itertuples(name = None):
    #     ig_params = row[-2]
    #     is_negative, full_solutions = solveAll(ig_params, exp_data, 'check negative')
    #     is_negative_list.append(is_negative)
    #     full_solutions_arr.append(full_solutions)
    # ig_df['negative vals?'] = is_negative_list
    # ig_df['full solutions'] = full_solutions_arr

    #Save initial guesses df with negative vals check
    # filename = 'INITIAL GUESSES NEGATIVE CHECK'
    # with pd.ExcelWriter(filename + '.xlsx') as writer:  # doctest: +SKIP
    #     ig_df.to_excel(writer, sheet_name='initial_guesses')

    all_opt_results = []
    
    if parallelization == 'no':  ###without multiprocessing###
        for row in df.itertuples(name = None):
            signal.alarm(2000)
            try:
                result_row, result_row_labels = optPar(row)
                all_opt_results.append(result_row)
            except Exception:
                print('timed out')
                # result_row = [0] * 26
            finally:
                signal.alarm(0)
            # all_opt_results.append(result_row)

    elif parallelization == 'yes':  ###with multiprocessing###
        with mp.Pool(num_cores) as pool:
            result = pool.imap(optPar, df.itertuples(name=None))
            pool.close()
            pool.join()
            output = [[list(x[0]), list(x[1])] for x in result]
            
        for ig in range(0, len(output)):
            all_opt_results.append(output[ig][0])
            result_row_labels = output[ig][1]
            
    print('Optimization complete.')
    
    #Save results of the opt
    df_opt = pd.DataFrame(all_opt_results, columns = result_row_labels)
    
    #Sort by chi_sq
    df = df_opt.sort_values(by=['chi_sq'], ascending = True)
    
    #Save results of the opt before calculating R2 values
    with pd.ExcelWriter('OPT RESULTS BEFORE.xlsx') as writer:  # doctest: +SKIP
        df.to_excel(writer, sheet_name='OPT Results')
    
    #Save best case calibrated params (lowest chi_sq)
    best_case_params = []
    for i in range(0, len(p_all)):
        col_name = real_param_labels_all[i] + '*'
        val = df[col_name].iloc[0]
        best_case_params.append(round(val, 8))
        
    #Plot training data and model fits for best case params
    doses, norm_solutions_best_case, chi_sq, df_sim_best_case = solveAll(best_case_params, exp_data, '')
    parityPlot(norm_solutions_best_case, exp_data, data)
        
    #Calculate R2 for each optimized parameter set
    Rsq_list = []
    for j in range(0, n_initial_guesses):
        params = []
        for i in range(0, len(p_all)):
            col_name = real_param_labels_all[i] + '*'
            val = df[col_name].iloc[j]
            params.append(val)
        doses, norm_solutions, chi_sq, df_sim = solveAll(params, exp_data, '')
        Rsq = calcRsq(norm_solutions, exp_data)  
        Rsq_list.append(Rsq)
    df['Rsq'] = Rsq_list

    #Save results of the opt
    with pd.ExcelWriter('OPT RESULTS.xlsx') as writer:  # doctest: +SKIP
        df.to_excel(writer, sheet_name='OPT Results')
    
    '''FIT CRITERIA'''
    print('*************************')
    print('')
    print('Calibrated parameters: ' + str(best_case_params))
    print('')
    R2opt_max = round(df['Rsq'].iloc[0], 3)
    print('Rsq = ' + str(R2opt_max))
    print('')
    chi_sq_opt_min = round(df['chi_sq'].iloc[0], 3)
    print('chi_sq = ' + str(chi_sq_opt_min))
    print('')
    print('*************************')
    return df, best_case_params, norm_solutions_best_case, df_sim_best_case

def simLowCas13(p: list) -> None:
    """Generates data for plotting low vs high Cas13a-gRNA conditions
    Parameters
    ----------
    p
        a list of floats containing the parameter set

    Returns
    -------
    None
    """
    data = 'all echo drop high error'
    conditions_dictionary["data"] = data
    x, exp_data, error, timecourses, timecourses_err = defineExp(conditions_dictionary["data"], conditions_dictionary["model"])
    data_dictionary["x_vals"] = x
    data_dictionary["exp_data"] = exp_data
    x, solutions, mse, df_sim = solveAll(p, exp_data, '')
    plotLowCas13(df_sim, 'sim')

# =============================================================================
#  Module 1 - code to generate and simulate PEM EVALUATION data
# =============================================================================
def savePemEvalData(df_PEM_evaluation: pd.DataFrame, filename: str, count: int) -> None:
    """Saves PEM evaluation data in format usable by downstream code

    Parameters
    ----------
    df_PEM_evaluation
        a df containing the normalized reporter expression values for each datapoint 
            
    filename
        a string defining the filename to save the results to
        
    count
        an integer defining the data set number that is being saved 
  
    Returns
    -------
    None
    
    Files
    ----------
    filename + data_type + '.xlsx' (df_PEM_evaluation)  
    """
    
    filename = filename + ' ' + model + '.xlsx'

    if count==1:
        with pd.ExcelWriter(filename) as writer:
            df_PEM_evaluation.to_excel(writer, sheet_name = str(count))
            
    else: 
        path = filename
        book = load_workbook(path)
        writer = pd.ExcelWriter(path, engine = 'openpyxl')
        writer.book = book      
        df_PEM_evaluation.to_excel(writer, sheet_name = str(count))
        writer.save()
        writer.close() 

 
def defineMeasurementErrorModel_PEM_Eval() -> Tuple[list, list]:
    """Calculates the mean and standard deviation of the measurement error 
    (standard error) for each condition (set of component doses)

    Parameters
    ----------
    None 
  
    Returns
    -------
    mean_list
        a list of floats containing the means of the measurement error associated with each condition
        for example, the 0th index contains the mean measurement error for all time points in the 0th condition
        
    mean_list
        a list of floats containing the standard deviation of the measurement error associated with each condition
        
    """
    
    standard_error = [i/math.sqrt(3) for i in error]
    error_lists = list(chunks(standard_error, 61))
    error_lists = [list_ for list_ in error_lists] 
    
    mean_list = []
    sd_list = []
    for list_ in error_lists:
        mean_list.append(np.mean(list_[10:])) #remove first 10 data points
        sd_list.append(np.std(list_[10:])) #remove first 10 data points
        
    return mean_list, sd_list
        
def generatePemEvalData(df_global_search: pd.DataFrame) -> list:
    """Generates PEM evaluation data based on results of a global search

    Parameters
    ----------
    df_global_search
        a dataframe containing global search results

    Returns
    -------
    df_list
        a list of dataframes containing the PEM evaluation data
    
    Files
    ----------
    "PEM evaluation criterion.json"
        contains PEM evaluation criterion using both chi_sq and R_sq
    
    """
    
    saveConditions(conditions_dictionary, initial_params_dictionary, data_dictionary)
    
    #filter data to choose parameter sets used to generate PEM evaluation data
    filtered_data, df_params  = filterGlobalSearch(df_global_search, num_datasets_pem_eval, 
                                                   all_param_labels, 'chi_sq')
    
    #Define measurement error metrics for each condition
    mean_list, sd_list = defineMeasurementErrorModel_PEM_Eval()
    
    #Define, add noise to, and save PEM evaluation data
    count = 1
    df_list = []
    Rsq_list = []
    chi_sq_list = []
    for row in df_params.itertuples(name = None):
        #Define parameters
        p = []
        for i in range(2, len(p_all) + 2):
            p.append(row[i])
            
        #Solve for raw data
        doses_ligand, norm_solutions, chi_sq, df_sim = solveAll(p, exp_data, '')
    
        #Add noise
        df_noise = pd.DataFrame()
        noise_solutions = []
        i = 0
        for column in df_sim:
            noise_col = addNoise(list(df_sim[column]), mean_list[i], sd_list[i])
            df_noise[column] = noise_col
            noise_solutions = noise_solutions + noise_col
            i += 1
        
        #Re-normalize data with noise
        max_vals = df_noise.max()
        max_val = max_vals.max()
        df_noise = df_noise.div(max_val)
        
        #Calculate cost function metrics between PEM evaluation training data with and without noise
        Rsq = calcRsq(norm_solutions, noise_solutions)
        Rsq_list.append(Rsq)
        
        chi_sq = calc_chi_sq(norm_solutions, noise_solutions)
        mse = chi_sq/len(norm_solutions)
        chi_sq_list.append(mse)
   
        #Save results
        savePemEvalData(df_sim, 'PEM EVALUATION DATA RAW', count)
        savePemEvalData(df_noise, 'PEM EVALUATION DATA NOISE', count)
        df_list.append(df_noise)
        count += 1
    
    #Define PEM evaluation criterion
    mean_Rsq = np.round(np.mean(Rsq_list), 4)
    min_Rsq = np.round(min(Rsq_list), 4)
    print('Mean R2 between PEM evaluation data with and without noise: ' + str(mean_Rsq))
    print('Min R2 between PEM evaluation data with and without noise: ' + str(min_Rsq))
    
    mean_chi_sq = np.round(np.mean(chi_sq_list), 4)
    max_chi_sq = np.round(max(chi_sq_list), 4)
    print('Mean chi_sq between PEM evaluation data with and without noise: ' + str(mean_chi_sq))
    print('Max chi_sq between PEM evaluation data with and without noise: ' + str(max_chi_sq))
    
    #Save PEM evaluation criterion
    with open("PEM evaluation criterion.json", 'w') as f:
        json.dump(Rsq_list, f, indent=2) 
        json.dump(chi_sq_list, f, indent=2) 
  
    return df_list

def addNoise(raw_vals: list, mu: float, sigma: float) -> list:
    """Adds technical error to a dataset, according to a normal distribution 
        (defined by a mean and standard deviation)

    Parameters
    ----------
    raw_vals
        a list of floats defining the values before technical error is added 
         (length = # datapoints)
         
     mu
         a float defining the mean of measurement error distribution for the given condition
         
     sigma
         a float defining the standard deviation of measurement error distribution for the given condition


    Returns
    -------
    noise_vals
        a list of floats defining the values (raw_vals) after technical error is added 
        (length = # datapoints)
    
    """

   
    noise_vals = []
    count_val = 0

    for i, val in enumerate(raw_vals):
        #do not add noise to the first 10 values
        if i < 10:
            new_val = val
            
        #for each value, randomly generate an error value and add or substract from raw value
        else:
            #set defaults
            new_val = -1
            noise = -1    
            count_val += 1
            count = 0
            
            #Try again if any value goes below 0 with the addition of noise
            while new_val < 0 or noise < 0:  
                count += 1
            
                #Generate error value
                noise = float(np.random.normal(mu, sigma, 1))
                
                #Determine whether to add or subtract
                k = random.randint(0, 1)
                
                #Calculate new value
                if k == 0:
                    new_val = val - noise
                elif k == 1:
                    new_val = val + noise
            
        noise_vals.append(new_val)
  
    return noise_vals


def runGlobalSearchPemEval() -> pd.DataFrame:
    """Runs global search to generate and define PEM evaluation data

    Parameters
    ----------
    None


    Returns
    -------
    df_results
        a dataframe containing the results of the global search
    
    Files
    -------
        './GLOBAL SEARCH RESULTS.xlsx' (df_results in Excel form)
        './GLOBAL SEARCH RESULTS.pkl' (df_results in pickle form)
    
    """
    #Perform global search
    df_params = generateParams(problem_free, n_search, p_all, problem_all_params, model, data)
    chi_sq_list = []
    norm_solutions_list = []
    if parallelization == 'yes':
        with mp.Pool(conditions_dictionary["num_cores"]) as pool:
            result = pool.imap(solvePar, df_params.itertuples(name = None))
            pool.close()
            pool.join()
            output = [[round(x[0],4), x[1]] for x in result]
            
        #Restructure results
        for pset in range(0, len(output)):
            chi_sq_list.append(output[pset][0])
            norm_solutions_list.append(output[pset][1])
       
            
    elif parallelization == 'no':
        for row in df_params.itertuples(name = None):
            signal.alarm(100)
            try:
                chi_sq, norm_solutions = solvePar(row)
            except Exception:
                print('timed out')
                chi_sq = 3
                norm_solutions = [0] * len(exp_data)
            finally:
                signal.alarm(0)
            chi_sq_list.append(chi_sq)
            norm_solutions_list.append(norm_solutions)
            
    df_results = df_params
    df_results['chi_sq'] = chi_sq_list
    df_results['normalized solutions'] = norm_solutions_list
   
    #Save results
    filename = './GLOBAL SEARCH RESULTS ' + model
    df_results.to_pickle(filename + '.pkl')
    with pd.ExcelWriter(filename + '.xlsx') as writer: 
        df_results.to_excel(writer, sheet_name=' ')
    print('Global search complete')
    
    return df_results


# =============================================================================
# Code for k-fold cross validation
# =============================================================================
def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst
    Parameters
    ----------
    lst
        a list of values
        
    n
        an integer defining the size of each chunk

    Returns
    -------
    lst[i:i + n]
        a list of lists containing the values from lst structured as n-sized chunks
    
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
            
def defineExpData_CV(data_lists: list, n_indicies: int, n_sets: int) -> Tuple[list, list, list, list]:
    """Partitions data for cross-validation
    Parameters
    ----------
    data_list
        a list of lists containing the experimental data to choose from 
        
    n_indicies
        an integer defining the number of conditions used to define the training data for each set
        
    n_sets
        an integer defining the number of sets to generate

    Returns
    -------
    x_CV_train
        a list of lists containing the conditions used to define each set of training data (total length = n_sets)
    
    exp_data_CV_train
        a list of lists containing the experimental data used to define each set of training data (total length = n_sets)
    
    x_CV_test
        a list of lists containing the conditions used to define each set of test data 
    
    exp_data_CV_test
        a list of lists containing the experimental data used to define each set of test data 
        
    Files
    -------
    CV INDICIES.svg
        scatter plot showing the indices used to define the training data for each k_CV
        
    'x_CV_train.json'
        .json file containing the data from x_CV_train
        
    'x_CV_test.json'
        .json file containing the data from x_CV_test
    
    """

    def stratifyData(n_indicies: int, n_attempt: int, seed_: int) -> Tuple[list, list, list, list, list, list, list, list, list, list]:
        """Attempts to stratify training data for cross validation 
        Parameters
        ----------
     
        n_indicies
            an integer defining the number of conditions used to define the training data for each set
            
        n_attempt
            an integer defining attempt number 
            
        seed_
            an integer defining the seed for the random number generator 
    
        Returns
        -------
        indicies_train
            a list of integers defining the indices that were stratified into training data
        
        indicies_test
            a list of integers defining the indices that were stratified into test data
        
        labels_train
            a list of strings defining the enzyme conditions that were stratified into training data
        
        labels_test
            a list of strings defining the enzyme conditions that were stratified into test data
        
        T7_doses_train_1
            a list of floats defining T7 RNAP doses that were included in the training data at 1fM input RNA
        
        RT_doses_train_1
            a list of floats defining RT doses that were included in the training data at 1fM input RNA
        
        RNAse_doses_train_1
            a list of floats defining RNAse doses that were included in the training data at 1fM input RNA
        
        T7_doses_train_10
            a list of floats defining T7 RNAP doses that were included in the training data at 10fM input RNA
        
        RT_doses_train_10
             a list of floats defining RT doses that were included in the training data at 10fM input RNA
        
        RNAse_doses_train_10
            a list of floats defining RNAse doses that were included in the training data at 10fM input RNA
        """
        #randomly stratify data
        print('*************')
        random.seed(seed_) 
        indicies_train = random.sample(range(0, len(x)), n_indicies)
        
        indicies_test = []
        for i in range(0, len(data_lists)):
            if i not in indicies_train:
                indicies_test.append(i)
   
        #restructure labels
        labels_1 = []
        labels_10 = []
        labels_test = []
        labels_train = []
        
        for index in indicies_test:
            label = x[index]
            labels_test.append(label)
            
        for index in indicies_train:
            label = x[index]
            labels_train.append(label)
            if label[3] == 1.0:
                labels_1.append(label)
            elif label[3] == 10.0:
                labels_10.append(label)
       
        T7_doses_train_1 = [list_[0] for list_ in labels_1]
        RT_doses_train_1 = [list_[1] for list_ in labels_1]
        RNAse_doses_train_1 = [list_[2] for list_ in labels_1]

        T7_doses_train_10 = [list_[0] for list_ in labels_10]
        RT_doses_train_10 = [list_[1] for list_ in labels_10]
        RNAse_doses_train_10 = [list_[2] for list_ in labels_10]
    
        return indicies_train, indicies_test, labels_train, labels_test, T7_doses_train_1, RT_doses_train_1, RNAse_doses_train_1, T7_doses_train_10, RT_doses_train_10, RNAse_doses_train_10

    def checkStratifiedData(labels_train: list, T7_doses_train_1: list, RT_doses_train_1: list, RNAse_doses_train_1: list, T7_doses_train_10: list, RT_doses_train_10: list, RNAse_doses_train_10: list) -> bool:
        """Checks whether the stratified data meets the requirements  
        Parameters
        ----------
        labels_train
            a list of strings defining the enzyme conditions that were stratified into training data
        
        T7_doses_train_1
            a list of floats defining T7 RNAP doses that were included in the training data at 1fM input RNA
        
        RT_doses_train_1
            a list of floats defining RT doses that were included in the training data at 1fM input RNA
        
        RNAse_doses_train_1
            a list of floats defining RNAse doses that were included in the training data at 1fM input RNA
        
        T7_doses_train_10
            a list of floats defining T7 RNAP doses that were included in the training data at 10fM input RNA
        
        RT_doses_train_10
             a list of floats defining RT doses that were included in the training data at 10fM input RNA
        
        RNAse_doses_train_10
            a list of floats defining RNAse doses that were included in the training data at 10fM input RNA
     
    
        Returns
        -------
        flag
            a boolean representing whether there is a problem with this attempt (True) or not (False)
        
        """
        flag = False
        
        #max readout condition
        max_readout_label = [1.0, 2.5, 0.005, 10, 90]
        if max_readout_label not in labels_train:
            flag = True
            print('max readout flag')
            
        #vrna = 1
        T7_vals = [1.0, 5.0, 20.0]
        for item in T7_vals:
            if item in T7_doses_train_1:
                continue
            else:
                flag = True
                print('T7 flag')

        RT_vals = [0.5, 2.5, 10.0]
        for item in RT_vals:
            if item in RT_doses_train_1:
                continue
            else:
                flag = True
                print('RT flag')

        RNAse_vals = [0.001, 0.005, 0.02]
        for item in RNAse_vals:
            if item in RNAse_doses_train_1:
                continue
            else:
                flag = True
                print('RNAse flag')

        #vrna = 10
        T7_vals = [1.0, 5.0, 20.0]
        for item in T7_vals:
            if item in T7_doses_train_10:
                continue
            else:
                flag = True
                print('T7 flag')

        RT_vals = [0.5, 2.5, 10.0]
        for item in RT_vals:
            if item in RT_doses_train_10:
                continue
            else:
                flag = True
                print('RT flag')

        RNAse_vals = [0.001, 0.005, 0.02]
        for item in RNAse_vals:
            if item in RNAse_doses_train_10:
                continue
            else:
                flag = True
                print('RNAse flag')

        return flag

    def flatten(big_list: list) -> list:
        """flattens a list of lists into a single list
        Parameters
        ----------
        big_list
            a list of lists to be flattened
     
    
        Returns
        -------
        flattened_list
            the flattened list
        
        """
        flattened_list = [item for sublist in big_list for item in sublist]
        return flattened_list
    
    #main code for defineExpData_CV
    #initialize lists
    x_CV_train = []
    x_CV_test = []
    indicies_CV_train = []
    indicies_CV_test = []
    exp_data_CV_train = []
    exp_data_CV_test = []
    num_complete = 0
    
    #set list of seeds
    random.seed(16834)
    seeds = random.sample(range(10000), k=1000)
    
    #try each seed and check whether requirements are met
    for n_attempt in range(1, 1000):
        seed_ = seeds[n_attempt-1]
    
        indicies_train, indicies_test, labels_train, labels_test, T7_doses_train_1, RT_doses_train_1, RNAse_doses_train_1, T7_doses_train_10, RT_doses_train_10, RNAse_doses_train_10 = stratifyData(n_indicies, n_attempt, seed_)
        flag = checkStratifiedData(labels_train, T7_doses_train_1, RT_doses_train_1, RNAse_doses_train_1, T7_doses_train_10, RT_doses_train_10, RNAse_doses_train_10)
        
        #if no flag detected, accept attempt and re-structure 
        if flag == False:
            print('Attempt #: ' + str(n_attempt))
            print('Number complete: ' + str(num_complete))
            print('Complete.')
            x_CV_train.append(labels_train)
            indicies_CV_train.append(indicies_train)
        
            x_CV_test.append(labels_test)
            indicies_CV_test.append(indicies_test)
        
            data_train = []
            data_test = []

            for value in indicies_train:
                list_ = data_lists[value]
                data_train.append(list_)
                
            for value in indicies_test:
                list_ = data_lists[value]
                data_test.append(list_)
            
            data_train = flatten(data_train)
            exp_data_CV_train.append(data_train)
    
            data_test = flatten(data_test)
            exp_data_CV_test.append(data_test)
           
            num_complete += 1
            if num_complete >= n_sets:
                break
                
        #Plot training indicies
        fig = plt.figure(figsize = (4,4))
        for k, list_ in enumerate(indicies_CV_train):
            list_ = [i + 1 for i in list_]
            x_ = [k+1] * len(list_)
            plt.plot(x_, list_, color = 'dimgrey', marker = 'o', linestyle = 'None')
        plt.xticks(range(1, 11))
        plt.yticks([1, 10, 20, 30, 40])
        plt.ylabel('Index in full data set')
        plt.savefig('./SF14b CV INDICIES.svg', dpi = 600)
    
        #save results
        with open("x_CV_train2.json", 'w') as f:
            json.dump(x_CV_train, f, indent=2) 
            
        with open("x_CV_test2.json", 'w') as f:
            json.dump(x_CV_test, f, indent=2) 
         
    return x_CV_train, exp_data_CV_train, x_CV_test, exp_data_CV_test

# =============================================================================
# Code to RUN the workflow
# =============================================================================
if __name__ == '__main__':
    if run_type == 'generate PEM evaluation data':
        #Record start time
        startTime_1 = datetime.datetime.now()
            
        #Set up file structure
        os.chdir(full_path)
        sub_folder_name = 'GENERATE PEM EVALUATION DATA'
        createFolder('./' + sub_folder_name)
        os.chdir('./' + sub_folder_name)
        
        print('Generating PEM evaluation data...')
        #Run the global search and output the df of filtered results
        df_results = runGlobalSearchPemEval()
        
        #Generate PEM evaluation data (add noise and save)
        df_pem_eval = generatePemEvalData(df_results)
    
        
    if run_type == 'parameter estimation':
        #Record start time
        start_time = datetime.datetime.now()

        #Set data_type to experimental 
        data_dictionary["data_type"] = 'experimental'
        data_type = data_dictionary["data_type"]
        
        #Set up file structure and save conditions
        os.chdir(full_path)
        sub_folder_name = 'PARAMETER ESTIMATION ' + data
        if data == 'PEM evaluation':
            sub_folder_name =  sub_folder_name + ' ' + str(k_PEM_evaluation)
            
        createFolder('./' + sub_folder_name)
        os.chdir('./' + sub_folder_name)
        saveConditions(conditions_dictionary, initial_params_dictionary, data_dictionary)
        
        #Run parameter estimation method and analysis
        df_opt, calibrated_params, solutions_best_case, df_sim_best_case = runParameterEstimation()
        
        if data != 'cross-validation train' and data != 'PEM evaluation':
            #Plot parameter distributions 
            plotParamDistributions(df_opt)
            plotParamBounds(calibrated_params, bounds)
            
            #Plot modeling objectives
            plotModelingObjectives123(solutions_best_case)
            
            #Solve equations for data = 'slice'
            plotModelingObjectives456(df_sim_best_case)
        
        #Print stop time
        stop_time = datetime.datetime.now()
        elapsed_time = stop_time - start_time
        elapsed_time_total = round(elapsed_time.total_seconds(), 1)
        elapsed_time_total = elapsed_time_total/60
        print('')
        print('********************************************')
        print('PARAMETER ESTIMATION')
        print('Total run time (min): ' + str(round(elapsed_time_total, 3)))
        print('Total run time (hours): ' + str(round(elapsed_time_total/60, 3))) 
        print('********************************************')

    if run_type == 'generate cross-validation data':
        #Set up file structure and save conditions
        os.chdir(full_path)
        sub_folder_name = 'CROSS VALIDATION'
        createFolder('./' + sub_folder_name)
        os.chdir('./' + sub_folder_name)
        saveConditions(conditions_dictionary, initial_params_dictionary, data_dictionary)
        
        #Subsection data for 
        data_lists = list(chunks(exp_data, 61))
        n_indicies = 14
        n_sets = 10
        run_type = ''
        x_CV_train, exp_data_CV_train, x_CV_test, exp_data_CV_test = defineExpData_CV(data_lists, n_indicies, n_sets)
        

    
        
        



        