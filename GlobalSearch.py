#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kateldray

"""

#Package imports
import numpy as np
import pandas as pd
from SALib.sample import latin

def generateParams(problem, n_search, parameters, problem_all_params, model, data):
    ''' 
    Purpose: Generate parameter sets for global search 
        
    Inputs: 
        problem: a dictionary including the number, labels, and bounds for the free parameters 
            (defined in Settings.py)
            
        n_Search: the total number of parameter sets in the global search
        
        parameters: a list of initial guesses  for each parameter (length equal to number of 
            free parameters, labels based on thse defined in Settings.py)
        
        problem_all_params: a dictionary including the number, labels, and bounds for all 
            potentially free parameters (defined in Settings.py), even if the parameter is not free 
            in this specific simulation
       
        model: a string that defines the model identity
        
        data: a string that defines the data identity
           
    Outputs:
        df_params: a dataframe with columns corresponding to parameter identities 
            (# columns = # parameters) and rows corresponding to parameter values 
            (# rows = # parameter sets)
          
    Files: 
        PARAM SWEEP.xlsx (dataframe df_params from output)
    '''
  
    #Define specific conditions from the dictionaries
    fit_params = problem['names'] #only the currently free parameters (as set in settings)
    num_params = problem['num_vars']
    all_params = problem_all_params['names'] #all params that are potentially free 
    
    #Create an empty dataframe to store results of the parameter sweep
    df_params = pd.DataFrame()

    #Fill each column of the dataframe with the intial values set in Settings. 
    for item in range(0, len(all_params)):
        param = all_params[item]
        param_array = np.full((1, n_search), parameters[item])
        param_array = param_array.tolist()
        df_params[param] = param_array[0]

    #Perform LHS
    param_values = latin.sample(problem, n_search, seed=123) #og seed= 456767, seed2 = 123, seed3 = 654

    #To sample parameters over a log scale, we sample the exponent space and then 
    #transform the values following LHS.
    #Each parameter x generated by the search method is transformed such that the new 
    #parameter to be fed into the model = 10^x
    params_converted = []
    for item in param_values:
        item1 = [10**(val) for val in item]
        params_converted.append(item1)
    params_converted = np.asarray(params_converted)
        
    #Replace the column of each fit parameter with the list of parameters from the sweep
    for item in range(0, num_params):   
        for name in fit_params:
            if fit_params[item] == name:
                df_params[name] = params_converted[:,item]
       
    #add model, names, bounds to df
    m_ = np.full((1, n_search), model)
    m_array = m_.tolist()
    df_params['model'] =  m_array[0]
    
    names = [problem['names']] * n_search
    df_params['names'] =  names
      
    bounds= [problem['bounds']] * n_search
    df_params['bounds'] =  bounds

    #Save df
    filename = './PARAM SWEEP.xlsx'
    with pd.ExcelWriter(filename) as writer:  # doctest: +SKIP
        df_params.to_excel(writer, sheet_name='GSA params')
    
    return df_params        

def filterGlobalSearch(df_results, num_ig, all_param_labels, sort_col):
    '''
    Purpose: Filter results of global search   
    
    Inputs: 
        data: a dataframe containing the results of the global search
        num_ig: the number of initial guesses to choose (this defines the size of the filtered df)
        runID: the runID, which is used for saving purposes
        all_param_labels: a list of labels for all parameters involved in the run 
            (defined in Settings.py)
        sort_col: a string defining the column name that parameter sets should be 
            filtered by (usually 'chi2')
    
    Outputs:
        df_filtered: a dataframe containing the filtered parameter sets, along with all metrics and 
                     condition identities saved in the global search df
                     
        initial_guesses: a dataframe containing only the filtered paramter sets - this is fed 
                     directly into the optimization module
        
    Files:
        'INITIAL GUESSES.xlsx' (dataframe initial_guesses from output)
    
    '''
    
    #Calculate number of rows in input dataframe 
    num_rows = len(df_results.index)
        
    #Initialize dataframe 
    df_filtered = pd.DataFrame()
    df_results = df_results.sort_values(by=[sort_col])
    
    if sort_col != 'chi_sq': #if data is not PEM evaluation data
        #drop first row (this is the parameter set used to generate the PEM evaluation data)
        df_results = df_results.iloc[1: , :]
        
    #If more than the desired number of parameter sets meet the MSE filter requirement, 
    #use rows with lowest values
    if num_ig < num_rows:
        for index1 in range(num_ig):
            df_filtered = df_filtered.append(df_results.iloc[index1])
    
    #if num_ig >= num_rows, use all rows 
    else:
        for index1 in range(num_rows):
            df_filtered = df_filtered.append(df_results.iloc[index1])
    
    #Reset index of df
    df_filtered=df_filtered.reset_index()   
    
    #Save parameter sets remaining after filtering. These are fed directly into 
    #the optimization model as initial guesses.
    grab_cols = []
    columnNames = all_param_labels
    for item in columnNames:
        grab_cols.append(item)

    initial_guesses = df_filtered[grab_cols]
    initial_guesses = pd.DataFrame(initial_guesses, columns = columnNames)
    initial_guesses = initial_guesses.reset_index()  
    
    #Add column to hold list of params
    params_column = []
    for i in range(0,  num_ig):
        params = initial_guesses.loc[i, :].values.tolist()
        params = params[1:]
        params_column.append(params)
    initial_guesses['initial params'] = params_column
    initial_guesses['free params'] = df_filtered['names']
    
    #Save initial guesses df
    filename = 'INITIAL GUESSES'
    with pd.ExcelWriter(filename + '.xlsx') as writer:  # doctest: +SKIP
        initial_guesses.to_excel(writer, sheet_name='initial_guesses')

    return df_filtered, initial_guesses
  
  