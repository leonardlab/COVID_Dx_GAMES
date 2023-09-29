#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:04:22 2022

@author: kate
"""

#Package imports
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

#GAMES imports
from Saving_COVID_Dx import createFolder
import Settings_COVID_Dx
from DefineExpData_COVID_Dx import defineExp
from Solvers_COVID_Dx import  calc_chi_sq, calcRsq

#Unpack conditions from Settings.py
conditions_dictionary, initial_params_dictionary, data_dictionary = Settings_COVID_Dx.init()
exp_data = data_dictionary["exp_data"]
x = data_dictionary["x_vals"]
real_param_labels_all = conditions_dictionary['real_param_labels_all']
param_labels = real_param_labels_all 
model = conditions_dictionary["model"] 
data = conditions_dictionary["data"]
df_data = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED DATA EXP.pkl')
df_error = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED DATA ERR.pkl')

#Import custom style file for plotting
plt.style.use('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/paper.mplstyle.py')
    

dpi_ = 600
colors = ['teal', 'deeppink' ,'rebeccapurple', 'darkseagreen', 'darkorange', 'dimgrey', 
          'crimson', 'cornflowerblue', 'springgreen', 'sandybrown', 'lightseagreen', 'blue', 
          'palegreen', 'lightcoral', 'lightpink', 'lightsteelblue', 'indigo', 'darkolivegreen',
          'maroon', 'lightblue', 'gold', 'olive', 'silver', 'darkmagenta'] * 5 


def analyzeSingleRun(df_opt, count):
    
    '''
    Purpose: Analyze the results of a single PEM run and plot the CF trajectory for each 
             initial guess
    
    Inputs: 
        df_opt: a dataframe containing the optimization results
        count: an integer containing the number of the run
   
    Outputs: 
        list(df['chi_sq']): list of final, optimized chi_sq values across initial guesses 
        list(df['Rsq']): list of final, optimized Rsq values across initial guesses 
    
    Figures:
        'CF TRAJECTORY PLOTS RUN ' + str(count) + '.svg' 
            (plot of CF vs function evaluations for each initial guess, count = run #)
        
    '''
    for i in range(0, df_opt.shape[0]):
        holder = list(df_opt['chi_sq_list'])
        chi_sq_list = holder[i]
        chi_sq_list = chi_sq_list.strip('][').split(', ')
        chi_sq_list = [float(i) for i in chi_sq_list]

    return list(df_opt['chi_sq']),list(df_opt['Rsq'])

def plotPemEvaluation(files, folder_name):
    
    '''
    Purpose: Evaluate and plot PEM evaluation criterion
   
    Inputs: 
        files: a list of dataframes containing the results for comparison
        folder_name: a string containing the name of the folder to save the results to
   
    Outputs: 
        None
    
    Figures: 
        'PEM EVALUATION CRITERION.svg' 
            (plot of optimized Rsq values for each PEM evaluation dataset, 
             shows PEM evaluation criterion)
    '''
    
    createFolder('./Results/' + folder_name)
    os.chdir('./Results/' + folder_name)
    
    run = []
    chi_sq_list = []
    R_sq_list = []
  
    for i, file in enumerate(files):
        df = pd.read_excel(file, engine='openpyxl')
        chi_sq, Rsq = analyzeSingleRun(df, i + 1)
        chi_sq_list = chi_sq_list + chi_sq
        R_sq_list = R_sq_list + Rsq
        run_ = [i + 1] * len(Rsq)
        run = run  + run_
        
    df_all = pd.DataFrame(columns = ['run', 'chi_sq', 'Rsq'])
    df_all['run'] = run
    df_all['chi_sq'] = chi_sq_list
    df_all['Rsq'] = R_sq_list

    #Plot PEM evaluation criterion
    plt.subplots(1,1, figsize=(4,3), sharex = True)
    ax = sns.boxplot(x='run', y='Rsq', data=df_all, color = 'white')
    ax = sns.swarmplot(x='run', y='Rsq', data=df_all, color="black", size = 4)
    ax.set(xlabel='PEM evaluation dataset', ylabel='Rsq, opt')
    plt.savefig('PEM EVALUATION CRITERION R2.svg', dpi = dpi_)
    
    df_all=df_all[df_all.Rsq > 0.9]
    plt.subplots(1,1, figsize=(4,3), sharex = True)
    ax = sns.boxplot(x='run', y='Rsq', data=df_all, color = 'white')
    ax = sns.swarmplot(x='run', y='Rsq', data=df_all, color="black", size = 4)
    ax.set(xlabel='PEM evaluation dataset', ylabel='Rsq, opt')
    plt.savefig('PEM EVALUATION CRITERION R2 >= 0.90.svg', dpi = dpi_)
 

#Note that paths need to be updated before running

#'slice drop high error' #'rep2 slice drop high error' #'rep3 slice drop high error'
if model == 'model A':
    if data == 'slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230922_ModelA_PEM_rep1/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230922_ModelA_PEM_rep1/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230922_ModelA_PEM_rep1/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230922_ModelA_PEM_rep1'
    elif data == 'rep2 slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230923_ModelA_PEM_rep2/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230923_ModelA_PEM_rep2/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230923_ModelA_PEM_rep2/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230923_ModelA_PEM_rep2'
    elif data == 'rep3 slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230924_ModelA_PEM_rep3/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230924_ModelA_PEM_rep3/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230924_ModelA_PEM_rep3/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230924_ModelA_PEM_rep3'

elif model == 'model B':
    if data == 'slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230920_ModelB_PEM_rep1/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230920_ModelB_PEM_rep1/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230920_ModelB_PEM_rep1/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230920_ModelB_PEM_rep1'
    elif data == 'rep2 slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230920_ModelB_PEM_rep2/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230920_ModelB_PEM_rep2/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230920_ModelB_PEM_rep2/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230920_ModelB_PEM_rep2'
    elif data == 'rep3 slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230920_ModelB_PEM_rep3/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230920_ModelB_PEM_rep3/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230920_ModelB_PEM_rep3/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230920_ModelB_PEM_rep3'  

elif model == 'model C':
    if data == 'slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230731_ModelC_PEM_rep1_slice_nofilter_redo/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230731_ModelC_PEM_rep1_slice_nofilter_redo/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230731_ModelC_PEM_rep1_slice_nofilter_redo/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230731_ModelC_PEM_rep1_slice_nofilter_redo'
    elif data == 'rep2 slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230917_ModelC_PEM_rep2/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230917_ModelC_PEM_rep2/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230917_ModelC_PEM_rep2/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230917_ModelC_PEM_rep2'
    elif data == 'rep3 slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230917_ModelC_PEM_rep3/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230917_ModelC_PEM_rep3/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230917_ModelC_PEM_rep3/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230917_ModelC_PEM_rep3'

elif model == 'model D':
    if data == 'slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230906_ModelD_PEM_rep1_beta/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230906_ModelD_PEM_rep1_beta/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230906_ModelD_PEM_rep1_beta/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230906_ModelD_PEM_rep1_beta'
    elif data == 'rep2 slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230901_ModelD_PEM_rep2_beta_redo/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230901_ModelD_PEM_rep2_beta_redo/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230901_ModelD_PEM_rep2_beta_redo/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230901_ModelD_PEM_rep2_beta_redo'
    elif data == 'rep3 slice drop high error':
        path1 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230904_ModelD_PEM_rep3_beta/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
        path2 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230904_ModelD_PEM_rep3_beta/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
        path3 = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230904_ModelD_PEM_rep3_beta/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
        folder_name = '230904_ModelD_PEM_rep3_beta'


files = [path1, path2, path3]
plotPemEvaluation(files, folder_name)

   
    

        