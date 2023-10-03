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

#Import custom style file for plotting
#Note that this path needs to be updated before running
plt.style.use('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/paper.mplstyle.py')
dpi_ = 600

def analyzeSingleRun(df_opt: pd.DataFrame) -> None:
    
    '''
    Generates a list of chi_sq and Rsq values for a single PEM evalution run
    
    Args: 
        df_opt: a dataframe containing the optimization results
   
    Returns: 
        list(df['chi_sq']): a list of final, optimized chi_sq values across 
            initial guesses 

        list(df['Rsq']): a list of final, optimized Rsq values across initial
            guesses 
    '''

    for i in range(0, df_opt.shape[0]):
        holder = list(df_opt['chi_sq_list'])
        chi_sq_list = holder[i]
        chi_sq_list = chi_sq_list.strip('][').split(', ')
        chi_sq_list = [float(i) for i in chi_sq_list]

    return list(df_opt['chi_sq']),list(df_opt['Rsq'])

def plotPemEvaluation(files: list[str, str, str], folder_name: str) -> None:
    
    '''
    Generates df of chi_2 and Rsq values for each PEM evaluation run and 
    plots as box plot
   
    Args: 
        files: a list of strings containing the file paths of the results
            dataframes for comparison

        folder_name: a string containing the name of the folder to save the
            results to
   
    Outputs: 
        None
    
    Figures: 
        'PEM EVALUATION CRITERION.svg':
            a plot of optimized Rsq values for each PEM evaluation dataset

        'PEM EVALUATION CRITERION R2 >= 0.90.svg':
            a plot of optimized Rsq values >= 0.90 for each PEM evalution
                dataset  
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

   
    

        