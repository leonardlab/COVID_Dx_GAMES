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
        df = pd.read_excel(file)
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
 

model = 'model C'
#Note that filenames may have been changed and paths may need to be updated before running
if model == 'model A':
    path1 = '../2022-06-08 model A PEM evaluation data 1 5000 + 24/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
    path2 = '../2022-06-08 model A PEM evaluation data 2 5000 + 24/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
    path3 = '../2022-06-08 model A PEM evaluation data 3 5000 + 24/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
    folder_name = '2022-06-17 PEM evaluation analysis model A'
    
if model == 'model B':
    path1 = '../2022-06-08 model B PEM evaluation data 1 5000 + 24/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
    path2 = '../2022-06-08 model B PEM evaluation data 2 5000 + 24/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
    path3 = '../2022-06-08 model B PEM evaluation data 3 5000 + 24/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
    folder_name = '2022-06-17 PEM evaluation analysis model B'
    
if model == 'model C':
    path1 = '../2022-06-08 model B PEM evaluation data 1 5000 + 24/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
    path2 = '../2022-06-08 model B PEM evaluation data 2 5000 + 24/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
    path3 = '../2022-06-08 model B PEM evaluation data 3 5000 + 24/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
    folder_name = '2022-06-17 PEM evaluation analysis model C'

files = [path1, path2, path3]
plotPemEvaluation(files, folder_name)





# =============================================================================
# outdated
# =============================================================================
#calculate chi_sq_PEM_eval
# filename = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/PEM evaluation data/GLOBAL SEARCH RESULTS model B TEST.pkl'
# df = pd.read_pickle(filename)
# df = df.sort_values(by=['chi_sq'])
# df = df.reset_index(drop=True)
# chi_sq_list = []
# R2_list = []
# for i in range(0, 1):
#     sim = df['normalized solutions'][i]
#     _, noise_solutions, _, _, _ = defineExp('PEM evaluation', 'model B','', i + 1)
#     #print(noise_solutions)
#     chi_sq = calcchi_sq(sim, noise_solutions)
#     mse = chi_sq/len(sim)
#     chi_sq_list.append(mse)
#     R2 = calcRsq(sim, noise_solutions)
#     R2_list.append(R2)
   
# mean_chi_sq = np.round(np.mean(chi_sq_list), 4)
# max_chi_sq = np.round(max(chi_sq_list), 4)
# print('Mean chi_sq between PEM evaluation data with and without noise: ' + str(mean_chi_sq))
# print('Max chi_sq between PEM evaluation data with and without noise: ' + str(max_chi_sq))

# with open("PEM evaluation criterion chi_sq.json", 'w') as f:
#     json.dump(chi_sq_list, f, indent=2) 
    

   
    

        