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
import seaborn as sns

#Import custom style file for plotting
plt.style.use('/Users/kate/Documents/GitHub/GAMES_COVID_Dx/paper.mplstyle.py')
    

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
    os.makedirs('./' + folder_name)
    os.chdir('./' + folder_name)
    
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
    plt.savefig('PEM EVALUATION CRITERION R2 greater than 0.90.svg', dpi = dpi_)
 

model = 'model C'

if model == 'model A':
    path1 = '../model A PEM evaluation data 1 5000 + 24/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
    path2 = '../model A PEM evaluation data 2 5000 + 24/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
    path3 = '../model A PEM evaluation data 3 5000 + 24/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
    folder_name = './Model A/PEM evaluation analysis model A'
    
if model == 'model B':
    path1 = '../model B PEM evaluation data 1 5000 + 24/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
    path2 = '../model B PEM evaluation data 2 5000 + 24/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
    path3 = '../model B PEM evaluation data 3 5000 + 24/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
    folder_name = './Model B/PEM evaluation analysis model B'
    
if model == 'model C':
    path1 = '../model C PEM evaluation data 1 5000 + 24/PARAMETER ESTIMATION PEM evaluation 1/OPT RESULTS.xlsx'
    path2 = '../model C PEM evaluation data 2 5000 + 24/PARAMETER ESTIMATION PEM evaluation 2/OPT RESULTS.xlsx'
    path3 = '../model C PEM evaluation data 3 5000 + 24/PARAMETER ESTIMATION PEM evaluation 3/OPT RESULTS.xlsx'
    folder_name = './Model C/PEM evaluation analysis model C'

files = [path1, path2, path3]
plotPemEvaluation(files, folder_name)



        