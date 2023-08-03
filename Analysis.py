#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:25:15 2020

@author: kate
"""

#Package imports
import sys 
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import log10
from lmfit import Parameters, minimize
from sklearn.linear_model import LinearRegression
from Solvers_COVID_Dx import calcRsq

#GAMES imports
from Saving_COVID_Dx import createFolder
import Settings_COVID_Dx

#Unpack conditions from Settings.py
conditions_dictionary, initial_params_dictionary, data_dictionary = Settings_COVID_Dx.init()
exp_data = data_dictionary["exp_data"]
x = data_dictionary["x_vals"]
real_param_labels_all = conditions_dictionary['real_param_labels_all']
param_labels = real_param_labels_all 
data = conditions_dictionary["data"]

if 'rep2' in data:
    df_data = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED_DATA_rep2_EXP.pkl')
    df_error = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED_DATA_rep2_ERR.pkl') 

elif 'rep3' in data:
    df_data = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED_DATA_rep3_EXP.pkl')
    df_error = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED_DATA_rep3_ERR.pkl')

else:
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
   
    Outputs: 
        list(df['chi2']): list of final, optimized chi2 values across initial guesses 
        list(df['Rsq']): list of final, optimized Rsq values across initial guesses 
    
    Figures:
        'CF TRAJECTORY PLOTS RUN ' + str(count) + '.svg' 
            (plot of CF vs function evaluations for each initial guess, count = run #)
        
    '''
  
    fig1 = plt.figure(figsize=(8,3))
    fig1.subplots_adjust(hspace=.25)
    fig1.subplots_adjust(wspace=0.3)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    df = df_opt
    num_ig = len(list(df['chi2_list']))
    fontsize_ = 12
 
    #Plot CF trajectory 
    for i in range(0, num_ig):
        holder = list(df['chi2_list'])
        chi2_list = holder[i]
        fxn_evals = np.arange(1, len(chi2_list) + 1)
       
        #Plot chi2 (CF) vs. fxn evaluation
        ax1.plot(fxn_evals, chi2_list, color = colors[i], label = 'ig ' + str(i + 1)) 
   
    ax1.set_title('CF tracking', fontweight = 'bold', fontsize = fontsize_ + 2)
    ax1.set_xlabel('Function evaluations', fontweight = 'bold', fontsize = fontsize_)
    ax1.set_ylabel('chi2', fontweight = 'bold', fontsize = fontsize_) 

    #Plot initial and final chi2
    initial_list = []
    final_list = []
    for i in range(0, num_ig):
        holder = list(df['chi2_list'])
        chi2_list = holder[i]
        initial_list.append(chi2_list[0])
        final_list.append(chi2_list[-1])
        fxn_evals = range(0, len(chi2_list))
    
    labels_ = list(range(1, 1 + num_ig))
    y0 = initial_list
    y = final_list
    
    x = np.arange(len(labels_))  # the label locations
    width = 0.35  # the width of the bars
    
    ax2.bar(x - width/2, y0, width, label='ig', color = 'black')
    ax2.bar(x + width/2,y, width, label='opt', color = colors)

    # Add xticks on the middle of the group bars
    ax2.set_xticks(x)
    ax2.set_xlabel('IG #', fontweight = 'bold', fontsize = fontsize_)
    ax2.set_ylabel('chi2', fontweight = 'bold', fontsize = fontsize_) 
    ax2.set_yscale('log')
    
    plt.savefig('CF TRAJECTORY PLOTS RUN ' + str(count)  + '.svg', dpi = dpi_)
    
    return list(df['chi2']),list(df['Rsq'])

def plotCV(files, cal_params, R_sq_list_test, cf_list_test):
    createFolder('./ANALYSIS')
    os.chdir('./ANALYSIS')
    
    #Structure data for plotting
    fontsize_ = 12
    run = []
    chi2_list = []
    R_sq_list = []
    min_cf_list = []
    max_R_sq_list = []
  
    for i, file in enumerate(files):
        chi2 = list(file['chi2'])
        Rsq = list(file['Rsq'])
        chi2_list = chi2_list + chi2
        R_sq_list = R_sq_list + Rsq
        run_ = [i + 1] * len(Rsq)
        run = run  + run_
        
        val, idx = min((val, idx) for (idx, val) in enumerate(chi2))
        min_cf_list.append(val)
        max_R_sq = Rsq[idx]
        max_R_sq_list.append(max_R_sq)  
        
    df_all = pd.DataFrame(columns = ['run', 'chi2', 'Rsq'])
    df_all['run'] = run
    df_all['chi2'] = chi2_list
    df_all['Rsq'] = R_sq_list

    #Plot CV results
    fig1 = plt.figure(figsize=(6,6))
    fig1.subplots_adjust(hspace=.25)
    fig1.subplots_adjust(wspace=0.3)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    #Rsq vs. data set number - all initial guesses
    sns.boxplot(x='run', y='Rsq', data=df_all, color = 'dimgrey', ax = ax1)
    sns.swarmplot(x='run', y='Rsq', data=df_all, color="black", ax = ax1)
    ax1.set(xlabel='training data set k', ylabel='Rsq, opt')
    ax1.set_ylim([0, 1])
    
    #chi2 vs. data set number - all initial guesses
    sns.boxplot(x='run', y='chi2', data=df_all, color = 'dimgrey', ax = ax2)
    sns.swarmplot(x='run', y='chi2', data=df_all, color="black", ax = ax2)
    ax2.set(xlabel='training data set k', ylabel='chi2, opt')

    #plot r2 and chi2 - best values only
    y1 = min_cf_list
    y2 = max_R_sq_list
    labels_ = list(range(1, len(y1) + 1))
    
    x = np.arange(len(labels_))  # the label locations
    width = 0.35  # the width of the bars
    
    ax4.bar(x, y1, width, color = 'dimgrey')
    ax3.bar(x, y2, width,  color = 'dimgrey')

    # Add xticks on the middle of the group bars
    ax3.set_xticks(x)
    ax3.set_xlabel('training data set k', fontweight = 'bold', fontsize = fontsize_)
    ax3.set_ylabel('chi2, opt', fontweight = 'bold', fontsize = fontsize_) 
    
    ax4.set_xticks(x)
    ax4.set_xlabel('training data set k', fontweight = 'bold', fontsize = fontsize_)
    ax4.set_ylabel('R2, opt', fontweight = 'bold', fontsize = fontsize_) 
    ax4.set_ylim([0, 1])
    
    plt.savefig('CROSS VALIDATION R2 chi2 results plots.svg', dpi = dpi_)
    
    #parameter distributions
    param_labels = ['k_cas13*', 'k_degv*', 'k_txn*', 'k_FSS*', 'k_RHA*', 'k_loc_deactivation*', 'k_txn_poisoning*', 'n_txn_poisoning*','k_scale_deactivation*'] 
    df = pd.DataFrame(cal_params, columns = param_labels)
    
    df['Rsq'] = [1] * len(cal_params)
    for label in param_labels:
        new_list = [log10(i) for i in list(df[label])]
        df[label] = new_list
        
    consensus_parameters = []
    for (columnName, columnData) in df.iteritems():
        consensus_parameters.append(np.round(np.mean(columnData), 8))
    consensus_parameters = [10 ** i for i in consensus_parameters]
    print('The consensus parameters are: ' + str(consensus_parameters))
   
    plt.subplots(1,1, figsize=(8,3), sharex = True)
    df = pd.melt(df, id_vars=['Rsq'], value_vars=param_labels)
    ax = sns.boxplot(x='variable', y='value', data=df, color = "dimgrey")
    ax = sns.swarmplot(x='variable', y='value', data=df, color="black")
    ax.set(xlabel='Parameter', ylabel='log(value)')
    plt.savefig('CROSS VALIDATION parameter distributions.svg', dpi = 600)
    
    #mean R2 and chi2 for training data
    mean_Rsq = np.mean(max_R_sq_list)
    mean_chi2 = np.mean(min_cf_list)
    print('The mean R2, train is: ' + str(np.round(mean_Rsq, 3)))
    print('The mean chi2, train is: ' + str(np.round(mean_chi2, 4)))
    
    #mean R2 and chi2 for test data
    mean_Rsq = np.mean(R_sq_list_test)
    mean_chi2 = np.mean(cf_list_test)
    print('The mean R2, test is: ' + str(np.round(mean_Rsq, 3)))
    print('The mean chi2, test is: ' + str(np.round(mean_chi2, 4)))
    
    fig1 = plt.figure(figsize=(6,6))
    fig1.subplots_adjust(hspace=.25)
    fig1.subplots_adjust(wspace=0.3)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    labels_ = list(range(1, 1 + len(R_sq_list_test)))
    y0 = max_R_sq_list
    y = R_sq_list_test
    
    x = np.arange(len(labels_))  # the label locations
    width = 0.35  # the width of the bars
    
    ax1.bar(x - width/2, y0, width, label='train', color = 'gray')
    ax1.bar(x + width/2,y, width, label='test', color = 'black')

    # Add xticks on the middle of the group bars
    ax1.set_xticks(x)
    ax1.set_xlabel('training data set k', fontweight = 'bold', fontsize = fontsize_)
    ax1.set_ylabel('R2', fontweight = 'bold', fontsize = fontsize_) 
    ax1.legend()
    
    labels_ = list(range(1, 1 + len(R_sq_list_test)))
    y0 = min_cf_list
    y = cf_list_test
    
    x = np.arange(len(labels_))  # the label locations
    width = 0.35  # the width of the bars
    
    ax2.bar(x - width/2, y0, width, label='train', color = 'gray')
    ax2.bar(x + width/2,y, width, label='test', color = 'black')

    # Add xticks on the middle of the group bars
    ax2.set_xticks(x)
    ax2.set_xlabel('training data set k', fontweight = 'bold', fontsize = fontsize_)
    ax2.set_ylabel('chi2', fontweight = 'bold', fontsize = fontsize_) 
    ax2.legend()
    
    ax3.scatter(max_R_sq_list, R_sq_list_test, color = 'black')
    ax3.set_xlabel('Rsq, train')
    ax3.set_ylabel('Rsq, test')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    ax4.scatter( min_cf_list, cf_list_test, color = 'black')
    ax4.set_xlabel('chi2, train')
    ax4.set_ylabel('chi2, test')
    
    plt.savefig('CROSS VALIDATION training and test goodness of fit metrics.svg', dpi = dpi_)
    
    return consensus_parameters
    
 

def plotPemEvaluation(files, R_sq_pem_eval_pass):
    
    '''
    Purpose: Evaluate and plot PEM evaluation criterion
   
    Inputs: 
        files: a list of dataframes containing the results for comparison
        R_sq_PEM_Eval_pass: a float defining the Rsq value for the PEM evaluation criterion
   
    Outputs: 
        None
    
    Figures: 
        'PEM EVALUATION CRITERION.svg' 
            (plot of optimized Rsq values for each PEM evaluation dataset, 
             shows PEM evaluation criterion)
    '''
    
    createFolder('./ANALYSIS')
    os.chdir('./ANALYSIS')
    
    run = []
    chi2_list = []
    R_sq_list = []
    min_cf_list = []
    max_R_sq_list = []
  
    for i, file in enumerate(files):
        chi2, Rsq = analyzeSingleRun(file, i + 1)
        chi2_list = chi2_list + chi2
        R_sq_list = R_sq_list + Rsq
        run_ = [i + 1] * len(Rsq)
        run = run  + run_
        
        val, idx = min((val, idx) for (idx, val) in enumerate(chi2))
        min_cf_list.append(val)
        max_R_sq = Rsq[idx]
        max_R_sq_list.append(max_R_sq)  
        
    df_all = pd.DataFrame(columns = ['run', 'chi2', 'Rsq'])
    df_all['run'] = run
    df_all['chi2'] = chi2_list
    df_all['Rsq'] = R_sq_list

    #Plot PEM evaluation criterion
    plt.subplots(1,1, figsize=(4,3), sharex = True)
    ax = sns.boxplot(x='run', y='Rsq', data=df_all, color = 'white')
    ax = sns.swarmplot(x='run', y='Rsq', data=df_all, color="black")
    ax.set(xlabel='PEM evaluation dataset', ylabel='Rsq, opt')
    plt.savefig('PEM EVALUATION CRITERION.svg', dpi = dpi_)
    
    
    df_all=df_all[df_all.Rsq > 0.9]
    plt.subplots(1,1, figsize=(4,3), sharex = True)
    ax = sns.boxplot(x='run', y='Rsq', data=df_all, color = 'white')
    ax = sns.swarmplot(x='run', y='Rsq', data=df_all, color="black")
    ax.set(xlabel='PEM evaluation dataset', ylabel='Rsq, opt')
    #ax.set_ylim(.9, 1.0)
    plt.savefig('PEM EVALUATION CRITERION R2 >= 0.90.svg', dpi = dpi_)
    
    print('*************************')
    print('PEM EVALUATION CRITERION')
    #min_R_sq = min(max_R_sq_list)
    # if min_R_sq < R_sq_pem_eval_pass:
    #     print('FAIL')
    #     print('EXITING CODE NOW')
    #     sys.exit()
    # else:
    #     print('PASS')
    #     print('*************************')
        
def plotParamDistributions(df):
    '''Plot distribution of all parameter sets for R2 values within 10% of the highest value'''
    
    #Only keep rows with Rsq within 10% of the highest value
    chi2 = df.sort_values(by=['chi2'])
    max_R2 = df['Rsq'][0]
    cutoff_R2 = max_R2 - (max_R2 * .1)
    df = df[df['Rsq']>= cutoff_R2]
   
    #Restructure df for plotting
    df_new = pd.DataFrame(columns = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA', 'k_loc_deactivation', 'k_txn_poisoning', 'n_txn_poisoning','k_scale_deactivation', 'Rsq'])
    for label in param_labels:
        new_list = [log10(i) for i in list(df[label + '*'])]
        df_new[label] = new_list
    new_list = [log10(i) for i in list(df['Rsq'])] 
    df_new['Rsq'] = new_list
    
    #Make box plot of parameter distributions
    plt.subplots(1,1, figsize=(8,3), sharex = True)
    df_new = pd.melt(df_new, id_vars=['Rsq'], value_vars=param_labels)
    ax = sns.boxplot(x='variable', y='value', data=df_new, color = "gray")
    ax = sns.swarmplot(x='variable', y='value', data=df_new, color="black")
    ax.set(xlabel='Parameter', ylabel='log(value)')
    plt.savefig('OPTIMIZED PARAMETER DISTRIBUTIONS.svg', dpi = 600)
     
        
def plotParamBounds(cal_params, bounds):
    '''Plot min bound, max bound, and calibrated value for each parameter'''
    #Define min and bmax bounds for each parameter and log of calibrated parameters
    min_bounds = [list_[0] for list_ in bounds]
    max_bounds = [list_[1] for list_ in bounds]
    cal_params_log = [log10(i) for i in cal_params]

    #Make plot
    fig = plt.figure(figsize=(7,5))
    plt.plot(range(0, len(cal_params_log)), min_bounds, label = 'min bound', color = 'grey', marker = '*', linestyle = 'None', zorder = 3, markersize = 8)
    plt.plot(range(0, len(cal_params_log)), max_bounds, label = 'max bound', color = 'grey', marker = '*', linestyle = 'None', zorder = 2, markersize = 8)
    plt.plot(range(0, len(cal_params_log)), cal_params_log, color = 'black', marker = 'o', linestyle = 'None', zorder = 1)
    
    #Set tick labels and axes labels
    labels = param_labels
    plt.xticks(range(len(cal_params_log)), labels)
    plt.xlabel('Free parameter')
    plt.ylabel('log10(value)')
    
    #Save figure
    plt.tight_layout()
    plt.savefig('./COMPARISON OF CALIBRATED PARAMETERS AND BOUNDS.svg')

    
def plotModelingObjectives123(solutions):
    '''Objectives involving summary metrics (1, 2, 3)''' 
    
    def fitHill(y_exp, runID):
        x = list(np.linspace(0, 240, 61)) #time (min)
        
        #Set v max to the final value of the time course
        fmax = y_exp[-1]
       
        #Set v0 to the intiial value of the time course
        f0 = y_exp[0]
    
        
        #Define a function to calculate the residual between the input simulation value (sim) and the Hill fit (model)
        def residual(p, x, y_exp):
            km = p['km'].value
            #print('km: ' + str(km))
            n = p['n'].value
            #print('n: ' + str(n))
            model = (((fmax - f0) * (x ** n)) / (km ** n + x ** n)) + f0
            #print(model)
            #print('***')
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
        
        if runID == 'plot':
            figure2 = plt.figure()
            plt.plot(x, y_exp, color = 'rebeccapurple', marker='o', linestyle = 'None')
            plt.xscale('log')
            plt.xlabel('vRNA dose (nM)')
            plt.ylabel('Final fluorophore concentration (nM)')
    
            plt.plot(x, y_Hill, color = 'grey',  linestyle = 'dashed')
            plt.xscale('log')
            plt.title('Hill fit') 
            plt.legend(['Simulation', 'Hill fit'])
        
        return f0, fmax, km, n, R_sq

    #Unnpack the data from "exp_data" and "error"
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
    def fitHillAllData(data_lists, type_):
        fit_params = []
        for time_course in data_lists:
            f0, fmax, km, n, R_sq = fitHill(time_course, '')
            fit_params.append([f0, fmax, km, n, R_sq])
    
        df = pd.DataFrame()
        f0 = [list_[0] for list_ in fit_params]
        fmax = [list_[1] for list_ in fit_params]
        km = [list_[2] for list_ in fit_params]
        n = [list_[3] for list_ in fit_params]
        R_sq = [list_[4] for list_ in fit_params]
    
        df['f0'] = f0
        df['fmax'] = fmax
        df['km'] = km
        df['n'] = n
        df['R_sq'] = R_sq
    
        fig = plt.figure(figsize = (8,5))
        ax1 = plt.subplot(231)   
        ax2 = plt.subplot(232)
        ax3 = plt.subplot(233)
        ax4 = plt.subplot(234)
        ax5 = plt.subplot(235)
        
    
        color_ = 'dimgrey'
    
        bins_ = 25
        ax1.hist(f0, bins=bins_, color = color_)
        ax1.set_xlabel('f0')
    
        ax2.hist(fmax, bins=bins_, color = color_)
        ax2.set_xlabel('fmax')
      
        ax3.hist(km, bins=bins_, color = color_)
        ax3.set_xlabel('t 1/2')
        ax3.set_xlim([50, 90])
    
        ax4.hist(n, bins=bins_, color = color_)
        ax4.set_xlabel('n')
    
        ax5.hist(R_sq, bins=bins_, color = color_)
        ax5.set_xlabel('R_sq')
        ax5.set_xlim([0.994, 0.999])
    
        plt.savefig('Modeling objectives 123 ' + type_ + '.svg', dpi = 600)
 
    data_lists = list(chunks(solutions, 61))
    print('Determining Hill fits...')
    fitHillAllData(data_lists, 'sim')
    print('Modeling objectives 1 2 and 3 plotted.')

def resultsPanel(dfSim, dfExp, dfErr, labels, varyCondition):    
    fig = plt.figure(figsize = (12,3))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ax1 = plt.subplot(141)   
    ax2 = plt.subplot(142)
    ax3 = plt.subplot(143)   
    ax4 = plt.subplot(144)
   
    def grabData(labels):
        count = 0
        sim = [1,1,1]
           
        for (columnName, columnData) in dfSim.iteritems():
            label = str(columnName)
    
            time = np.linspace(0, 240, 61)
            t_course = columnData
   
            if label == str(labels[0]):
                sim[0] = t_course
            elif label == str(labels[1]):
                sim[1] = t_course
            elif label == str(labels[2]):
                sim[2] = t_course
 
                
        maxVal = 0.6599948235700113
        exp = [1,1, 1]  
        for (columnName, columnData) in dfExp.iteritems():
            label = list(columnData.iloc[0])
            
            time = np.linspace(0, 240, 61)
            if label == labels[0]:
                t_course = columnData.iloc[1:]
                exp[0] = [i/maxVal for i in t_course]
            elif label == labels[1]:
                t_course = columnData.iloc[1:]
                exp[1] = [i/maxVal for i in t_course]
            elif label == labels[2]:
                t_course = columnData.iloc[1:]
                exp[2] = [i/maxVal for i in t_course]
                
        err = [1,1,1]  
        for (columnName, columnData) in dfErr.iteritems():
            label = list(columnData.iloc[0])
            
            time = np.linspace(0, 240, 61)
            if label == labels[0] and label != [5.0, 2.5, 0.001, 0, 90]:
                t_course = columnData.iloc[1:]
                err[0] = [i/maxVal for i in t_course]
            elif label == labels[1]:
                t_course = columnData.iloc[1:]
                err[1] = [i/maxVal for i in t_course]
            elif label == labels[2]:
                t_course = columnData.iloc[1:]
                err[2] = [i/maxVal for i in t_course]
            
                
        '''if varyCondition == 'RNAse' and labels[0][3] == 0: #don't have this data
            t_course = [0] * 61  
            sim[0] = t_course
            exp[0] = t_course
            err[0] = t_course'''
            
        # if varyCondition == 'T7' and labels[0][3] == 1.0: #don't have this data
        #     t_course = [0] * 61 
        #     sim[0] = t_course
        #     exp[0] = t_course
        #     err[0] = t_course
                
        return sim, exp, err

    sim1, exp1, err1 = grabData(labels)

    if varyCondition == 'T7':
        varyIndex = 0
        vals = [1.0, 5.0, 20.0]
        colors = ['lightgreen', 'mediumseagreen', 'darkgreen']    
    elif varyCondition == 'RT':
        varyIndex = 1
        vals = [0.5, 2.5, 10.0]
        colors = ['lightsteelblue', 'royalblue', 'midnightblue']
    elif varyCondition == 'RNAse':
        varyIndex = 2
        vals = [0.001, 0.005, 0.02]
        colors = ['lightcoral', 'red', 'maroon']
        
    time = np.linspace(0, 240, 61)
    '''for i in range(0, len(sim0)):
        ax4.plot(time, sim0[i], linestyle = ':', color = colors[i])
        ax4.set_xscale('linear')'''
    
   
    
    for i in range(0, len(exp1)):
        list_exp = exp1[i]
        list_err = err1[i]
        upper_y = []
        lower_y = []
        for j, val in enumerate(list_exp):
            upper_y.append(val + list_err[j])
            lower_y.append(val - list_err[j])

        ax1.fill_between(time, lower_y, upper_y, alpha = .2, color = colors[i])
        ax1.plot(time, exp1[i],  marker = None, linestyle = 'solid', color = colors[i])
        ax1.set_xscale('linear')
        
    for i in range(0, len(sim1)):
        ax3.plot(time, sim1[i],  marker = None, linestyle = 'dashed', color = colors[i])
    
    
    #ax1.legend([str(i) for i in vals], title = varyCondition, bbox_to_anchor=(1.0, .7))
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Normalized exp output')
    #ax4.set_xlabel('Time (min)')
    #ax4.set_ylabel('Normalized sim output')
    ax1.set_title('vRNA = 1fM', fontsize = 10, fontweight = 'bold')
    ax3.set_title('vRNA = 1fM', fontsize = 10, fontweight = 'bold')
    ax4.set_title('vRNA = 10fM', fontsize = 10, fontweight = 'bold')
    
   
    for i in range(0, len(labels)):
        labels[i][3] = 10
    
    sim10, exp10, err10 = grabData(labels)
    
    for i in range(0, len(sim1)):
        if varyCondition == 'RT' and i == 0: #condition dropped due to high error
            continue
            #ax4.plot(time, sim10[i],  marker = None, linestyle = 'dashed', color = 'dimgrey')
        else:
            ax4.plot(time, sim10[i],  marker = None, linestyle = 'dashed', color = colors[i])
        
    '''for i in range(0, len(sim1)):
        ax5.plot(time, sim1[i], linestyle = ':', color = colors[i])
        ax5.set_xscale('linear')'''
        
    for i in range(0, len(exp10)):
        list_exp = exp10[i]
        list_err = err10[i]
        upper_y = []
        lower_y = []
        for j, val in enumerate(list_exp):
            upper_y.append(val + list_err[j])
            lower_y.append(val - list_err[j])

        ax2.fill_between(time, lower_y, upper_y, alpha = .2, color = colors[i])
        ax2.plot(time, exp10[i],  marker = None, linestyle = 'solid', color = colors[i])
        ax2.set_xscale('linear')
    
    ax2.set_xlabel('Time (min)')
   
    ax2.set_title('vRNA = 10fM', fontsize = 10, fontweight = 'bold')
  
    if varyCondition == 'RT':
        objective = 5
    elif varyCondition == 'RNAse':
        objective = 4
    elif varyCondition == 'T7':
        objective = 6
    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)
    ax3.set_ylim(0, None)
    ax4.set_ylim(0, None)
    plt.savefig('./Modeling objective ' + str(objective) + '.svg', dpi = 600, bbox_inches="tight")

def plotModelingObjectives456(df_sim):
    cas =  90
    labels = [[1.0, 2.5, 0.005, 1, cas], [5.0, 2.5, 0.005, 1, cas], [20.0, 2.5, 0.005, 1, cas]]
    varyCondition = 'T7'
    resultsPanel(df_sim, df_data, df_error, labels, varyCondition) 
    print('T7 done')

    labels = [[5.0, 0.5, 0.005, 1, cas], [5.0, 2.5, 0.005, 1, cas], [5.0, 10.0, 0.005, 1, cas]]
    varyCondition = 'RT'
    resultsPanel(df_sim, df_data, df_error, labels, varyCondition) 
    print('RT done')
    
    labels = [[5.0, 2.5, 0.001, 1, cas], [5.0, 2.5, 0.005,1, cas], [5.0, 2.5, 0.02, 1, cas]]
    varyCondition = 'RNAse'
    resultsPanel(df_sim, df_data, df_error, labels, varyCondition) 
    print('RNAse done')

 
    
def plotModelingObjective4(df, data_type):
    # Plot Fmax distributions (box plot) for different slices of the data
    background_only = []
    lowCas13_only = []
    highCas13_only = []

    if data_type == 'exp':
        maxVal = 0.6599948235700113
        for (columnName, columnData) in df.iteritems():
            label = list(columnData.iloc[0])

            if label[3] == 0.0:
                final_val = list(columnData.iloc[1:])[-1]/maxVal
                background_only.append(final_val)
            if label[4] == 90.0 and label[3] != 0.0:
                final_val = list(columnData.iloc[1:])[-1]/maxVal
                highCas13_only.append(final_val)
            if label[4] == 4.5 and label[3] != 0.0:
                final_val = list(columnData.iloc[1:])[-1]/maxVal
                lowCas13_only.append(final_val)
                
    elif data_type == 'sim':
        
         def Convert(string):
            li = string.strip("[]")

            li = list(li.split(", "))
            return li
        
         for (columnName, columnData) in df.iteritems():
            label = Convert(columnName)
            time = np.linspace(0, 240, 61)
          
            final_val = columnData[60]
           
            if label[4] == '90' and label[3] != '0.0':
                highCas13_only.append(final_val)
            if label[4] == '4.5' and label[3] != '0.0':
                lowCas13_only.append(final_val)

    df = pd.DataFrame()
    if data_type == 'exp':
        df['Readout at final timepoint'] = background_only + lowCas13_only + highCas13_only 
        df['label'] = ['vRNA=0.0'] * len(background_only) + ['Cas13=2.25'] * len(lowCas13_only) + ['Cas13=45.0'] * len(highCas13_only)
    
    elif data_type == 'sim':
        df['Readout at final timepoint'] = lowCas13_only + highCas13_only 
        df['label'] = ['Cas13=2.25'] * len(lowCas13_only) + ['Cas13=45.0'] * len(highCas13_only)


    fig = plt.figure(figsize = (3,3.5))
    sns.boxplot(x="label", y="Readout at final timepoint", data=df, color = 'lightgrey')
    sns.swarmplot(x="label", y="Readout at final timepoint", data=df, color = 'dimgrey')
    plt.savefig('./Modeling objective 4.svg')
    
    
    
    
    
    
    
