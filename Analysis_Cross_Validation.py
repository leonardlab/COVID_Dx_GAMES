#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:30:11 2022

@author: kate
"""
#Package imports
import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import pandas as pd
import json 
from lmfit import Parameters, minimize

#GAMES imports
from Saving_COVID_Dx import createFolder
from Solvers_COVID_Dx import calcRsq, solveSingle, calcChi2
from DefineExpData_COVID_Dx import defineExp
from Analysis_Plots import plotModelingObjectives123, plotModelingObjectives456

#Import custom style file for plotting
plt.style.use('/Users/kate/Documents/GitHub/GAMES_COVID_Dx/paper.mplstyle.py')

#Import experimental data and experimental error
df_data = pd.read_pickle('/Users/kate/Documents/GitHub/GAMES_COVID_Dx/PROCESSED DATA EXP.pkl')
df_error = pd.read_pickle('/Users/kate/Documents/GitHub/GAMES_COVID_Dx/PROCESSED DATA ERR.pkl')
    
fontsize_ = 10
dpi_ = 600

model = 'model C'

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parityPlot(sim, exp):
     ''' Purpose: plot the experimental and simulated data in the form of a parity plot

     Input: 
        sim: a list of floats containing simulated values
        exp: a list of floats containing experimental values
        
     Output: none
    
     Plots: FIT TO TRAINING DATA PARITY PLOT.svg - 
            parity plot of the experimental and simulated data'''
            
     fig = plt.figure(figsize = (3.375,3))
     plt.plot(sim, exp, marker = 'o', markersize = 1, linestyle = 'none', color = 'steelblue')
     plt.ylabel('Experimental value')
     plt.xlabel('Simulated value')
     x_ = [0, 1]
     y_ = [0, 1]
     plt.plot(x_, y_, linestyle = '-', marker = None, color = 'dimgrey')
     plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
     plt.savefig('./FIT TO TRAINING DATA.svg', dpi = 600, bbox_inches="tight")
     
def parityPlotColoredByCondition(sim, exp, x, doses_to_color_differently, vary_condition, image_filename):
     ''' Purpose: plot the experimental and simulated data in the form of a parity plot

     Input: 
        sim: a list of floats containing simulated values
        exp: a list of floats containing experimental values
        x: a list of lists containing the enzyme and input RNA doses
        
     Output: none
    
     Plots: FIT TO TRAINING DATA PARITY PLOT.svg - 
            parity plot of the experimental and simulated data'''
            
     fig = plt.figure(figsize = (3.375,3))
     plt.plot(sim, exp, marker = 'o', markersize = 1, linestyle = 'none', color = 'dimgrey')
     plt.ylabel('Experimental value')
     plt.xlabel('Simulated value')
     x_ = [0, 1]
     y_ = [0, 1]
     plt.plot(x_, y_, linestyle = '-', marker = None, color = 'dimgrey')
     plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
     plt.savefig('./FIT TO TRAINING DATA.svg', dpi = 600, bbox_inches="tight")   
     
     if vary_condition == 'T7':
        colors = ['lightgreen', 'mediumseagreen', 'darkgreen']    
     elif vary_condition == 'RT':
        colors = ['lightsteelblue', 'royalblue', 'midnightblue']
     elif vary_condition == 'RNAse':
        colors = ['lightcoral', 'red', 'maroon']
     
     chunked_sim = list(chunks(sim, 61))
     chunked_exp = list(chunks(exp, 61))
     for i, doses in enumerate(x):
         if doses == doses_to_color_differently[0]:
             plt.plot(chunked_sim[i], chunked_exp[i], marker = 'None', linestyle = 'solid', color = colors[0])
         elif doses == doses_to_color_differently[1]:
             plt.plot(chunked_sim[i], chunked_exp[i], marker = 'None',  linestyle = 'solid', color = colors[1])
         elif doses == doses_to_color_differently[2]:
             plt.plot(chunked_sim[i], chunked_exp[i], marker = 'None',  linestyle = 'solid', color = colors[2])
             
     plt.savefig('./FIT TO TRAINING DATA ' + image_filename + '.svg', dpi = 600, bbox_inches="tight")
             
     
     

def solveAll(p, exp_data, x):
    
    '''
    Purpose: Solve ODEs for the entire dataset using parameters defined in p 
        p = ['k_cas13', 'k_degv', 'k_txn', 'k_FSS', 'k_RHA', 'k_loc_deactivation', 'k_scale_deactivation']
    
    Inputs: 
        p: a list of floats containing the parameter set (order of parameter defined in 
            Settings.py and in the ODE defintion function in Solvers.py)
        exp_data: a list of floats containing the experimental data (length = # data points)
        x:  a list of lists containing the component dose conditions
            
    Outputs: 
        x: list of lists containing the conditions
        solutions_norm: list of floats containing the normalized simulation values
        mse: float defining the chi2/number of data points
        dfSimResults: df containing the normalized simulation values'''

   
    dfSimResults = pd.DataFrame()
    solutions = []
   
    for doses in x: #For each condition (dose combination) in x 
  
        t, solutions_all, reporter_timecourse = solveSingle(doses, p, model)
     
        if len(reporter_timecourse) == len(t):
            reporter_timecourse = reporter_timecourse
            
        else:
            reporter_timecourse = [0] * len(t)
        
        for i in reporter_timecourse:
            solutions.append(float(i))
            
        dfSimResults[str(doses)] = reporter_timecourse
        
   
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
            chi2 = 10000000000
            return x, solutions_norm, chi2, dfSimResults

    chi2 = calcChi2(exp_data, solutions_norm)
    mse = chi2/len(solutions_norm)
    
    #max val filter
    if max(solutions) < 2000:
        print('Failed filter 1')
        print('The maximum value of x_f is: ' + str(max(solutions)))
        mse = 1
        
    #low iCas13 filter
    else:
        doses = [5.0, 0.5, 0.005, 1, 4.5]
        t, solutions_all, reporter_timecourse = solveSingle(doses, p, model)
        final_timepoint_iCas13 = reporter_timecourse[-1] #no norm
        max_high_iCas13 = max(solutions) #no norm
        ratio_2 = final_timepoint_iCas13/max_high_iCas13
        if ratio_2 > 0.10:
            print('Failed filter 2')
            print('ratio2: The max lowiCas13/max high iCas13 ratio is: ' + str(ratio_2))
            mse = 2


    return x, solutions_norm, mse, dfSimResults

def resultsPanel_ensemble(dfSim, dfExp, dfErr, labels, varyCondition, df_ensemble_error):   
    '''Purpose: Plot selected readout time courses for the ensemble model (after cross-validation)
    
       Input: 
            dfSim: df containing the simulated data
            dfExp: df containing the experimental data
            dfErr: df containing the measurement error associated with the experimental data
            labels: list of lists containing the condition labels to be plotted
            varyCondition: string contraining the label of the condition that is sweeped over in the plot
   
       Output: none
        
       Plots: MODELING OBJECTIVE ' + str(objective) + '.svg' - 
                Plot of readout dynamics associated with the given modeling objective 
    '''
    
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
                
        sim_err = [1,1,1]    
        for (columnName, columnData) in df_ensemble_error.iteritems():
            label = str(columnName)
    
            time = np.linspace(0, 240, 61)
            t_course_err = columnData
   
            if label == str(labels[0]):
                sim_err[0] = t_course_err 
            elif label == str(labels[1]):
                sim_err[1] = t_course_err 
            elif label == str(labels[2]):
                sim_err[2] = t_course_err 
 
                
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

                
        return sim, exp, err, sim_err

    sim1, exp1, err1, sim_err1 = grabData(labels)

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
        list_sim = sim1[i]
        list_sim_err = sim_err1[i]
        upper_y = []
        lower_y = []
        for j, val in enumerate(list_sim):
            upper_y.append(val + list_sim_err[j])
            lower_y.append(val - list_sim_err[j])

        ax3.fill_between(time, lower_y, upper_y, alpha = .2, color = colors[i])
        ax3.plot(time, sim1[i],  marker = None, linestyle = 'dashed', color = colors[i])
        ax3.set_xscale('linear')
    
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Normalized exp output')
    ax1.set_title('vRNA = 1fM', fontsize = 10, fontweight = 'bold')
    ax3.set_title('vRNA = 1fM', fontsize = 10, fontweight = 'bold')
    ax4.set_title('vRNA = 10fM', fontsize = 10, fontweight = 'bold')

    for i in range(0, len(labels)):
        labels[i][3] = 10
    
    sim10, exp10, err10, sim_err10 = grabData(labels)
    
    for i in range(0, len(sim10)):
        if varyCondition == 'RT' and i == 0: #condition dropped due to high error
            continue
        else:
            list_sim = sim10[i]
            list_sim_err = sim_err10[i]
            upper_y = []
            lower_y = []
            for j, val in enumerate(list_sim):
                upper_y.append(val + list_sim_err[j])
                lower_y.append(val - list_sim_err[j])
    
            ax4.fill_between(time, lower_y, upper_y, alpha = .2, color = colors[i])
            ax4.plot(time, sim10[i],  marker = None, linestyle = 'dotted', color = colors[i])
            ax4.set_xscale('linear')
    
    for i in range(0, len(sim1)):
        if varyCondition == 'RT' and i == 0: #condition dropped due to high error
            continue
        else:
            ax4.plot(time, sim10[i],  marker = None, linestyle = 'dashed', color = colors[i])
        
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
        y_max_1 = 0.35
        y_max_10 = 1.0
    elif varyCondition == 'RNAse':
        objective = 4
        y_max_1 = 0.25
        y_max_10 = 0.85
    elif varyCondition == 'T7':
        objective = 6
        y_max_1 = 0.65
        y_max_10 = 1.0
    ax1.set_ylim(0, y_max_1)
    ax2.set_ylim(0, y_max_10)
    ax3.set_ylim(0, y_max_1)
    ax4.set_ylim(0, y_max_10)
    plt.savefig('./MODELING OBJECTIVE ' + str(objective) + '.svg', dpi = 600, bbox_inches="tight")

def plotModelingObjectives456_ensemble(df_sim, df_ensemble_error):
    '''Purpose: Plot selected readout time courses for objectives 4, 5, and 6 for the ensemble model (after cross-validation)
    
       Input: 
            dfSim: df containing the simulated data
   
       Output: none
        
       Plots: None
    '''
     
    cas = 90
    labels = [[1.0, 2.5, 0.005, 1, cas], [5.0, 2.5, 0.005, 1, cas], [20.0, 2.5, 0.005, 1, cas]]
    varyCondition = 'T7'
    resultsPanel_ensemble(df_sim, df_data, df_error, labels, varyCondition, df_ensemble_error) 
    print('T7 done')

    labels = [[5.0, 0.5, 0.005, 1, cas], [5.0, 2.5, 0.005, 1, cas], [5.0, 10.0, 0.005, 1, cas]]
    varyCondition = 'RT'
    resultsPanel_ensemble(df_sim, df_data, df_error, labels, varyCondition, df_ensemble_error) 
    print('RT done')
    
    labels = [[5.0, 2.5, 0.001, 1, cas], [5.0, 2.5, 0.005,1, cas], [5.0, 2.5, 0.02, 1, cas]]
    varyCondition = 'RNAse'
    resultsPanel_ensemble(df_sim, df_data, df_error, labels, varyCondition, df_ensemble_error) 
    print('RNAse done')


def fitHill(y_exp):
   
    x = list(np.linspace(0, 240, 61)) #time (min)
    
    #Set v max to the final value of the time course
    fmax = y_exp[-1]
   
    #Set v0 to the initial value of the time course
    f0 = y_exp[0]

    #Define a function to calculate the residual between the data sets
    def residual(p, x, y_exp):
        km = p['km'].value
        n = p['n'].value
        model = (((fmax - f0) * (x ** n)) / (km ** n + x ** n)) + f0
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
    
    return f0, fmax, km, n, R_sq

def plot_tradeoff(x, fmax, t_half, k_CV, fmax_sd_list, t_half_sd_list):
    fig, axes = plt.subplots(ncols=3,nrows=1, sharex=True, sharey=True, figsize = (9,3))
    
    for color_by, ax in zip(['T7', 'RT', 'RNAse'], axes.flat):
        
        if color_by == 'T7':
            vals = [1.0, 5.0, 20.0]
            colors = ['lightgreen', 'mediumseagreen', 'darkgreen']    
            varyIndex = 0
        elif color_by == 'RT':
            vals = [0.5, 2.5, 10.0]
            colors = ['lightsteelblue', 'royalblue', 'midnightblue']
            varyIndex = 1
        elif color_by == 'RNAse':
            vals = [0.001, 0.005, 0.02]
            colors = ['lightcoral', 'red', 'maroon']
            varyIndex = 2
            
        for i, enzyme_condition in enumerate(x):
            if enzyme_condition[varyIndex] == vals[0]:
                color_ = colors[0]
            elif enzyme_condition[varyIndex] == vals[1]:
                color_ = colors[1]
            elif enzyme_condition[varyIndex] == vals[2]:
                color_ = colors[2]     

            if enzyme_condition[3] == 1 and enzyme_condition[4] == 90:
                ax.errorbar(t_half[i], fmax[i],  marker = 'o', xerr = t_half_sd_list[i], yerr = fmax_sd_list[i], linestyle = 'None', color = color_)

            if enzyme_condition == [5.0, 2.5, 0.005, 1, 90]:
                ax.errorbar(t_half[i], fmax[i],  marker = 'o', xerr = t_half_sd_list[i], yerr = fmax_sd_list[i], linestyle = 'None', color = 'dimgrey')
                
            if enzyme_condition == [5.0, 10.0, 0.02, 1, 90]:
                ax.errorbar(t_half[i], fmax[i],  marker = 'o', xerr = t_half_sd_list[i], yerr = fmax_sd_list[i], linestyle = 'None', color = 'black')
                
                
            if color_by == 'RT':
                ax.set_xlabel('t 1/2 (min.)')
            if color_by == 'T7':
                ax.set_ylabel('f max (normalized)')
            ax.set_ylim([0, 1])
            #ax.set_xlim([60, 100])
        
    plt.savefig('./experimental performance metric tradeoff ensemble ' + str(k_CV) + '.svg')

def calcCFTestAllk(cal_parameters):
    R_sq_list_test = []
    cf_list_test = []
    for k_CV in range(1, len(filepaths)+1):
        x, exp_data, error, timecourses, timecourses_err = defineExp('cross-validation test','', k_CV, '')
        p = cal_parameters[k_CV - 1]
        x, solutions_norm, mse, dfSimResults = solveAll(p, exp_data, x)
        R_sq = calcRsq(solutions_norm, exp_data)  
        R_sq_list_test.append(R_sq)
        cf_list_test.append(mse)
        
    return R_sq_list_test, cf_list_test
    
def analyzeCV(filepaths, folder_name):
    '''
    Purpose: Analyze CV simulations and make plots to interpret results
        
    
    Inputs: 
       filepaths: a list of strings defining the filepaths to be analyzed
       folder_name: a string defining the name of the folder for the results to be saved in
       
            
    Outputs: 
        x: list of lists containing the conditions
        solutions_norm: list of floats containing the normalized simulation values
        mse: float defining the chi2/t# of data points
        dfSimResults: df containing the normalized simulation values
        
        
    Plots:
        'ENSEMBLE MODEL RESULTS.svg' - a bar plot with k_cv on the x-axis and the overall, ensemble R2 on the y-axis
        'ENSEMBLE PARITY PLOT.svg' - a parity plot showing agreement between the experimental data and the ensemble model
        'CROSS VALIDATION training and test goodness of fit metrics.svg'
        'MODELING OBJECTIVE ' + str(objective) + '.svg' - 
                Plot of readout dynamics associated with the given modeling objective for the ensemble model
        
        
        '''
        
  
    
    #Import dfs
    df_list = []
    for filepath in filepaths:
        df = pd.read_excel(filepath, index_col=0)  
        df_list.append(df)
        
    #Grab data from dfs
    k_CV = []
    chi2_list = []
    R_sq_list = []
    min_cf_list = []
    max_R_sq_list = []
    cal_parameters = []
  
    for i, df in enumerate(df_list):
        chi2 = list(df['chi2'])
        Rsq = list(df['Rsq'])
        chi2_list = chi2_list + chi2
        R_sq_list = R_sq_list + Rsq
        run_ = [i + 1] * len(Rsq)
        k_CV = k_CV  + run_
        
        val, idx = min((val, idx) for (idx, val) in enumerate(chi2))
        min_cf_list.append(val)
        max_R_sq = Rsq[idx]
        max_R_sq_list.append(max_R_sq)  
        
        param_labels = ['k_cas13*', 'k_degv*', 'k_txn*', 'k_FSS*', 'k_RHA*', 'k_loc_deactivation*', 'k_scale_deactivation*'] 
        #param_labels = ['k_cas13*', 'k_degv*', 'k_txn*', 'k_FSS*', 'k_RHA*', 'mean_aCas13*','k_Mg*', 'n_Mg*','sd_aCas13*']
        p = []
        for label in param_labels:
            p.append(df[label].iloc[idx])
        cal_parameters.append(p)
            
        
    #Create new df to hold combined results from all runs
    df_all = pd.DataFrame(columns = ['k_CV', 'chi2', 'Rsq'])
    df_all['k_CV'] = k_CV
    df_all['chi2'] = chi2_list
    df_all['Rsq'] = R_sq_list
    
    return df_all, cal_parameters, min_cf_list, max_R_sq_list
    
def calcMean(data_points, weights):
    numerator = 0
    denominator = 0
    for i in range(0, len(data_points)):
        numerator += weights[i] * data_points[i]
        denominator += weights[i]
    mean_data_point = numerator/denominator
    return mean_data_point
    
def calcAllSolutionsForEachK(cal_parameters, data_):
    solutions_list = []
    #fmax_list_all_k = [] #first index is k, second index is condition
    #t_half_list_all_k = []
    for k_CV in range(1, len(filepaths)+1):
        #data_ = 'slice drop high error'
        
        x, exp_data, error, timecourses, timecourses_err = defineExp(data_,'', '','')       
        index = k_CV - 1
        p = cal_parameters[index]
        x, solutions_norm, mse, dfSimResults = solveAll(p, exp_data, x)
        solutions_list.append(solutions_norm)
        #fmax_list, t_half_list = plotPerformanceMetricTradeoffs(solutions_norm, k_CV)
        #fmax_list_all_k.append(fmax_list)
        #t_half_list_all_k.append(t_half_list)
        print(str(k_CV) + ' done.' )
        
    return exp_data, solutions_list, x
    
def calcEnsembleSolution(num_k, solutions_list, exp_data, data_, x, cf_list_test):
    weights = [1/i for i in cf_list_test]
    ensemble_solutions = []
    ensemble_solutions_error = []
    for data_point in range(0, len(solutions_list[0])):
        data_points = []
        for k_CV_index in range(0, num_k):
            data_points.append(solutions_list[k_CV_index][data_point])
        if metric_type == 'mean':
            mean_data_point = np.mean(data_points)
            ensemble_solutions.append(mean_data_point)
        elif metric_type == 'weighted mean':
            mean_data_point = calcMean(data_points, weights)
            ensemble_solutions.append(mean_data_point)
        elif metric_type == 'median':
            median_data_point = np.median(data_points)
            ensemble_solutions.append(median_data_point)    
        sd_data_point = np.std(data_points)
        ensemble_solutions_error.append(sd_data_point)
    R_sq_ensemble = calcRsq(ensemble_solutions, exp_data)
   
    if data_ == 'all echo without low iCas13 or 0 vRNA and drop high error':
         parityPlot(ensemble_solutions, exp_data)
         plt.savefig('SF14e ENSEMBLE PARITY PLOT ' + str(num_k) + '.svg', dpi = dpi_)
    
    return R_sq_ensemble, ensemble_solutions, ensemble_solutions_error

def plotModelingObjectivesEnsembleModel(ensemble_solutions, ensemble_solutions_error, x):
    plotModelingObjectives123(ensemble_solutions)
    chunked_data = list(chunks(ensemble_solutions, 61))
    chunked_data_error = list(chunks(ensemble_solutions_error, 61))
    df_ensemble = pd.DataFrame()
    df_ensemble_error = pd.DataFrame()
    for i, time_course in enumerate(chunked_data):
        col_name = str(x[i])
        df_ensemble[col_name] = time_course
        df_ensemble_error[col_name] = chunked_data_error[i]
    plotModelingObjectives456_ensemble(df_ensemble, df_ensemble_error)
        

def plotOverallEnsembleCF(solutions_list, exp_data, x, cf_list_test):
    ensemble_metrics = []
    for num_k in range(1, len(solutions_list)+1):
        R_sq_ensemble, ensemble_solutions, ensemble_solutions_error = calcEnsembleSolution(num_k, solutions_list, exp_data, data_, x, cf_list_test)
        ensemble_metrics.append(np.round(R_sq_ensemble, 3))
        
    fig = plt.figure(figsize = (3,3))
    labels_ = list(range(1, 1 + len(ensemble_metrics)))
    y = ensemble_metrics
    
    x_ = np.arange(len(labels_))  # the label locations
    width = 0.35  # the width of the bars
    
    plt.bar(x_, y, width, color = 'steelblue')

    # Add xticks on the middle of the group bars
    plt.xticks(x_)
    plt.xlabel('k_CV', fontweight = 'bold', fontsize = fontsize_)
    plt.ylabel('R2, overall', fontweight = 'bold', fontsize = fontsize_) 
    plt.ylim([0, 1])
    plt.savefig('SF14d ENSEMBLE MODEL RESULTS.svg', dpi = 600)
      
    return ensemble_metrics

def plotParityPlotsSF16():
    vary_condition = 'RNAse'
    doses_to_color_differently = [[1.0, 0.5, 0.001, 1.0, 90.0], [1.0, 0.5, 0.005, 1.0, 90.0], [1.0, 0.5, 0.02, 1.0, 90.0]]
    image_filename = 'SF16a low T7, low RT, low input RNA'
    parityPlotColoredByCondition(ensemble_solutions, exp_data, x, doses_to_color_differently, vary_condition, image_filename)
    
    doses_to_color_differently = [[1.0, 0.5, 0.001, 10.0, 90.0], [1.0, 0.5, 0.005, 10.0, 90.0], [1.0, 0.5, 0.02, 10.0, 90.0]]
    image_filename = 'SF16a low T7, low RT, high input RNA'
    parityPlotColoredByCondition(ensemble_solutions, exp_data, x, doses_to_color_differently, vary_condition, image_filename)

    doses_to_color_differently = [[1.0, 2.5, 0.001, 1.0, 90.0], [1.0, 2.5, 0.005, 1.0, 90.0], [1.0, 2.5, 0.02, 1.0, 90.0]]
    image_filename = 'SF16b low T7, mid RT, low input RNA'
    parityPlotColoredByCondition(ensemble_solutions, exp_data, x, doses_to_color_differently, vary_condition, image_filename)

    doses_to_color_differently = [[1.0, 2.5, 0.001, 10.0, 90.0], [1.0, 2.5, 0.005, 10.0, 90.0], [1.0, 2.5, 0.02, 10.0, 90.0]]
    image_filename = 'SF16b low T7, mid RT, high input RNA'
    parityPlotColoredByCondition(ensemble_solutions, exp_data, x, doses_to_color_differently, vary_condition, image_filename)

    vary_condition = 'RT'
    doses_to_color_differently = [[1.0, 0.5, 0.005, 1.0, 90.0], [1.0, 2.5, 0.005, 1.0, 90.0], [1.0, 10.0, 0.02, 1.0, 90.0]]
    image_filename = 'SF16c low T7, mid RNAse, low input RNA'
    parityPlotColoredByCondition(ensemble_solutions, exp_data, x, doses_to_color_differently, vary_condition, image_filename)
    
    doses_to_color_differently = [[1.0, 0.5, 0.005, 10.0, 90.0], [1.0, 2.5, 0.005, 10.0, 90.0], [1.0, 10.0, 0.02, 10.0, 90.0]]
    image_filename = 'SF16c low T7, mid RNAse, high input RNA'
    parityPlotColoredByCondition(ensemble_solutions, exp_data, x, doses_to_color_differently, vary_condition, image_filename)
  
def plotTrainAndTestForAllK(R_sq_list_train, cf_list_train, R_sq_list_test, cf_list_test):

    # #mean R2 and chi2 for training data
    mean_Rsq = np.mean(R_sq_list_train)
    mean_chi2 = np.mean(cf_list_train)
    print('The mean R2, train is: ' + str(np.round(mean_Rsq, 3)))
    print('The mean chi2, train is: ' + str(np.round(mean_chi2, 4)))
    
    #mean R2 and chi2 for test data
    mean_Rsq = np.mean(R_sq_list_test)
    mean_chi2 = np.mean(cf_list_test)
    print('The mean R2, test is: ' + str(np.round(mean_Rsq, 3)))
    print('The mean chi2, test is: ' + str(np.round(mean_chi2, 4)))
    
    #plot test and train metrics
    fig1 = plt.figure(figsize=(7, 6))
    fig1.subplots_adjust(hspace=.25)
    fig1.subplots_adjust(wspace=0.3)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    labels_ = list(range(1, 1 + len(R_sq_list_test)))
    y0 = R_sq_list_train
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
    y0 = cf_list_train
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
    
    ax3.scatter(R_sq_list_train, R_sq_list_test, color = 'black')
    ax3.set_xlabel('Rsq, train')
    ax3.set_ylabel('Rsq, test')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    ax4.scatter(cf_list_train, cf_list_test, color = 'black')
    ax4.set_xlabel('chi2, train')
    ax4.set_ylabel('chi2, test')
    
    plt.savefig('SF14c CROSS VALIDATION training and test goodness of fit metrics.svg', dpi = dpi_)
  
  
path1 = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/Results/2022-03-03 new data partitioning k = 1/PARAMETER ESTIMATION cross-validation train/OPT RESULTS.xlsx'
path2 = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/Results/2022-03-03 new data partitioning k = 2/PARAMETER ESTIMATION cross-validation train/OPT RESULTS.xlsx'
path3 = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/Results/2022-03-04 new data partitioning k = 3/PARAMETER ESTIMATION cross-validation train/OPT RESULTS.xlsx'
path4 = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/Results/2022-03-04 new data partitioning k = 4/PARAMETER ESTIMATION cross-validation train/OPT RESULTS.xlsx'
path5 = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/Results/2022-03-04 new data partitioning k = 5/PARAMETER ESTIMATION cross-validation train/OPT RESULTS.xlsx'
path6 = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/Results/2022-03-04 new data partitioning k = 6/PARAMETER ESTIMATION cross-validation train/OPT RESULTS.xlsx'
path7 = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/Results/2022-03-04 new data partitioning k = 7/PARAMETER ESTIMATION cross-validation train/OPT RESULTS.xlsx'
path8 = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/Results/2022-03-04 new data partitioning k = 8/PARAMETER ESTIMATION cross-validation train/OPT RESULTS.xlsx'
path9 = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/Results/2022-03-04 new data partitioning k = 9/PARAMETER ESTIMATION cross-validation train/OPT RESULTS.xlsx'
path10 = '/Users/kate/Documents/GitHub/GAMES_COVID_Dx/Results/2022-03-04 new data partitioning k = 10/PARAMETER ESTIMATION cross-validation train/OPT RESULTS.xlsx'

filepaths = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10]
metric_type = 'weighted mean'
folder_name = 'plot parity plots colored by outlier conditions'
createFolder('./Results/' + folder_name)
os.chdir('./Results/' + folder_name)

# data = 'all echo without low iCas13 or 0 vRNA and drop high error'
# x, exp_data, error, timecourses, timecourses_err = defineExp(data,'', '', '')
# plotModelingObjectives123(exp_data)

#Import all data and restructure into a single dataframe
df_all, cal_parameters, cf_list_train, R_sq_list_train = analyzeCV(filepaths, folder_name)

#Calculate R2 and chi2 for each k_CV
R_sq_list_test, cf_list_test = calcCFTestAllk(cal_parameters)

#Plot bar plot of training and test metrics 
#lotTrainAndTestForAllK(R_sq_list_train, cf_list_train, R_sq_list_test, cf_list_test)

#Calculate ensemble solutions for each k_cv for all data
data_ = 'all echo without low iCas13 or 0 vRNA and drop high error'
exp_data, all_solutions, x  = calcAllSolutionsForEachK(cal_parameters, data_)
R_sq_ensemble, ensemble_solutions, ensemble_solutions_error = calcEnsembleSolution(len(filepaths), all_solutions, exp_data, data_, x, cf_list_test)
plotModelingObjectivesEnsembleModel(ensemble_solutions, ensemble_solutions_error, x)

#Calculate and plot ensemble CF metrics
ensemble_metrics = plotOverallEnsembleCF(all_solutions, exp_data, x, cf_list_test)

#Calculate ensemble solutions for slice only and plot modeling objectives for ensemble model
data_ = 'slice drop high error'
exp_data, all_solutions, x = calcAllSolutionsForEachK(cal_parameters, data_)
num_k_for_ensemble_model = 5
R_sq_ensemble, ensemble_solutions, ensemble_solutions_error = calcEnsembleSolution(num_k_for_ensemble_model, all_solutions, exp_data, data_, x, cf_list_test)
plotModelingObjectivesEnsembleModel(ensemble_solutions, ensemble_solutions_error, x)

#plot parity plots colored by outlier conditions
R_sq_ensemble, ensemble_solutions, ensemble_solutions_error = calcEnsembleSolution(5, all_solutions, exp_data, data_, x, cf_list_test)
plotParityPlotsSF16()




# =============================================================================
# Outdated
# =============================================================================

  
    #Plot performance metric tradeoff for ensemble model
    # R_sq_ensemble, ensemble_solutions = calcEnsembleSolution(10)
    # plotPerformanceMetricTradeoffs(ensemble_solutions, 'ensemble 10')
    
    # R_sq_ensemble, ensemble_solutions = calcEnsembleSolution(13)
    # plotPerformanceMetricTradeoffs(ensemble_solutions, 'ensemble 13')

    # Plot ensemble model results for a given condition
    # R_sq_ensemble, ensemble_solutions = calcEnsembleSolution(10)
    # chunked_data = list(chunks(ensemble_solutions, 61))
    # chunked_error = list(chunks(error, 61))
    # chunked_exp = list(chunks(exp_data, 61))
    
    # fig = plt.figure(figsize = (3,3))
    # for i, enzyme_condition in enumerate(x):
    #     if enzyme_condition == [5.0, 10.0, 0.02, 1, 90]:
    #         t = list(np.linspace(0, 240, 61)) 
    #         plt.errorbar(t, chunked_exp[i],  yerr = chunked_error[i], linestyle = 'solid', marker = 'None',  color = 'silver')
    #         plt.plot(t, chunked_data[i], linestyle = 'dotted', marker = 'None', color = 'black')
    #         plt.savefig('./optimal condition ensemble model')


   

# if metric_type == 'mean':
    #     fmax_mean_list = [np.mean(sub_list) for sub_list in zip(*fmax_list_all_k)]
    #     t_half_mean_list = [np.mean(sub_list) for sub_list in zip(*t_half_list_all_k)]
    # elif metric_type == 'weighted mean':
    #     fmax_mean_list = [calcMean(sub_list, weights) for sub_list in zip(*fmax_list_all_k)]
    #     t_half_mean_list = [calcMean(sub_list, weights) for sub_list in zip(*t_half_list_all_k)]
  
    # fmax_sd_list = [np.std(sub_list) for sub_list in zip(*fmax_list_all_k)]
    # t_half_sd_list = [np.std(sub_list) for sub_list in zip(*t_half_list_all_k)]
    
    # plot_tradeoff(x, fmax_mean_list,  t_half_mean_list, k_CV, fmax_sd_list, t_half_sd_list)  

    
    # with open("x_doses.json", 'w') as f:
    #     json.dump(x, f, indent=2) 
        
    # with open("fmax_mean.json", 'w') as f:
    #     json.dump(fmax_mean_list, f, indent=2) 
        
    # with open("t_half_mean.json", 'w') as f:
    #     json.dump(t_half_mean_list, f, indent=2) 

    # with open("fmax_list_exp.json", 'w') as f:
    #     json.dump(fmax_list_exp, f, indent=2) 
        
    # with open("t_half_list_exp.json", 'w') as f:
    #     json.dump(t_half_list_exp, f, indent=2) 

            
        
    
      
    # def calcRankOrder(fmax_list_ensemble, t_half_mean_list, fmax_list_exp, t_half_list_exp, x):
    #     #normalize to max
    #     fmax_list_norm_ensemble = [i/max(fmax_list_ensemble) for i in fmax_list_ensemble]
    #     t_half_list_norm_ensemble= [i/max(t_half_list_ensemble) for i in t_half_list_ensemble]
    #     fmax_list_norm_exp = [i/max(fmax_list_exp) for i in fmax_list_exp]
    #     t_half_list_norm_exp = [i/max(t_half_list_exp) for i in t_half_list_exp]
       
    #     summary_metrics_ensemble = []
    #     summary_metrics_exp  = []
    #     for i in range(0, len(t_half_list_norm_ensemble)):
    #         #calculate ensemble metric
    #         metric_ensemble =  fmax_list_norm_ensemble[i] + (1 - t_half_list_norm_ensemble[i])
    #         summary_metrics_ensemble.append(metric_ensemble)
            
    #         #calculate exp metric
    #         metric_exp =  fmax_list_norm_exp [i] + (1 - t_half_list_norm_exp [i])
    #         summary_metrics_exp .append(metric_exp)
            
    #     fig, axes = plt.subplots(ncols=3,nrows=1, sharex=True, sharey=True, figsize = (9,3))
    
    #     for color_by, ax in zip(['T7', 'RT', 'RNAse'], axes.flat):
            
    #         if color_by == 'T7':
    #             vals = [1.0, 5.0, 20.0]
    #             colors = ['lightgreen', 'mediumseagreen', 'darkgreen']    
    #             varyIndex = 0
    #         elif color_by == 'RT':
    #             vals = [0.5, 2.5, 10.0]
    #             colors = ['lightsteelblue', 'royalblue', 'midnightblue']
    #             varyIndex = 1
    #         elif color_by == 'RNAse':
    #             vals = [0.001, 0.005, 0.02]
    #             colors = ['lightcoral', 'red', 'maroon']
    #             varyIndex = 2
                
    #         for i, enzyme_condition in enumerate(x):
    #             if enzyme_condition[varyIndex] == vals[0]:
    #                 color_ = colors[0]
    #             elif enzyme_condition[varyIndex] == vals[1]:
    #                 color_ = colors[1]
    #             elif enzyme_condition[varyIndex] == vals[2]:
    #                 color_ = colors[2]     
    
    #             if enzyme_condition[3] == 1 and enzyme_condition[4] == 90:
    #                 ax.errorbar(summary_metrics_exp[i], summary_metrics_ensemble[i],  marker = 'o', xerr = t_half_sd_list[i], yerr = fmax_sd_list[i], linestyle = 'None', color = color_)
    
    #             if color_by == 'RT':
    #                 ax.set_xlabel('Experiment')
    #             if color_by == 'T7':
    #                 ax.set_ylabel('Ensemble')
    #             #ax.set_ylim([0, 1])
    #             #ax.set_xlim([60, 100])
            
    #     plt.savefig('./summary metric ensemble vs exp ' + str(k_CV) + '.svg')
            
                
            
    # calcRankOrder(fmax_mean_list, t_half_mean_list, fmax_list_exp, t_half_list_exp, x)
             

 # def plotPerformanceMetricTradeoffs(solutions_norm, k_CV):
    #     chunked_data = list(chunks(solutions_norm, 61))
        
    #     fmax_list = []
    #     t_half_list = []
    #     for data_list in chunked_data:
    #         f0, fmax, km, n, R_sq = fitHill(data_list)
    #         fmax_list.append(fmax)
    #         t_half_list.append(km)
        
    #     fmax_sd_list = [0] * len(fmax_list)
    #     t_half_sd_list = [0] * len(fmax_list)
    #     plot_tradeoff(x, fmax_list, t_half_list, k_CV, fmax_sd_list, t_half_sd_list)
        
    #     return fmax_list, t_half_list
    
    
    # =============================================================================
    # perf metrics for exp data with error
    # =============================================================================
    
    # data_ = 'all echo without low iCas13 or 0 vRNA and drop high error'
    # x, exp_data, error, timecourses, timecourses_err = defineExp(data_,'', '','')    
    # for i in range(0, len(error)):
    #     if error[i] == 0.0:
    #         error[i] = error[i-1]
            
    # k_CV = 'na'
    
    # chunked_data = list(chunks(exp_data, 61))
    # fmax_list_exp = []
    # t_half_list_exp  = []
    # for data_list in chunked_data:
    #     f0, fmax, km, n, R_sq = fitHill(data_list)
    #     fmax_list_exp.append(fmax)
    #     t_half_list_exp.append(km)
        
    # #high limit
    # high_error_solutions = []
    # for i, exp_val in enumerate(exp_data):
    #     new_item = exp_val + error[i]
    #     high_error_solutions.append(new_item)
        
    # chunked_data = list(chunks(high_error_solutions, 61))
    # fmax_list_high_error = []
    # t_half_list_high_error = []
    # for data_list in chunked_data:
    #     f0, fmax, km, n, R_sq = fitHill(data_list)
    #     fmax_list_high_error.append(fmax)
    #     t_half_list_high_error.append(km)
        
    #low limit   
    # low_error_solutions = []
    # for i, exp_val in enumerate(exp_data):
    #     new_item = exp_val - error[i]
    #     low_error_solutions.append(new_item)
        
    # chunked_data = list(chunks(low_error_solutions, 61))
    # fmax_list_low_error = []
    # t_half_list_low_error = []
    # for data_list in chunked_data:
    #     f0, fmax, km, n, R_sq = fitHill(data_list)
    #     fmax_list_low_error.append(fmax)
    #     t_half_list_low_error.append(km)
        
    # fmax_sd_list = []
    # for i, val in enumerate(fmax_list):
    #     fmax_sd_list.append(abs(val - fmax_list_high_error[i]))
        
    # t_half_sd_list = []
    # for i, val in enumerate(t_half_list):
    #     t_half_sd_list.append(abs(val -  t_half_list_high_error[i]))

 
    # plot_tradeoff(x, fmax_list, t_half_list, k_CV, fmax_sd_list, t_half_sd_list)
    
    
   
    # =============================================================================
    #     perf metrics for sims fit on all data
    # =============================================================================
   
    # data_ = 'all echo without low iCas13 or 0 vRNA and drop high error'
    # x, exp_data, error, timecourses, timecourses_err = defineExp(data_,'', '','')    
    # k_CV = 'na'
    # p = [0.000467243,	23034.47396,	1174.784961,	0.006634409,	1.083733895,	25.37653242,	16.83162789]
    # x, solutions_norm, mse, dfSimResults = solveAll(p, exp_data, x)
    # plotPerformanceMetricTradeoffs(solutions_norm, k_CV)
                      
    
    # =============================================================================
    #  start plot prediction - 200aM
    # =============================================================================
    # solutions_list = []
    # fmax_list_all_k = [] #first index is k, second index is condition
    # t_half_list_all_k = []
    # for k_CV in range(1, len(filepaths)+1):
    #     #data_ = 'slice drop high error'
    #     data_ = 'all echo without low iCas13 or 0 vRNA and drop high error'
    #     x_, exp_data, error, timecourses, timecourses_err = defineExp(data_,'', '','')       
    #     index = k_CV - 1
    #     p = cal_parameters[index]
        
    #     x = []
    #     for doses_ in x_:
    #         if doses_[3] == 1:
    #             doses = [doses_[0], doses_[1], doses_[2], .2, doses_[4]] #.2fM = 200aM
    #         else:
    #             doses = doses_
    #         x.append(doses)
            
        
    #     x, solutions_norm, mse, dfSimResults = solveAll(p, exp_data, x)
    #     fmax_list, t_half_list = plotPerformanceMetricTradeoffs(solutions_norm, k_CV)
    #     fmax_list_all_k.append(fmax_list)
    #     t_half_list_all_k.append(t_half_list)
        
    # fmax_mean_list = [np.mean(sub_list) for sub_list in zip(*fmax_list_all_k)]
    # fmax_sd_list = [np.std(sub_list) for sub_list in zip(*fmax_list_all_k)]
    
    # t_half_mean_list = [np.mean(sub_list) for sub_list in zip(*t_half_list_all_k)]
    # t_half_sd_list = [np.std(sub_list) for sub_list in zip(*t_half_list_all_k)]
    
    # plot_tradeoff(x, fmax_mean_list,  t_half_mean_list, k_CV, fmax_sd_list, t_half_sd_list)  
    
    # with open("fmax_mean_200aM.json", 'w') as f:
    #     json.dump(fmax_mean_list, f, indent=2) 
        
    # with open("fmax_sd_200aM.json", 'w') as f:
    #     json.dump(fmax_sd_list, f, indent=2) 
        
    # with open("t_half_mean_200aM.json", 'w') as f:
    #     json.dump(t_half_mean_list, f, indent=2) 
        
    # with open("t_half_sd_200aM.json", 'w') as f:
    #     json.dump(t_half_sd_list, f, indent=2) 
    
    # with open("x_doses_200aM.json", 'w') as f:
    #     json.dump(x, f, indent=2) 
            
        
    # =============================================================================
    #  end plot prediction - 200aM 
    # =============================================================================
       






        

