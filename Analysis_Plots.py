#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:25:15 2020

@author: kate
"""

#Package imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import log10
from lmfit import Parameters, minimize
from Solvers_COVID_Dx import calcRsq
from scipy.stats import expon
from typing import Tuple


#GAMES imports
import Settings_COVID_Dx

#Unpack conditions from Settings.py
conditions_dictionary, initial_params_dictionary, data_dictionary = Settings_COVID_Dx.init()
exp_data = data_dictionary["exp_data"]
x = data_dictionary["x_vals"]
real_param_labels_all = conditions_dictionary['real_param_labels_all']
model = conditions_dictionary['model']
param_labels = real_param_labels_all 
error = data_dictionary["error"]
model_states = conditions_dictionary["model states"]
data = conditions_dictionary["data"]

#import dataframes for experimental data and error based on dataset being used
#(Settings_COVID_Dx.py conditions_dictionary["data"])

#Note that paths need to be updated before running
#dataset 2
if 'rep2' in data:
    df_data = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED_DATA_rep2_EXP.pkl')
    df_error = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED_DATA_rep2_ERR.pkl') 

#dataset 3
elif 'rep3' in data:
    df_data = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED_DATA_rep3_EXP.pkl')
    df_error = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED_DATA_rep3_ERR.pkl')

#dataset 1
else:
    df_data = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED DATA EXP.pkl')
    df_error = pd.read_pickle('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/PROCESSED DATA ERR.pkl')

#Import custom style file for plotting
#Note that path needs to be updated before running
plt.style.use('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/paper.mplstyle.py')
dpi_ = 600

def plotMeanMeasurementError() -> None:
    
    '''
    Plots the mean proportion measurement error for the experimental data
        (mean across each condition).
    Uses experimental data and error as defined in Settings_COVID_Dx.py
    
    Args: none
        
    Returns: none
        
    Figures: 
        'proportion error distribution means.svg': 
            a plot of mean proportion measurement error for the experimental data 
    '''

    def chunks(lst: list, n: int) -> list:
        """
        Yield successive n-sized chunks from lst
        
        Args:
            lst: a list of values
            
            n: an integer defining the size of each chunk

        Returns:
            lst[i:i + n]: a list of lists containing the values
            from lst structured as n-sized chunks
        """

        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
    error_lists = list(chunks(error, 61))
    data_lists = list(chunks(exp_data, 61))

    proportion_error_means = []
    for i, list_ in enumerate(error_lists):
        data_list = data_lists[i]
        error_list = error_lists[i]
        per_error_list = [j/max(data_list) for j in error_list]
        proportion_error_means.append(np.mean(per_error_list))
   
    fig = plt.figure(figsize = (4,3))
    plt.hist(proportion_error_means, bins=50, color = 'dimgrey')
    plt.xlabel('mean proportion_error')
    plt.ylabel('count')
    plt.savefig('./proportion error distribution means.svg')


def parityPlot(sim: list, exp: list, data: str) -> None:
    
    ''' 
    Plots the experimental and simulated data in the form of a parity plot
    
    Args: 
        sim: a list of floats containing simulated values

        exp: a list of floats containing experimental values

        data: a string defining the data identity
        
    Returns: none
        
    Figures: 
        'FIT TO TRAINING DATA PARITY PLOT.svg':
            a parity plot of the experimental vs simulated data
    '''
    
    if data == 'PEM evaluation':
        color_ = [i/255 for i in [204, 121, 167]] #PINK
    
    if data == 'cross validation ensemble':
        color_ = 'steelblue'
   
    else:
        color_ = 'black'
        
    fig = plt.figure(figsize = (3.375,3))
    plt.plot(sim, exp, marker = 'o', markersize = 1, linestyle = 'none', color = color_)
    plt.ylabel('Experimental value')
    plt.xlabel('Simulated value')
    x_ = [0, 1]
    y_ = [0, 1]
    plt.plot(x_, y_, linestyle = '-', marker = None, color = 'dimgrey')
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig('./FIT TO TRAINING DATA PARITY PLOT.svg', dpi = 600, bbox_inches="tight")


def plotParamDistributions(df: pd.DataFrame) -> None:
    
    ''' 
    Plots the distribution of parameter values for all parameter sets 
        with R2 values within 10% of the highest value
    
    Args: 
        df: a dataframe containing optimization results
         
    Returns: none
        
    Figures: 
        'OPTIMIZED PARAMETER DISTRIBUTIONS.svg':
            a plot of distribution of parameter values for all parameter
            sets with R2 values within 10% of the highest value
    '''

    #Only keep rows with Rsq within 10% of the highest value
    chi2 = df.sort_values(by=['chi_sq'])
    max_R2 = df['Rsq'][0]
    cutoff_R2 = max_R2 - (max_R2 * .1)
    df = df[df['Rsq']>= cutoff_R2]
   
    #Restructure df for plotting
    df_new = pd.DataFrame(columns = [
        'k_cas13', 
        'k_degv', 
        'k_txn', 
        'k_FSS', 
        'k_RHA', 
        'k_loc_deactivation', 
        'k_txn_poisoning', 
        'n_txn_poisoning',
        'k_scale_deactivation', 
        'Rsq'])
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
     
        
def plotParamBounds(cal_params: list, bounds: list) -> None:
    
    '''
    Plots the min bound, max bound, and calibrated value for each parameter
    
    Args: 
        cal_params: a list of floats defining the calibrated parameters

        bounds: a list of lists defining the bounds for each parameter (each 
        inner list is structured as [min_bound, max_bound])
         
    Returns: none
        
    Figures: 
        'COMPARISON OF CALIBRATED PARAMETERS AND BOUNDS.svg':
            a plot of min bound, max bound, and calibrated value for each
            parameter
    '''
   
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

    
def plotModelingObjectives123(solutions: list) -> None:
    
    '''
    Plots the modeling objectives involving Hill fit summary metrics
        (1, 2, 3).
    Can also be used to plot the Hill fit summary metrics for the
        experimental data.
    
    Args: 
        solutions: a list of floats defining the data for each condition
        and timepoint (length = # data points total, for all doses at all 
        timepoints)
            
    Returns: none
    '''
    
    def fitHill(y_exp: list, runID: str) -> Tuple[float, float, float, float, float]:

        """
        Fits data to a hill function for a single set of
            conditions (component doses).

        Args: 
            y_exp: a list of floats defining the normalized 
                simulation values for a single set of conditions

            runID: a string defining the run ID (for plotting)

        Returns:
            f0: a float defining the initial value in the dataset

            fmax: a float defining the final value in the dataset

            km: a float defining the optimized parameter for t1/2

            n: a float defining the optimized parameter for the 
                Hill coefficient

            R_sq: a float defining the R squared value between the 
                data and the Hill fit
        """

        x = list(np.linspace(0, 240, 61)) #time (min)
        
        #Set v max to the final value of the time course
        fmax = y_exp[-1]
       
        #Set v0 to the intiial value of the time course
        f0 = y_exp[0]
    
        def residual(p: list, x: list, y_exp: list) -> float:

            """
            Calculates the residual between the input simulation
                values and the Hill fit. Used in the minimization
                function as the cost function to be minimized
                between the simulation and the Hill fit.

            Args:
                p: a list of floats defining the parameters for
                    the hill function

                x: a list of floats defining the time values for
                    the simulation

                y_exp: a list of floats defining the simulation
                    values

            Returns: 
                (y_exp - model): a float defining the residual 
                between the input simulation values and the Hill 
                fit
            """

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
    def chunks(lst: list, n: int) -> list:
        """
        Yield successive n-sized chunks from lst
        
        Args:
            lst: a list of floats
            
            n: an integer defining the size of each chunk

        Returns:
            lst[i:i + n]: a list of lists containing the values
            from lst structured as n-sized chunks
        """

        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
    def fitHillAllData(data_lists: list, type_: str) -> None:

        """
        Fits data to a hill function for all conditions
            (component doses).

        Args:
            data_lists: a list of lists containing the
                simulation values for each set of
                conditions
            
            type_: a string defining the type of data 
                being fit- 'exp or 'sim'
        
        Returns: none

        Figures: 
        ''MODELING OBJECTIVES 123 ' + type_ + '.svg'':
            Plot of modeling objectives involving Hill fit summary
                metrics (1, 2, 3)
        """

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
        
        filename = 'experimental summary metrics' + model
        with pd.ExcelWriter(filename + '.xlsx') as writer:  
            df.to_excel(writer, sheet_name=' ')
    
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
        #ax3.set_xlim([50, 90])
    
        ax4.hist(n, bins=bins_, color = color_)
        ax4.set_xlabel('n')
    
        ax5.hist(R_sq, bins=bins_, color = color_)
        ax5.set_xlabel('R_sq')
        #ax5.set_xlim([0.994, 0.999])
    
        plt.savefig('MODELING OBJECTIVES 123 ' + type_ + '.svg', dpi = 600)
 
    data_lists = list(chunks(solutions, 61))
    print('Determining Hill fits...')
    fitHillAllData(data_lists, 'sim')
    print('Modeling objectives 1 2 and 3 plotted.')

def resultsPanel(
        dfSim: pd.DataFrame, dfExp: pd.DataFrame,
        dfErr: pd.DataFrame, labels: list,
        varyCondition: str
 ) -> None:   
   
    '''
    Plots selected readout time courses (sweeps) for a given enzyme
    
    Args: 
        dfSim: a dataframe containing the simulated data

        dfExp: a dataframe containing the experimental data

        dfErr: a dataframe containing the measurement error associated with
            the experimental data
        
        labels: a list of lists containing the condition labels to be plotted

        varyCondition: a string contraining the label of the enzyme condition
            that is sweeped over in the plot
   
    Returns: none
        
    Figures: 
        'MODELING OBJECTIVE ' + str(objective) + '.svg'':
            plot of readout dynamics associated with the given modeling objective
                (enzyme sweep)
    '''
    
    fig = plt.figure(figsize = (12,3))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ax1 = plt.subplot(141)   
    ax2 = plt.subplot(142)
    ax3 = plt.subplot(143)   
    ax4 = plt.subplot(144)
   
    def grabData(labels: list) -> Tuple[list, list, list]:
        """
        Compiles the simulation values and experimental
            data for the given set of labels

        Args:
            labels: a list of lists defining the labels
                (component doses) to compile data for
            
        Returns:
            sim: a list of lists defining the simulation
                values for the set of labels

            exp: a list of lists defining the experimental
                data for the set of labels

            err: a list of lists defining the experimental
                error for the set of labels
        """

        count = 0
        sim = [[0] * 61] * 3
           
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
 
        if 'rep2' in data:
            maxVal = 2.94995531724754
        elif 'rep3' in data:
            maxVal = 1.12314566577301
        else:        
            maxVal = 0.6599948235700113
        exp =  [[0] * 61] * 3
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
                
        err =  [[0] * 61] * 3
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
    
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Normalized exp output')
    ax1.set_title('vRNA = 1fM', fontsize = 10, fontweight = 'bold')
    ax3.set_title('vRNA = 1fM', fontsize = 10, fontweight = 'bold')
    ax4.set_title('vRNA = 10fM', fontsize = 10, fontweight = 'bold')

    for i in range(0, len(labels)):
        labels[i][3] = 10
    
    sim10, exp10, err10 = grabData(labels)
    
    for i in range(0, len(sim1)):
        if data == 'slice drop high error':
            if varyCondition == 'RT' and i == 0: #condition dropped due to high error
                continue
            else:
                ax4.plot(time, sim10[i],  marker = None, linestyle = 'dashed', color = colors[i])
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
        if 'rep2' in data:
            y_max_1 = 0.15
            y_max_10 = 0.85
        elif 'rep3' in data:
            y_max_1 = 0.7 #0.4
            y_max_10 = 1.0 #0.7
        else:
            y_max_1 = 0.4 #0.35
            y_max_10 = 1.0 #0.85
     
        
    elif varyCondition == 'RNAse':
        objective = 4
        if 'rep2' in data:
            y_max_1 = 0.3
            y_max_10 = 1.0 #0.5
        elif 'rep3' in data:
            y_max_1 = 0.4
            y_max_10 = 1.0 #0.5
        else:
            y_max_1 = 0.7 #0.25
            y_max_10 = 1.0 #0.85
    
    elif varyCondition == 'T7':
        objective = 6
        if 'rep2' in data:
            y_max_1 = 0.45
            y_max_10 = 1.0
        elif 'rep3' in data:
            y_max_1 = 0.5
            y_max_10 = 1.0
        else:
            y_max_1 = 0.7 #0.7, 0.65
            y_max_10 = 1.0
        
        
    ax1.set_ylim(0, y_max_1)
    ax2.set_ylim(0, y_max_10)
    ax3.set_ylim(0, y_max_1)
    ax4.set_ylim(0, y_max_10)
    plt.savefig('./MODELING OBJECTIVE ' + str(objective) + '.svg', dpi = 600, bbox_inches="tight")

def plotModelingObjectives456(df_sim: pd.DataFrame) -> None:
    
    '''
    Plots selected readout time courses for objectives 4, 5, and 6 (enzyme sweeps).
    Calls resultsPanel for each objective to generate plots
    
    Args: 
        dfSim: a dataframe containing the simulated data
   
    Returns: none
    '''
     
    cas = 90
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
    
def plotLowCas13(df: pd.DataFrame, data_type: str, rep: str) -> None:
    
    '''
    Plots Fmax distributions (box plot) for low and high Cas13a-gRNA conditions
    
    Args: 
        df: a dataframe containing the data (exp or sim)

        data_type: a string defining the type of data included in the df ('exp' 
            or 'sim')

        rep: a string defining the dataset ('rep1', 'rep2', or 'rep3'); only necessary
        for exp data, otherwise use '' 
   
    Returns: none
        
    Figures: 
        'LOW VS HIGH CAS13A-GRNA COMPARISON.svg'
            a box plot showing Fmax distributions for low and high Cas13a-gRNA conditions
    '''
    
    background_only = []
    lowCas13_only = []
    highCas13_only = []

    if data_type == 'exp':
        
        if rep == 'rep1':
            maxVal = 0.6599948235700113
        elif rep == 'rep2':
            maxVal = 2.94995531724754
        elif rep == 'rep3':
            maxVal = 1.12314566577301      
            
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
        
          def Convert(string: str) -> list:

            """
            Converts a string into a list of
                strings

            Args:
                string: a string to be converted

            Returns:
                li: a reulting list of strings 
            """

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
    plt.savefig('./' + rep + 'LOW VS HIGH CAS13A-GRNA COMPARISON.svg')


    
    
    
    
    
