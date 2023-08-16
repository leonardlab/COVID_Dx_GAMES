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

def plotMeanMeasurementError():
    
    ''' Purpose: plot the mean proportion measurement error for the experimental data (mean across each condition)
    
        Input: none
        
        Output: none
        
        Plots: 'proportion error distribution means.svg' - 
                plot of mean proportion measurement error for the experimental data '''

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
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
    
#plotMeanMeasurementError()  
    


def parityPlot(sim, exp, data):
    
    ''' Purpose: plot the experimental and simulated data in the form of a parity plot
    
        Input: 
            sim: a list of floats containing simulated values
            exp: a list of floats containing experimental values
            data: a string defining the data identity
        
        Output: none
        
        Plots: FIT TO TRAINING DATA PARITY PLOT.svg - 
                parity plot of the experimental and simulated data'''
    
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


def plotParamDistributions(df):
    ''' Purpose: Plot distribution of all parameter sets for R2 values within 10% of the highest value
    
        Input: 
            df: df containing optimization results
         
        Output: none
        
        Plots: OPTIMIZED PARAMETER DISTRIBUTIONS.svg - 
                Plot of distribution of all parameter sets for R2 values within 10% of the highest value'''

    #Only keep rows with Rsq within 10% of the highest value
    chi2 = df.sort_values(by=['chi_sq'])
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
    ''' Purpose: Plot min bound, max bound, and calibrated value for each parameter
    
        Input: 
            cal_params: list of floats defining the calibrated parameters
            bounds: list of lists defining the bounds for each parameter (each inner list is structured such that [min_bound, max_bound])
         
        Output: none
        
        Plots: COMPARISON OF CALIBRATED PARAMETERS AND BOUNDS.svg - 
                Plot of min bound, max bound, and calibrated value for each parameter'''
   
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
    ''' Purpose: Plot modeling objectives involving Hill fit summary metrics (1, 2, 3)
    
        Input: 
            solutions: list of floats defining the data for each condition and timepoint (length = # data points total)
            
        Output: none
        
        Plots: 'MODELING OBJECTIVES 123 ' + type_ + '.svg' - 
                Plot of modeling objectives involving Hill fit summary metrics (1, 2, 3) where type_ is 'exp' or 'sim'
    '''
    
    def fitHill(y_exp, runID):
        x = list(np.linspace(0, 240, 61)) #time (min)
        
        #Set v max to the final value of the time course
        fmax = y_exp[-1]
       
        #Set v0 to the intiial value of the time course
        f0 = y_exp[0]
    
        #Define a function to calculate the residual between the input simulation value (sim) and the Hill fit (model)
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

def resultsPanel(dfSim, dfExp, dfErr, labels, varyCondition):   
    '''Purpose: Plot selected readout time courses
    
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
        if 'rep2' in data:
            y_max_1 = 0.15
            y_max_10 = 0.85
        elif 'rep3' in data:
            y_max_1 = 0.4
            y_max_10 = 0.7
        else:
            y_max_1 = 0.35
            y_max_10 = 0.85
     
        
    elif varyCondition == 'RNAse':
        objective = 4
        if 'rep2' in data:
            y_max_1 = 0.15
            y_max_10 = 0.5
        elif 'rep3' in data:
            y_max_1 = 0.35
            y_max_10 = 0.5  
        else:
            y_max_1 = 0.25
            y_max_10 = 0.85 #1.0
    
    elif varyCondition == 'T7':
        objective = 6
        if 'rep2' in data:
            y_max_1 = 0.45
            y_max_10 = 1.0
        elif 'rep3' in data:
            y_max_1 = 0.5
            y_max_10 = 1.0
        else:
            y_max_1 = 0.65
            y_max_10 = 1.0
        
        
    ax1.set_ylim(0, y_max_1)
    ax2.set_ylim(0, y_max_10)
    ax3.set_ylim(0, y_max_1)
    ax4.set_ylim(0, y_max_10)
    plt.savefig('./MODELING OBJECTIVE ' + str(objective) + '.svg', dpi = 600, bbox_inches="tight")

def plotModelingObjectives456(df_sim):
    '''Purpose: Plot selected readout time courses for objectives 4, 5, and 6
    
       Input: 
            dfSim: df containing the simulated data
   
       Output: none
        
       Plots: None
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
    
def plotLowCas13(df, data_type):
    '''Purpose: Plot Fmax distributions (box plot) for low and high Cas13a-gRNA conditions
    
        Input: 
            df: df containing the data (exp or sim)
            data_type: string defining the type of data included in the df ('exp' or 'sim')
   
        Output: none
        
        Plots: LOW VS HIGH CAS13A-GRNA COMPARISON.svg - Box plot showing Fmax distributions 
            for low and high Cas13a-gRNA conditions
    
    '''
    # 
    background_only = []
    lowCas13_only = []
    highCas13_only = []

    if data_type == 'exp':
        
        if 'rep2' in data:
            maxVal = 2.94995531724754
        elif 'rep3' in data:
            maxVal = 1.12314566577301
        else:        
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
    plt.savefig('./LOW VS HIGH CAS13A-GRNA COMPARISON.svg')
    
def plot_all_states(
        df: pd.DataFrame, 
        dose: str, 
        params: str,
        t_len: str
    ):
    
    if dose == 'mid':
        doses = [5.0, 2.5, 0.005, 1, 90]

    elif dose == 'opt':
        doses = [5.0, 10.0, 0.02, 1, 90]

    elif dose == 'high RNase H':
        doses = [5.0, 2.5, 0.02, 1, 90]

    elif dose == 'high RNase H 10fM':
        doses = [5.0, 2.5, 0.02, 10, 90]
    
    time = np.linspace(0, 240, 61)
    t_points = 61
    if t_len == '2 hours':
        time = time[:31]
        t_points = 31

    fig, axs = plt.subplots(nrows=7, ncols=5, sharex=False, sharey=False, figsize = (10, 15))
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0)
    axs = axs.ravel()

    for i, state in enumerate(model_states):
        axs[i].plot(time, df.at[state, str(doses)][:t_points])
        axs[i].set_xlabel('time (min)')
        axs[i].set_ylabel('simulation value')
        axs[i].set_title(state)
        axs[i].set_box_aspect(1)
    axs[-1].axis('off')
    axs[-2].axis('off')
    fig.suptitle('All Model States '+params+' '+dose)
    # plt.show()
    plt.savefig('notitle_All_model_states_'+params+'_'+dose+'.svg')

def plot_states_RHS(
        df: pd.DataFrame,
        dose: str,
        params: str,
        p: list,
        t_len: str
    ):

    if dose == 'mid':
        doses = [5.0, 2.5, 0.005, 1, 90]

    elif dose == 'opt':
        doses = [5.0, 10.0, 0.02, 1, 90]
    
    time = np.linspace(0, 240, 61)
    t_points = 61
    if t_len == '2 hours':
        time = time[:31]
        t_points = 31


    #Parameters
    k_cas13  = p[0] #nM-1 min-1
    k_degv = p[1] #nM-1 min-1
    k_txn = p[2] #min-1
    k_FSS = p[3] #min-1
    k_RHA = p[4] #min-1
    k_bds = k_cas13 #nM-1 min-1
    k_RTon = .024 #nM-1 min-1
    k_RToff = 2.4 #min-1
    k_T7on = 3.36 #nM-1 min-1
    k_T7off = 12 #min-1
    k_SSS = k_FSS #min-1
    k_degRrep = k_degv  #nM-1 min-1
    k_RNaseon = .024 #nM-1 min-1
    k_RNaseoff = 2.4 #min-1
    k_loc_deactivation = p[5]
    k_scale_deactivation = p[6]

    all_state_tcs = df[str(doses)].tolist()
    
    x_v_list, x_p1_list, x_p2_list, x_p1v_list, x_p2u_list, x_p1cv_list, x_p2cu_list, x_RT_list, x_RNase_list, x_RTp1v_list, x_RTp2u_list, x_RTp1cv_list \
        , x_RTp2cu_list, x_cDNA1v_list, x_cDNA2u_list, x_RNasecDNA1v_list, x_RNasecDNA2u_list, x_cDNA1_list, x_cDNA2_list, x_p2cDNA1_list, x_p1cDNA2_list, x_RTp2cDNA1_list \
        , x_RTp1cDNA2_list, x_T7_list, x_pro_list, x_T7pro_list, x_u_list, x_iCas13_list, x_Cas13_list, x_uv_list, x_qRf_list, x_q_list, x_f_list = all_state_tcs

    dist = expon(loc = k_loc_deactivation, scale = k_scale_deactivation)
    frac_act_list = dist.sf(time)
    x_aCas13_list = [frac_act*x_Cas13 for frac_act, x_Cas13 in zip(frac_act_list, x_Cas13_list)]

    x_v_RHS = [k_degv*x_v*x_aCas13 - k_bds*x_v*x_u - k_bds*x_v*x_p1  \
        for x_v, x_aCas13, x_u, x_p1 in zip(
        x_v_list,
        x_aCas13_list,
        x_u_list,
        x_p1_list
    )]
    x_p1_RHS = [- k_bds*x_v*x_p1- k_bds*x_p1*x_cDNA2  \
        for x_v, x_p1, x_cDNA2 in zip(
            x_v_list,
            x_p1_list, 
            x_cDNA2_list
        )]
    x_p2_RHS = [- k_bds*x_u*x_p2 - k_bds*x_p2*x_cDNA1  \
        for x_u, x_p2, x_cDNA1 in zip(
            x_u_list,
            x_p2_list,
            x_cDNA1_list
        )]
    x_p1v_RHS = [k_bds*x_v*x_p1 - k_degv*x_p1v*x_aCas13 - k_RTon*x_p1v*x_RT + k_RToff*x_RTp1v  \
        for x_v, x_p1, x_p1v, x_aCas13, x_RT, x_RTp1v in zip(
        x_v_list,
        x_p1_list,
        x_p1v_list,
        x_aCas13_list, 
        x_RT_list,
        x_RTp1v_list
    )]
    x_p2u_RHS = [k_bds*x_u*x_p2 - k_degv*x_p2u*x_aCas13 - k_RTon*x_p2u*x_RT + k_RToff*x_RTp2u  \
        for x_u, x_p2, x_p2u, x_aCas13, x_RT, x_RTp2u in zip(
        x_u_list,
        x_p2_list,
        x_p2u_list,
        x_aCas13_list,
        x_RT_list,
        x_RTp2u_list
        )]
    x_p1cv_RHS = [k_degv*x_p1v*x_aCas13 - k_RTon*x_p1cv*x_RT + k_RToff*x_RTp1cv  \
        for x_p1v, x_aCas13, x_p1cv, x_RT, x_RTp1cv in zip(
        x_p1v_list,
        x_aCas13_list,
        x_p1cv_list,
        x_RT_list,
        x_RTp1cv_list
        )]
    x_p2cu_RHS = [k_degv*x_p2u*x_aCas13 - k_RTon*x_p2cu*x_RT + k_RToff*x_RTp2cu  \
        for x_p2u, x_aCas13, x_p2cu, x_RT, x_RTp2cu in zip(
        x_p2u_list,
        x_aCas13_list,
        x_p2cu_list,
        x_RT_list,
        x_RTp2cu_list
        )]
    x_RT_RHS = [+ k_RToff*x_RTp1v + k_RToff*x_RTp1cv + k_RToff*x_RTp2cDNA1 + k_RToff*x_RTp2u
        + k_RToff*x_RTp2cu + k_RToff*x_RTp1cDNA2 - k_RTon*x_RT*x_p1v - k_RTon*x_RT*x_p1cv
        - k_RTon*x_RT*x_p2cDNA1 - k_RTon*x_RT*x_p2u - k_RTon*x_RT*x_p2cu - k_RTon*x_RT*x_p1cDNA2
        + k_FSS*x_RTp1v + k_FSS*x_RTp2u + k_SSS*x_RTp2cDNA1 + k_SSS*x_RTp1cDNA2  \
        for x_RTp1v, x_RTp1cv, x_RTp2cDNA1, x_RTp2u, x_RTp2cu, x_RTp1cDNA2, x_RT, x_p1v,
        x_p1cv, x_p2cDNA1, x_p2u, x_p2cu, x_p1cDNA2 in zip(
        x_RTp1v_list,
        x_RTp1cv_list,
        x_RTp2cDNA1_list,
        x_RTp2u_list,
        x_RTp2cu_list,
        x_RTp1cDNA2_list,
        x_RT_list,
        x_p1v_list,
        x_p1cv_list,
        x_p2cDNA1_list,
        x_p2u_list,
        x_p2cu_list,
        x_p1cDNA2_list  
        )]
    x_RNase_RHS = [+ k_RNaseoff*x_RNasecDNA1v + k_RNaseoff*x_RNasecDNA2u- k_RNaseon*x_RNase*x_cDNA1v
        - k_RNaseon*x_RNase*x_cDNA2u + k_RHA*x_RNasecDNA1v + k_RHA*x_RNasecDNA2u  \
        for x_RNasecDNA1v, x_RNasecDNA2u, x_RNase, x_cDNA1v, x_cDNA2u in zip(
        x_RNasecDNA1v_list, 
        x_RNasecDNA2u_list,
        x_RNase_list,
        x_cDNA1v_list,
        x_cDNA2u_list
        )]
    x_RTp1v_RHS = [- k_RToff*x_RTp1v + k_RTon*x_RT*x_p1v - k_degv*x_RTp1v*x_aCas13 - k_FSS*x_RTp1v  \
        for x_RTp1v, x_RT, x_p1v, x_aCas13 in zip(
        x_RTp1v_list,
        x_RT_list,
        x_p1v_list,
        x_aCas13_list 
        )]
    x_RTp2u_RHS = [- k_RToff*x_RTp2u + k_RTon*x_RT*x_p2u - k_degv*x_RTp2u*x_aCas13 - k_FSS*x_RTp2u  \
        for x_RTp2u, x_RT, x_p2u, x_RTp2u, x_aCas13 in zip(
        x_RTp2u_list,
        x_RT_list,
        x_p2u_list,
        x_RTp2u_list,
        x_aCas13_list    
        )]
    x_RTp1cv_RHS = [- k_RToff*x_RTp1cv + k_RTon*x_RT*x_p1cv + k_degv*x_RTp1v*x_aCas13  \
        for x_RTp1cv, x_RT, x_p1cv, x_RTp1v, x_aCas13 in zip(
        x_RTp1cv_list,
        x_RT_list,
        x_p1cv_list,
        x_RTp1v_list,
        x_aCas13_list
        )]
    x_RTp2cu_RHS = [- k_RToff*x_RTp2cu + k_RTon*x_RT*x_p2cu + k_degv*x_RTp2u*x_aCas13  \
        for x_RTp2cu, x_RT, x_p2cu, x_RTp2u, x_aCas13 in zip(
        x_RTp2cu_list,
        x_RT_list,
        x_p2cu_list,
        x_RTp2u_list,
        x_aCas13_list
        )]
    x_cDNA1v_RHS = [k_FSS*x_RTp1v - k_RNaseon*x_cDNA1v*x_RNase + k_RNaseoff*x_RNasecDNA1v  \
        for x_RTp1v, x_cDNA1v, x_RNase, x_RNasecDNA1v in zip(
        x_RTp1v_list,
        x_cDNA1v_list,
        x_RNase_list,
        x_RNasecDNA1v_list
        )]
    x_cDNA2u_RHS = [k_FSS*x_RTp2u - k_RNaseon*x_cDNA2u*x_RNase + k_RNaseoff *x_RNasecDNA2u  \
        for x_RTp2u, x_cDNA2u, x_RNase, x_RNasecDNA2u in zip(
        x_RTp2u_list,
        x_cDNA2u_list,
        x_RNase_list,
        x_RNasecDNA2u_list
        )]
    x_RNasecDNA1v_RHS = [- k_RHA*x_RNasecDNA1v - k_RNaseoff*x_RNasecDNA1v + k_RNaseon*x_RNase*x_cDNA1v  \
        for x_RNasecDNA1v, x_RNase, x_cDNA1v in zip(
        x_RNasecDNA1v_list,
        x_RNase_list,
        x_cDNA1v_list
        )]
    x_RNasecDNA2u_RHS = [- k_RHA*x_RNasecDNA2u - k_RNaseoff*x_RNasecDNA2u + k_RNaseon*x_RNase*x_cDNA2u  \
        for x_RNasecDNA2u, x_RNase, x_cDNA2u in zip(
        x_RNasecDNA2u_list,
        x_RNase_list,
        x_cDNA2u_list
        )]
    x_cDNA1_RHS = [k_RHA*x_RNasecDNA1v - k_bds*x_cDNA1*x_p2  \
        for x_RNasecDNA1v, x_cDNA1, x_p2 in zip(
        x_RNasecDNA1v_list,
        x_cDNA1_list,
        x_p2_list
        )]
    x_cDNA2_RHS = [k_RHA*x_RNasecDNA2u - k_bds*x_cDNA2*x_p1  \
        for x_RNasecDNA2u, x_cDNA2, x_p1 in zip(
        x_RNasecDNA2u_list,
        x_cDNA2_list,
        x_p1_list
        )]
    x_p2cDNA1_RHS = [k_bds*x_cDNA1*x_p2 + k_RToff*x_RTp2cDNA1 - k_RTon*x_RT*x_p2cDNA1  \
        for x_cDNA1, x_p2, x_RTp2cDNA1, x_RT, x_p2cDNA1 in zip(
        x_cDNA1_list,
        x_p2_list,
        x_RTp2cDNA1_list,
        x_RT_list,
        x_p2cDNA1_list
        )]
    x_p1cDNA2_RHS = [k_bds*x_cDNA2*x_p1 + k_RToff*x_RTp1cDNA2 - k_RTon*x_RT*x_p1cDNA2  \
        for x_cDNA2, x_p1, x_RTp1cDNA2, x_RT, x_p1cDNA2 in zip(
        x_cDNA2_list,
        x_p1_list,
        x_RTp1cDNA2_list,
        x_RT_list,
        x_p1cDNA2_list
        )]
    x_RTp2cDNA1_RHS = [k_RTon*x_RT*x_p2cDNA1 - k_RToff*x_RTp2cDNA1 - k_SSS*x_RTp2cDNA1  \
        for x_RT, x_p2cDNA1, x_RTp2cDNA1 in zip(
        x_RT_list,
        x_p2cDNA1_list,
        x_RTp2cDNA1_list
        )]
    x_RTp1cDNA2_RHS = [k_RTon*x_RT*x_p1cDNA2 - k_RToff*x_RTp1cDNA2 - k_SSS*x_RTp1cDNA2  \
        for x_RT, x_p1cDNA2, x_RTp1cDNA2 in zip(
        x_RT_list,
        x_p1cDNA2_list,
        x_RTp1cDNA2_list
        )]
    x_T7_RHS = [+ k_T7off*x_T7pro - k_T7on*x_T7*x_pro + k_txn*x_T7pro  \
        for x_T7pro, x_T7, x_pro in zip(
        x_T7pro_list,
        x_T7_list,
        x_pro_list
        )]
    x_pro_RHS = [k_SSS*x_RTp2cDNA1 + k_SSS*x_RTp1cDNA2 - k_T7on*x_T7*x_pro + k_T7off*x_T7pro + k_txn*x_T7pro  \
        for x_RTp2cDNA1, x_RTp1cDNA2, x_T7, x_pro, x_T7pro in zip(
        x_RTp2cDNA1_list,
        x_RTp1cDNA2_list, 
        x_T7_list,
        x_pro_list,
        x_T7pro_list
        )]
    x_T7pro_RHS = [- k_T7off*x_T7pro + k_T7on*x_T7*x_pro - k_txn*x_T7pro  \
        for x_T7pro, x_T7, x_pro in zip(
        x_T7pro_list,
        x_T7_list,
        x_pro_list
        )]
    x_u_RHS = [k_txn*x_T7pro - k_bds*x_u*x_v - k_degv*x_u*x_aCas13 - k_cas13*x_u*x_iCas13 - k_bds*x_u*x_p2  \
        for x_T7pro, x_u, x_v, x_aCas13, x_iCas13, x_p2 in zip(
        x_T7pro_list,
        x_u_list,
        x_v_list,
        x_aCas13_list,
        x_iCas13_list,
        x_p2_list
        )]
    x_iCas13_RHS = [- k_cas13*x_u*x_iCas13  \
        for x_u, x_iCas13 in zip(
            x_u_list, 
            x_iCas13_list
        )]
    x_Cas13_RHS = [k_cas13*x_u*x_iCas13  \
        for x_u, x_iCas13 in zip(
        x_u_list,
        x_iCas13_list
        )]
    x_uv_RHS = [k_bds*x_u*x_v  \
        for x_u, x_v in zip(
            x_u_list,
            x_v_list
        )]
    x_qRf_RHS = [- k_degRrep*x_aCas13*x_qRf  \
        for x_aCas13, x_qRf in zip(
            x_aCas13_list, 
            x_qRf_list
        )]
    x_q_RHS = [+ k_degRrep*x_aCas13*x_qRf  \
        for x_aCas13, x_qRf in zip(
        x_aCas13_list, 
        x_qRf_list
        )]
    x_f_RHS = [+ k_degRrep*x_aCas13*x_qRf  \
        for x_aCas13, x_qRf in zip(
        x_aCas13_list, 
        x_qRf_list
        )]

    all_state_RHS = [
        x_v_RHS, 
        x_p1_RHS,
        x_p2_RHS,
        x_p1v_RHS,
        x_p2u_RHS,
        x_p1cv_RHS,
        x_p2cu_RHS,
        x_RT_RHS,
        x_RNase_RHS,
        x_RTp1v_RHS,
        x_RTp2u_RHS,
        x_RTp1cv_RHS,
        x_RTp2cu_RHS,
        x_cDNA1v_RHS,
        x_cDNA2u_RHS,
        x_RNasecDNA1v_RHS,
        x_RNasecDNA2u_RHS,
        x_cDNA1_RHS,
        x_cDNA2_RHS,
        x_p2cDNA1_RHS,
        x_p1cDNA2_RHS,
        x_RTp2cDNA1_RHS,
        x_RTp1cDNA2_RHS,
        x_T7_RHS,
        x_pro_RHS,
        x_T7pro_RHS,
        x_u_RHS,
        x_iCas13_RHS,
        x_Cas13_RHS,
        x_uv_RHS,
        x_qRf_RHS,
        x_q_RHS,
        x_f_RHS 
    ]

    fig, axs = plt.subplots(nrows=7, ncols=5, sharex=False, sharey=False, figsize = (10, 15))
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0)
    axs = axs.ravel()

    for i, state in enumerate(model_states):
        axs[i].plot(time, all_state_RHS[i][:t_points])
        axs[i].set_xlabel('time (min)')
        axs[i].set_ylabel('ODE RHS sim. value')
        # axs[i].set_title(state)
        axs[i].set_box_aspect(1)
    
    axs[-1].axis('off')
    axs[-2].axis('off')
    fig.suptitle('All States RHS '+params+' '+dose)
    # plt.show()
    plt.savefig('notitle_All_states_RHS_'+params+'_'+dose+'.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize = (2, 3))
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0)

    ax.plot(time, x_aCas13_list[:t_points])
    ax.set_xlabel('time (min)')
    ax.set_ylabel('sim. value')
    plt.savefig('Active_cas13_time_series_'+params+'_'+dose+'.svg')

    
    
    
    
    
