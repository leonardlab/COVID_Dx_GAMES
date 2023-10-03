import os
import numpy as np
import matplotlib.pyplot as plt
from math import log10
from copy import deepcopy
from typing import Tuple
from gen_mechanism import *
      
#GAMES imports
import Settings_COVID_Dx
from Saving_COVID_Dx import createFolder
from Run_COVID_Dx import solveAll

#Define settings
conditions_dictionary, initial_params_dictionary, data_dictionary = Settings_COVID_Dx.init()
full_path = conditions_dictionary["directory"]
model = conditions_dictionary["model"]
data = conditions_dictionary["data"]
p_all = conditions_dictionary["p_all"]
real_param_labels_all = conditions_dictionary["real_param_labels_all"] 
error = data_dictionary["error"]
exp_data = data_dictionary["exp_data"]
timecourses_err = data_dictionary["timecourses_err"]
timecourses_exp = data_dictionary["timecourses"]
problem_free = conditions_dictionary["problem"]
bounds = problem_free['bounds']

#Note that this path needs to be updated before running
plt.style.use('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/paper.mplstyle.py')

def single_param_sweep_10pct(p: list, param_index: int
) -> Tuple[float, float]:

    """
    Solves model ODEs for all conditions (component doses) for two 
        cases of changing parameter at param_index: increase by 10%
        and decrease by 10%

    Args:
        p: a list of floats defining the parameter values for all 
            potentially free parameters (Settings_COVID_Dx.py
            conditions_dictionary["p_all"])

        param_index: an integer defining the index of the parameter for the sweep

    Returns:
        mse_low: a float defining the mse resulting from the 10% decrease in
            the parameter

        mse_high: a float defining the mse resulting from the 10% increase in
            the parameter
    """

    p_vals = deepcopy(p)
    p_low = p_vals[param_index] * 0.9
    p_high = p_vals[param_index] * 1.1
    p_vals[param_index] = p_low
    _, _, mse_low, _ = solveAll(p_vals, exp_data, '')

    p_vals[param_index] = p_high
    _, _, mse_high, _ = solveAll(p_vals, exp_data, '')   
    
    return mse_low, mse_high

def calc_percent_change(mse_mid: float, mse_new: float) -> float:
    """
    Calculates the percent change between mse_mid and mse_new

    Args:
        mse_mid: a float defining the mse for the original parameter set

        mse_new: a float defining the mse for the parameter set with increased
            or decreased parameter value

    Returns:
        100 * (mse_new-mse_mid)/mse_mid: a float defining percent change
    """

    return 100 * (mse_new-mse_mid)/mse_mid

def all_param_sweeps_10pct(p: list) -> Tuple[list, list]:

    """
    Performs all parameter sweeps for increasing or decreasing 
        each parameter value by 10%

    Args:
        p: a list of floats defining the parameter values for all 
            potentially free parameters (Settings_COVID_Dx.py
            conditions_dictionary["p_all"])
    
    Returns:
        pct_mse_low_list: a list of floats defining the percent changes 
            for decreasing each parameter by 10%

        pct_mse_high_list: a list of floats defining the percent changes 
            for increasing each parameter by 10%
    """

    _, _, mse_mid, _ = solveAll(p, exp_data, '')
    print('mse original: ', mse_mid)
    
    pct_mse_low_list = []
    pct_mse_high_list = []
    for param_index in range(0, len(p)):
        mse_low, mse_high = single_param_sweep_10pct(p, param_index)
        # print(mse_low, mse_high)
        pct_mse_low = calc_percent_change(mse_mid, mse_low)
        pct_mse_low_list.append(pct_mse_low)
        pct_mse_high = calc_percent_change(mse_mid, mse_high)
        pct_mse_high_list.append(pct_mse_high)
        print(mse_low, pct_mse_low)
        print(mse_high, pct_mse_high)

    return pct_mse_low_list, pct_mse_high_list


def tornado_plot(
        low_vals: list, high_vals: list, 
        param_labels: list, tolerance: str
) -> None:

    """
    Creates a tornado plot for the sensitivity analysis

    Args:
        low_vals: a list of floats defining the percent changes 
            for decreasing each parameter by 10%

        high_vals: a list of floats defining the percent changes 
            for increasing each parameter by 10% 
        
        param_labels: a list of strings defining the parameter
            labels for the plot (Settings_COVID_Dx.py
            conditions_dictionary["real_param_labels_all"])

        tolerance: a string defining whether the low or high error
            tolerances were used in the ode solver for the file name
            for saving the figure
    
    Returns: none

    Figures:
        './tornado plot_' + model + '_' + data + tolerance + '.svg':
            tornado plot for the sensitivity analysis
    """
     
    num_params = len(param_labels)

    pos = np.arange(num_params) + .5 # bars centered on the y axis
    
    fig, (ax_left, ax_right) = plt.subplots(ncols=2)
    ax_left.set_title('Change in chi2 from mid to low', fontsize = 8)
    ax_right.set_title('Change in chi2 from mid to high', fontsize = 8)
    bars_left = ax_left.barh(pos, low_vals, align='center', facecolor='dimgrey')
    ax_right.set_yticks([])
    ax_left.bar_label(bars_left)
    ax_left.set_xlabel('% change in chi2')
    bars_right = ax_right.barh(pos, high_vals, align='center', facecolor='dimgrey')
    ax_left.set_yticks(pos)
    ax_right.bar_label(bars_right)
    ax_left.set_yticklabels(param_labels, ha='center', x=-.1)
    ax_right.set_xlabel('% change in chi2')
    # plt.show()
    plt.savefig('./tornado plot_' + model + '_' + data + tolerance + '.svg', dpi = 600)


def run_sensitivity_analysis(p: list, p_labels: list, tolerance: str) -> None:

    """
    Runs the sensitivity analysis by calling the above functions

    Args:
        p: a list of floats defining the parameter values for all 
            potentially free parameters (Settings_COVID_Dx.py
            conditions_dictionary["p_all"])

        p_labels: a list of strings defining the parameter
            labels for the tornado plot (Settings_COVID_Dx.py
            conditions_dictionary["real_param_labels_all"])

        tolerance: a string defining whether the low or high error
            tolerances were used in the ode solver for the file name
            for saving the figure in tornado_plot()
    Returns: none
    """

    os.chdir(full_path)
    sub_folder_name = './Sensitivity_analysis_chi2'
    createFolder('./' + sub_folder_name)
    os.chdir('./' + sub_folder_name)

    pct_mse_low_list, pct_mse_high_list = all_param_sweeps_10pct(p)
    tornado_plot(pct_mse_low_list, pct_mse_high_list, p_labels, tolerance)

#need to set p_all = best case params from parameter estimation run
#and model to model used for parameter estimation run in
#Settings_COVID_Dx.py
run_sensitivity_analysis(p_all, real_param_labels_all, 'low_tol')
