import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log10
import math
import seaborn as sns
import itertools
from copy import deepcopy
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
plt.style.use('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/paper.mplstyle.py')

def single_param_sweep_10pct(p, param_index):
    p_vals = deepcopy(p)
    p_low = p_vals[param_index] * 0.9
    p_high = p_vals[param_index] * 1.1
    p_vals[param_index] = p_low
    _, _, mse_low, _ = solveAll(p_vals, exp_data, '')

    p_vals[param_index] = p_high
    _, _, mse_high, _ = solveAll(p_vals, exp_data, '')   
    
    return mse_low, mse_high

def calc_percent_change(mse_mid, mse_new):
    return 100 * (mse_new-mse_mid)/mse_mid

def all_param_sweeps_10pct(p):
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


def tornado_plot(low_vals, high_vals, param_labels):
     
    num_params = len(param_labels)

    pos = np.arange(num_params) + .5 # bars centered on the y axis
    
    fig, (ax_left, ax_right) = plt.subplots(ncols=2)
    ax_left.set_title('Change in chi2 from mid to low', fontsize = 8)
    ax_right.set_title('Change in chi2 from mid to high', fontsize = 8)
    ax_left.barh(pos, low_vals, align='center', facecolor='lightcoral')
    ax_right.set_yticks([])
    ax_left.set_xlabel('% change in chi2')
    ax_right.barh(pos, high_vals, align='center', facecolor='mediumaquamarine')
    ax_left.set_yticks(pos)
    ax_left.set_yticklabels(param_labels, ha='center', x=-.1)
    ax_right.set_xlabel('% change in chi2')
    # plt.show()
    plt.savefig('./tornado plot_' + model + '_' + data + '.svg', dpi = 600)


def run_sensitivity_analysis(p, p_labels):
    os.chdir(full_path)
    sub_folder_name = './Sensitivity_analysis_chi2'
    createFolder('./' + sub_folder_name)
    os.chdir('./' + sub_folder_name)

    pct_mse_low_list, pct_mse_high_list = all_param_sweeps_10pct(p)
    tornado_plot(pct_mse_low_list, pct_mse_high_list, p_labels)

#need to set p_all = opt params from run
run_sensitivity_analysis(p_all, real_param_labels_all)
