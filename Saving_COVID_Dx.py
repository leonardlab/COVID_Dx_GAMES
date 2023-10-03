#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

"""
Created on Mon Sep 28 11:13:10 2020

@author: kate
"""

import os

def makeMainDir(folder_name: str) -> str:    
    
    ''' 
    Creates the main directory to hold simulation results 
        
    Args: 
        folder_name: a string defining the name of the folder
   
    Returns: 
        results_folder_path + folder_name: a string defining the location of
            the new folder
    '''

    results_folder_path = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/'
    
    try:
        if not os.path.exists(results_folder_path + folder_name):
            os.makedirs(results_folder_path + folder_name)
            print('Main directory created at: ' + folder_name)
            print('***')
           
    except OSError:
        print ('Directory already exists')

    return results_folder_path + folder_name

def createFolder(directory: str) -> None:
    
    ''' 
    Creates a new folder.
        
    Args: 
        directory: a string defining the name of the folder to make
       
    Returns: None
    '''

    try:
        if not os.path.exists('./' + directory):
            os.makedirs('./' + directory)
    except OSError:
        print ('Error: Creating directory. ' +  './' + directory)

def saveConditions(
        conditions_dictionary: dict, initial_params_dictionary: dict, 
        data_dictionary: dict
) -> None: 
    
    ''' 
    Saves conditions, initial params, and data dictionaries as defined
        in Settings_COVID_Dx.py
        
    Args: 
        conditions_dictionary: a dictionary holding simulation conditions
            (defined in Settings_COVID_Dx.py)

        initial_params_dictionary: a dictionary holding initial parameters
            and related information (defined in Settings_COVID_Dx.py)

        data_dictionary: a dictionary holding experimental data and related
            information (defined in Settings_COVID_Dx.py)

    Returns: none
    
    Files:
        CONDITIONS.txt:
            holds the conditions defined in the dictionaries
    '''

    filename = 'CONDITIONS' 
    with open(filename + '.txt', 'w') as file:
        #f.write('Run ID: ' + str(runID) + '\n')
        file.write('Conditions: ' + str(conditions_dictionary) + '\n')
        file.write('\n')
        file.write('Initial parameters: ' + str(initial_params_dictionary) + '\n')
        file.write('\n')
        file.write('Data: ' + str(data_dictionary) + '\n')
    print('Conditions saved.')
    
