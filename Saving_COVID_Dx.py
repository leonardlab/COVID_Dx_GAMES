#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:13:10 2020

@author: kate
"""

import os

def makeMainDir(folder_name):    
    ''' 
    Purpose: Create the main directory to hold simulation results 
        
    Inputs: 
        folder_name: a string defining the name of the folder
   
    Output: 
        results_folder_path + folder_name: a string defining the location of the new folder
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

def createFolder(directory):
    ''' 
    Purpose: Create a new folder.
        
    Inputs: 
        directory: a string defining the name of the folder to make
       
    Output: None
    '''
    
    try:
        if not os.path.exists('./' + directory):
            os.makedirs('./' + directory)
    except OSError:
        print ('Error: Creating directory. ' +  './' + directory)

def saveConditions(conditions_dictionary, initial_params_dictionary, data_dictionary):  
    ''' 
    Purpose: Save conditions, initial params, and data dictionaries
        
    Inputs: 
        conditions_dictionary, initial_params_dictionary, data_dictionary: dictionaries holding
        the simulation conditions
       
    Output: None
    
    Files:
        CONDITIONS.txt (holds the conditions defined in the dictionaries)
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
    
