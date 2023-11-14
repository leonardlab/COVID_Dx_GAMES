#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kateldray
"""
#Package imports
import pandas as pd
import json
from typing import Tuple


def defineExp(data: str, k_PEM_evaluation: int
) -> Tuple[list, list, list, list, list]:
    
    ''' 
    Imports the experimental data from a .pkl file and structures the data to be
    compatible with downstream LMFIT (optimization) code. Different portions of
    the data are used based on the data arg
    
    Args:
        data: a string defining the data identity
            
        k_PEM_evaluation: an integer defining the value of k_PEM_evaluation for the 
            run (only used when data = 'PEM evaluation')
      
    Returns:
         x: a list of lists defining the component doses for each condition. Each list has 5
            items and there are the same number of lists as number of conditions. Each individual
            list holds the component doses in the following order: [x_T7, x_RT, x_RNase, x_inputRNA
            x_Cas13]. 
            Note that the component dose for x_Cas13 represents the Cas13 component dose, not the 
            Cas13a-gRNA component dose. The amount of gRNA added is exactly half the amount of Cas13,
            so it is assumed that [Cas13a-gRNA] = [Cas13]/2. This operation (dividing [Cas13] by 2) is
            completed in Solvers.py each time the equations are solved.

         exp_data: a list of floats defining the normalized readout values 
             for each datapoint (length = # datapoints)
                                 
         error: list of floats defining the normalized readout measurement error 
             values for each datapoint (length = # datapoints)
                                        
         timecourses: a list of lists containing the same data as exp_data, but each list contains a 
             time course for a different condition   
             
         timecourses_err: a list of lists containing the same data as error, but each list contains 
             a list of error values for a different condition      
    '''

    #Import training data
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

    if data == 'PEM evaluation':
        #Import df for appropriate model and k_PEM_evaluation
        #this path needs to be updated each time a PEM evaluation is run (i.e. for different models)
        filename = ('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230924_ModelA_PEM_rep3/'
                    'GENERATE PEM EVALUATION DATA/PEM EVALUATION DATA NOISE model A.xlsx')
        df = pd.read_excel(filename, sheet_name = str(k_PEM_evaluation), engine='openpyxl')
        
        exp_data = []
        i = 0
        for column in df:
            if i != 0: #skip column 0 (indicies)
                vals = list(df[column])
                exp_data = exp_data + vals
            i += 1
        exp_data = [i/max(exp_data) for i in exp_data]
       
        #error not used for PEM evaluation paramter estimation runs, so use placeholder
        error = [1] * len(exp_data)
        
        #the x values need to be selected based on which experimental dataset is being used
        #rep 1 x vals
        x = [
            [20.0, 2.5, 0.005, 10, 90], [5.0, 10.0, 0.005, 10, 90], [5.0, 2.5, 0.02, 10, 90],
            [5.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.001, 10, 90], [1.0, 2.5, 0.005, 10, 90], 
            [20.0, 2.5, 0.005, 1, 90], [5.0, 10.0, 0.005, 1, 90], [5.0, 2.5, 0.02, 1, 90],
            [5.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.001, 1, 90], [5.0, 0.5, 0.005, 1, 90], 
            [1.0, 2.5, 0.005, 1, 90]
        ]
        #rep 2 x vals
        # x = [
        #     [20.0, 2.5, 0.005, 1, 90], [5.0, 10.0, 0.005, 1, 90], [5.0, 2.5, 0.02, 1, 90], 
        #     [5.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.001, 1, 90], [5.0, 0.5, 0.005, 1, 90], 
        #     [1.0, 2.5, 0.005, 1, 90], [20.0, 2.5, 0.005, 10, 90], [5.0, 10.0, 0.005, 10, 90], 
        #     [5.0, 2.5, 0.02, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.001, 10, 90], 
        #     [5.0, 0.5, 0.005, 10, 90], [1.0, 2.5, 0.005, 10, 90]
        # ]
        #rep 3 x vals
        # x = [
        #     [20.0, 2.5, 0.005, 1, 90], [5.0, 10.0, 0.005, 1, 90], [5.0, 2.5, 0.02, 1, 90],
        #     [5.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.001, 1, 90], [5.0, 0.5, 0.005, 1, 90], 
        #     [1.0, 2.5, 0.005, 1, 90], [20.0, 2.5, 0.005, 10, 90], [5.0, 10.0, 0.005, 10, 90], 
        #     [5.0, 2.5, 0.02, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.001, 10, 90], 
        #     [5.0, 0.5, 0.005, 10, 90], [1.0, 2.5, 0.005, 10, 90]
        # ]
        #not used for PEM evaluation, so use placeholder
        timecourses = []
        timecourses_err = []
        

    #Choose conditions to include or drop                             
    elif data == 'all echo drop high error':
        drop_labels = [
            [5.0, 10.0, 0.001, 1, 90], [1.0, 2.5, 0.001, 10.0, 90.0], [20.0, 10.0, 0.001, 1.0, 90.0],
            [5.0, 0.5, 0.005, 10.0, 90.0], [20.0, 0.5, 0.005, 1.0, 90.0], [1.0, 2.5, 0.001, 1.0, 90.0], 
            [20.0, 0.5, 0.005, 10.0, 90.0]
        ]
        
        x = []
        exp_data = []
        error = []
        timecourses = []
        timecourses_err = []
        maxVal = 0.6599948235700113
        
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue
            
            elif label in drop_labels:
                continue
        
            else:
               
                x.append(label)
            
                timecourse = list(columnData.iloc[1:])
                timecourse = [i/maxVal for i in timecourse]
                timecourses.append(timecourse)
                exp_data = exp_data + timecourse
        
        for (columnName, columnData) in df_error.iteritems():
            label = list(columnData.iloc[0])
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue
            
            elif label in drop_labels:
                continue
            
            else:
                err = list(columnData.iloc[1:])
                err = [i/maxVal for i in err]
                timecourses_err.append(err)
                error = error + err 

    elif data == 'all echo not in slice drop high error':
        labels1 = [[1.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [20.0, 2.5, 0.005, 1, 90]]
        labels10 = [[1.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [20.0, 2.5, 0.005, 10, 90]]
        labels_T7 = labels1 + labels10
        
        labels1 = [[5.0, 0.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 10.0, 0.005, 1, 90]]
        labels10 = [[5.0, 0.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 10.0, 0.005, 10, 90]]
        labels_RT = labels1 + labels10
        
        labels1 = [[5.0, 2.5, 0.001, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.02, 1, 90]]
        labels10 = [[5.0, 2.5, 0.001, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.02, 10, 90]]
        labels_RNase =labels1 + labels10
        
        labels_slice = labels_T7 + labels_RT + labels_RNase

        drop_labels = [
            [5.0, 10.0, 0.001, 1, 90], [1.0, 2.5, 0.001, 10.0, 90.0], [20.0, 10.0, 0.001, 1.0, 90.0],
            [5.0, 0.5, 0.005, 10.0, 90.0], [20.0, 0.5, 0.005, 1.0, 90.0], [1.0, 2.5, 0.001, 1.0, 90.0], 
            [20.0, 0.5, 0.005, 10.0, 90.0]
        ]

        x = []
        exp_data = []
        error = []
        timecourses = []
        timecourses_err = []
        maxVal = 0.6599948235700113

        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue

            elif label in labels_slice or label in drop_labels:
                continue

            else:
                x.append(label)
                timecourse = list(columnData.iloc[1:])
                timecourse = [i/maxVal for i in timecourse]
                timecourses.append(timecourse)
                exp_data = exp_data + timecourse

        for (columnName, columnData) in df_error.iteritems():
            label = list(columnData.iloc[0])
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue

            elif label in labels_slice or label in drop_labels:
                continue

            else:
                err = list(columnData.iloc[1:])
                err = [i/maxVal for i in err]
                timecourses_err.append(err)
                error = error + err 


    elif data == 'all echo':
        x = []
        exp_data = []
        error = []
        timecourses = []
        timecourses_err = []
        maxVal = 0.6599948235700113
        
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue
        
            else:
                x.append(label)
                timecourse = list(columnData.iloc[1:])
                timecourse = [i/maxVal for i in timecourse]
                timecourses.append(timecourse)
                exp_data = exp_data + timecourse
        
        for (columnName, columnData) in df_error.iteritems():
            label = list(columnData.iloc[0])
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue
            else:
                err = list(columnData.iloc[1:])
                err = [i/maxVal for i in err]
                timecourses_err.append(err)
                error = error + err   
                
    
    elif data == 'slice drop high error':
        labels1 = [[1.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [20.0, 2.5, 0.005, 1, 90]]
        labels10 = [[1.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [20.0, 2.5, 0.005, 10, 90]]
        labels_T7 = labels1 + labels10
        
        labels1 = [[5.0, 0.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 10.0, 0.005, 1, 90]]
        labels10 = [[5.0, 0.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 10.0, 0.005, 10, 90]]
        labels_RT = labels1 + labels10
        
        labels1 = [[5.0, 2.5, 0.001, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.02, 1, 90]]
        labels10 = [[5.0, 2.5, 0.001, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.02, 10, 90]]
        labels_RNase =labels1 + labels10
        
        labels_pre_drop = labels_T7 + labels_RT + labels_RNase
        drop_labels = [
            [5.0, 10.0, 0.001, 1, 90], [1.0, 2.5, 0.001, 10.0, 90.0], [20.0, 10.0, 0.001, 1.0, 90.0],
            [5.0, 0.5, 0.005, 10.0, 90.0], [20.0, 0.5, 0.005, 1.0, 90.0], [1.0, 2.5, 0.001, 1.0, 90.0],
            [20.0, 0.5, 0.005, 10.0, 90.0]
        ]
        labels = []
        for label in labels_pre_drop:
            if label not in drop_labels:
                labels.append(label)
     
        x = []
        exp_data = []
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
            
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue
            elif label in labels:
                x.append(label)
                timecourse = list(columnData.iloc[1:])

                exp_data = exp_data + timecourse
        
        maxVal = max(exp_data)
        exp_data = []
        timecourses = []
        timecourses_err = []
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
            
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue
            elif label in labels:
                timecourse = list(columnData.iloc[1:])
                timecourse = [i/maxVal for i in timecourse]
                
                timecourses.append(timecourse)
                exp_data = exp_data + timecourse           
     
        error = []
        for (columnName, columnData) in df_error.iteritems():
            label = list(columnData.iloc[0])
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue
            
            elif label in labels:
                err = list(columnData.iloc[1:])
                err = [i/maxVal for i in err]
                timecourses_err.append(err)
                error = error + err 


    elif data == 'rep2 all echo drop high error':
        drop_labels = [[20.0, 2.5, 0.02, 10, 90]]        
        x = []
        exp_data = []
        error = []
        timecourses = []
        timecourses_err = []
        maxVal = 2.94995531724754
        
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
            
            if label in drop_labels:
                continue
            else:
               
                x.append(label)
            
                timecourse = list(columnData.iloc[1:])
                timecourse = [i/maxVal for i in timecourse]
                timecourses.append(timecourse)
                exp_data = exp_data + timecourse
        
        for (columnName, columnData) in df_error.iteritems():
            label = list(columnData.iloc[0])
            
            if label in drop_labels:
                continue
            
            else:
                err = list(columnData.iloc[1:])
                err = [i/maxVal for i in err]
                timecourses_err.append(err)
                error = error + err    


    elif data == 'rep2 all echo not in slice drop high error':
        labels1 = [[1.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [20.0, 2.5, 0.005, 1, 90]]
        labels10 = [[1.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [20.0, 2.5, 0.005, 10, 90]]
        labels_T7 = labels1 + labels10
        
        labels1 = [[5.0, 0.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 10.0, 0.005, 1, 90]]
        labels10 = [[5.0, 0.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 10.0, 0.005, 10, 90]]
        labels_RT = labels1 + labels10
        
        labels1 = [[5.0, 2.5, 0.001, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.02, 1, 90]]
        labels10 = [[5.0, 2.5, 0.001, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.02, 10, 90]]
        labels_RNase =labels1 + labels10
        
        labels_slice = labels_T7 + labels_RT + labels_RNase

        drop_labels = [[20.0, 2.5, 0.02, 10, 90]]

        x = []
        exp_data = []
        error = []
        timecourses = []
        timecourses_err = []
        maxVal = 2.94995531724754

        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0]) 
            if label in labels_slice or label in drop_labels:
                continue

            else:
                x.append(label)
                timecourse = list(columnData.iloc[1:])
                timecourse = [i/maxVal for i in timecourse]
                timecourses.append(timecourse)
                exp_data = exp_data + timecourse

        for (columnName, columnData) in df_error.iteritems():
            label = list(columnData.iloc[0])
            if label in labels_slice or label in drop_labels:
                continue

            else:
                err = list(columnData.iloc[1:])
                err = [i/maxVal for i in err]
                timecourses_err.append(err)
                error = error + err 


    elif data == 'rep2 slice drop high error':
        labels1 = [[1.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [20.0, 2.5, 0.005, 1, 90]]
        labels10 = [[1.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [20.0, 2.5, 0.005, 10, 90]]
        labels_T7 = labels1 + labels10
        
        labels1 = [[5.0, 0.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 10.0, 0.005, 1, 90]]
        labels10 = [[5.0, 0.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 10.0, 0.005, 10, 90]]
        labels_RT = labels1 + labels10
        
        labels1 = [[5.0, 2.5, 0.001, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.02, 1, 90]]
        labels10 = [[5.0, 2.5, 0.001, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.02, 10, 90]]
        labels_RNase =labels1 + labels10
        
        labels_pre_drop = labels_T7 + labels_RT + labels_RNase
        drop_labels = [[20.0, 2.5, 0.02, 10, 90]] #not in the slice labels so not relevant here
        labels = []
        for label in labels_pre_drop:
            if label not in drop_labels:
                labels.append(label)
     
        x = []
        exp_data = []
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
        
            if label in labels:
                x.append(label)
                timecourse = list(columnData.iloc[1:])

                exp_data = exp_data + timecourse
        
        maxVal = max(exp_data)
        exp_data = []
        timecourses = []
        timecourses_err = []
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
  
            if label in labels:
                timecourse = list(columnData.iloc[1:])
                timecourse = [i/maxVal for i in timecourse]
                
                timecourses.append(timecourse)
                exp_data = exp_data + timecourse           
     
        error = []
        for (columnName, columnData) in df_error.iteritems():
            label = list(columnData.iloc[0])
            
            if label in labels:
                err = list(columnData.iloc[1:])
                err = [i/maxVal for i in err]
                timecourses_err.append(err)
                error = error + err 

    elif data == 'rep3 all echo drop high error':
        drop_labels = [[5.0, 10.0, 0.02, 10, 90]]        
        x = []
        exp_data = []
        error = []
        timecourses = []
        timecourses_err = []
        maxVal = 1.12314566577301
        
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
            
            if label in drop_labels:
                continue
            else:
               
                x.append(label)
            
                timecourse = list(columnData.iloc[1:])
                timecourse = [i/maxVal for i in timecourse]
                timecourses.append(timecourse)
                exp_data = exp_data + timecourse
        
        for (columnName, columnData) in df_error.iteritems():
            label = list(columnData.iloc[0])
            
            if label in drop_labels:
                continue
            
            else:
                err = list(columnData.iloc[1:])
                err = [i/maxVal for i in err]
                timecourses_err.append(err)
                error = error + err    


    elif data == 'rep3 all echo not in slice drop high error':
        labels1 = [[1.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [20.0, 2.5, 0.005, 1, 90]]
        labels10 = [[1.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [20.0, 2.5, 0.005, 10, 90]]
        labels_T7 = labels1 + labels10
        
        labels1 = [[5.0, 0.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 10.0, 0.005, 1, 90]]
        labels10 = [[5.0, 0.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 10.0, 0.005, 10, 90]]
        labels_RT = labels1 + labels10
        
        labels1 = [[5.0, 2.5, 0.001, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.02, 1, 90]]
        labels10 = [[5.0, 2.5, 0.001, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.02, 10, 90]]
        labels_RNase =labels1 + labels10
        
        labels_slice = labels_T7 + labels_RT + labels_RNase

        drop_labels = [[5.0, 10.0, 0.02, 10, 90]]

        x = []
        exp_data = []
        error = []
        timecourses = []
        timecourses_err = []
        maxVal = 1.12314566577301

        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0]) 
            if label in labels_slice or label in drop_labels:
                continue

            else:
                x.append(label)
                timecourse = list(columnData.iloc[1:])
                timecourse = [i/maxVal for i in timecourse]
                timecourses.append(timecourse)
                exp_data = exp_data + timecourse

        for (columnName, columnData) in df_error.iteritems():
            label = list(columnData.iloc[0])
            if label in labels_slice or label in drop_labels:
                continue

            else:
                err = list(columnData.iloc[1:])
                err = [i/maxVal for i in err]
                timecourses_err.append(err)
                error = error + err 


    elif data == 'rep3 slice drop high error':
        labels1 = [[1.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [20.0, 2.5, 0.005, 1, 90]]
        labels10 = [[1.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [20.0, 2.5, 0.005, 10, 90]]
        labels_T7 = labels1 + labels10
        
        labels1 = [[5.0, 0.5, 0.005, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 10.0, 0.005, 1, 90]]
        labels10 = [[5.0, 0.5, 0.005, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 10.0, 0.005, 10, 90]]
        labels_RT = labels1 + labels10
        
        labels1 = [[5.0, 2.5, 0.001, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.02, 1, 90]]
        labels10 = [[5.0, 2.5, 0.001, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.02, 10, 90]]
        labels_RNase =labels1 + labels10
        
        labels_pre_drop = labels_T7 + labels_RT + labels_RNase
        drop_labels = [[5.0, 10.0, 0.02, 10, 90]] #not in the slice labels so not relevant here
        labels = []
        for label in labels_pre_drop:
            if label not in drop_labels:
                labels.append(label)
     
        x = []
        exp_data = []
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
        
            if label in labels:
                x.append(label)
                timecourse = list(columnData.iloc[1:])

                exp_data = exp_data + timecourse
        
        maxVal = max(exp_data)
        exp_data = []
        timecourses = []
        timecourses_err = []
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
  
            if label in labels:
                timecourse = list(columnData.iloc[1:])
                timecourse = [i/maxVal for i in timecourse]
                
                timecourses.append(timecourse)
                exp_data = exp_data + timecourse           
     
        error = []
        for (columnName, columnData) in df_error.iteritems():
            label = list(columnData.iloc[0])
            
            if label in labels:
                err = list(columnData.iloc[1:])
                err = [i/maxVal for i in err]
                timecourses_err.append(err)
                error = error + err 
            
    
    return x, exp_data, error, timecourses, timecourses_err
