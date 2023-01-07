#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kateldray
"""
#Package imports
import pandas as pd
import json

def defineExp(data, model, k_CV, k_PEM_evaluation):
    ''' 
    Purpose: Import the experimental data from a file and
        structure the data to be compatible with downstream LMFIT (optimization) code.
    
    Inputs:
        data: a string defining the data identity
        
        model: a string defining the model identity
        
        k_CV: an integer defining the value of k_CV for this run 
            (only used when data = 'cross-validation train' or data = 'cross-validation test')
            
        k_PEM_evaluation: an integer defining the value of k_PEM_evaluation for this run 
            (only used when data = 'PEM evaluation')
      
    Outputs:
         x: a list of lists defining the component doses for each condition. Each list has 5 items and there 
             are the same number of lists as number of conditions. Each individual list holds the component 
             doses in the following order: [x_T7, x_RT, x_RNase, x_inputRNA, x_Cas13]. Note that the component 
             dose for x_Cas13 represents the Cas13 component dose, not the Cas13a-gRNA component dose. The amount 
             of gRNA added is exactly half the amount of Cas13, so it is assumed that [Cas13a-gRNA] = [Cas13]/2. 
             This operation (dividing [Cas13] by 2) is completed in Solvers.py each time the equations are solved.

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
    df_data = pd.read_pickle('/Users/kdreyer/Desktop/Github/COVID_Dx_GAMES/PROCESSED DATA EXP.pkl')
    df_error = pd.read_pickle('/Users/kdreyer/Desktop/Github/COVID_Dx_GAMES/PROCESSED DATA ERR.pkl')

    with open("/Users/kdreyer/Desktop/Github/COVID_Dx_GAMES/x_CV_train2.json", "r") as fp:
        x_CV_train = json.load(fp)
        
    with open("/Users/kdreyer/Desktop/Github/COVID_Dx_GAMES/x_CV_test2.json", "r") as fp:
        x_CV_test = json.load(fp)

    # if data == 'PEM evaluation':
    #     #Import df for appropriate model and k_PEM_evaluation
    #     filename = '/Users/kdreyer/Desktop/Github/COVID_Dx_GAMES/PEM evaluation data/PEM EVALUATION DATA NOISE ' + model + '.xlsx'
    #     df = pd.read_excel(filename, sheet_name = str(k_PEM_evaluation))
        
    #     exp_data = []
    #     i = 0
    #     for column in df:
    #         if i != 0: #skip column 0 (indicies)
    #             vals = list(df[column])
    #             exp_data = exp_data + vals
    #         i += 1
    #     exp_data = [i/max(exp_data) for i in exp_data]
       
    #     #error not used for PEM evaluation paramter estimation runs, so use placeholder
    #     error = [1] * len(exp_data)
        
    #     #same x as 'slice drop high error'
    #     x = [[20.0, 2.5, 0.005, 10, 90], [5.0, 10.0, 0.005, 10, 90], [5.0, 2.5, 0.02, 10, 90], [5.0, 2.5, 0.005, 10, 90], [5.0, 2.5, 0.001, 10, 90], [1.0, 2.5, 0.005, 10, 90], [20.0, 2.5, 0.005, 1, 90], [5.0, 10.0, 0.005, 1, 90], [5.0, 2.5, 0.02, 1, 90], [5.0, 2.5, 0.005, 1, 90], [5.0, 2.5, 0.001, 1, 90], [5.0, 0.5, 0.005, 1, 90], [1.0, 2.5, 0.005, 1, 90]]
        
    #     #not used for PEM evaluation, so use placeholder
    #     timecourses = []
    #     timecourses_err = []
        

    #Choose conditions to include or drop
    if data == 'all echo without low iCas13 or 0 vRNA':
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
                if label[3] != 0.0:
                    if label[4] != 4.5:
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
                if label[3] != 0.0:
                    if label[4] != 4.5:
                        err = list(columnData.iloc[1:])
                        err = [i/maxVal for i in err]
                        timecourses_err.append(err)
                        error = error + err
                        
    elif data == 'all echo without low iCas13 or 0 vRNA and drop high error':
        x = []
        exp_data = []
        error = []
        timecourses = []
        timecourses_err = []
        maxVal = 0.6599948235700113
        
        drop_labels = [[5.0, 10.0, 0.001, 1, 90], [1.0, 2.5, 0.001, 10.0, 90.0], [20.0, 10.0, 0.001, 1.0, 90.0], [5.0, 0.5, 0.005, 10.0, 90.0], [20.0, 0.5, 0.005, 1.0, 90.0], [1.0, 2.5, 0.001, 1.0, 90.0], [20.0, 0.5, 0.005, 10.0, 90.0]]
        
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue
        
            elif label in drop_labels:
                continue
        
            else:
                if label[3] != 0.0:
                    if label[4] != 4.5:
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
                if label[3] != 0.0:
                    if label[4] != 4.5:
                        err = list(columnData.iloc[1:])
                        err = [i/maxVal for i in err]
                        timecourses_err.append(err)
                        error = error + err 
                        
    elif data == 'cross-validation train':
        
        labels = x_CV_train[k_CV - 1]
       
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
                
    elif data == 'cross-validation test':
    
        labels = x_CV_test[k_CV - 1]
    
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
            
        
    
    elif data == 'cross-validation data partitioning':
        x = []
        exp_data = []
        error = []
        timecourses = []
        timecourses_err = []
        maxVal = 0.6599948235700113
        
        drop_labels = [[5.0, 10.0, 0.001, 1, 90], [1.0, 2.5, 0.001, 10.0, 90.0], [20.0, 10.0, 0.001, 1.0, 90.0], [5.0, 0.5, 0.005, 10.0, 90.0], [20.0, 0.5, 0.005, 1.0, 90.0], [1.0, 2.5, 0.001, 1.0, 90.0], [20.0, 0.5, 0.005, 10.0, 90.0]]
        
        for (columnName, columnData) in df_data.iteritems():
            label = list(columnData.iloc[0])
            if label == [20.0, 10.0, 0.001, 10.0, 90.0] or label == [20.0, 10.0, 0.001, 0.0, 90.0] or label == [5.0, 2.5, 0.001, 0.0, 90.0]:
                continue
        
            elif label in drop_labels:
                continue
        
            else:
                if label[3] != 0.0:
                    if label[4] != 4.5:
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
                if label[3] != 0.0:
                    if label[4] != 4.5:
                        err = list(columnData.iloc[1:])
                        err = [i/maxVal for i in err]
                        timecourses_err.append(err)
                        error = error + err          
                        
    elif data == 'all echo drop high error':
        drop_labels = [[5.0, 10.0, 0.001, 1, 90], [1.0, 2.5, 0.001, 10.0, 90.0], [20.0, 10.0, 0.001, 1.0, 90.0], [5.0, 0.5, 0.005, 10.0, 90.0], [20.0, 0.5, 0.005, 1.0, 90.0], [1.0, 2.5, 0.001, 1.0, 90.0], [20.0, 0.5, 0.005, 10.0, 90.0]]
        
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
        drop_labels = [[5.0, 10.0, 0.001, 1, 90], [1.0, 2.5, 0.001, 10.0, 90.0], [20.0, 10.0, 0.001, 1.0, 90.0], [5.0, 0.5, 0.005, 10.0, 90.0], [20.0, 0.5, 0.005, 1.0, 90.0], [1.0, 2.5, 0.001, 1.0, 90.0], [20.0, 0.5, 0.005, 10.0, 90.0]]
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
                
   
    elif data == 'slice drop high error add optimal':
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
        labels_pre_drop.append([5.0, 10.0, 0.02, 1, 90]) #optimal condition
       
        

        drop_labels = [[5.0, 10.0, 0.001, 1, 90], [1.0, 2.5, 0.001, 10.0, 90.0], [20.0, 10.0, 0.001, 1.0, 90.0], [5.0, 0.5, 0.005, 10.0, 90.0], [20.0, 0.5, 0.005, 1.0, 90.0], [1.0, 2.5, 0.001, 1.0, 90.0], [20.0, 0.5, 0.005, 10.0, 90.0]]
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
                

            
    
    return x, exp_data, error, timecourses, timecourses_err
