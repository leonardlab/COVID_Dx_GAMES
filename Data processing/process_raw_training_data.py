#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:27:58 2021

@author: kate
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def processData(path, originalFilename, sheetName, vRNA_dose, Cas13_dose):
    stri = path + originalFilename
    x1 =pd.ExcelFile(stri, engine='openpyxl')
    df = x1.parse(sheetName)
    ndf = df.copy()
    # print(df)
    for col in df.columns[1:28]:
        if pd.isna(df[col][0]) == False:
            T7_value = float(str(df[col][0])[8:10])
        if pd.isna(df[col][1]) == False:
            RT_value = float(str(df[col][1])[10:12])

            if RT_value == 2.0:
                RT_value = 2.5
            elif RT_value == 0.0:
                RT_value = 0.5
        # print(float(str(df.loc[2, col])[8:12]))
        ndf.at[2, col] = [T7_value, RT_value, float(str(df.loc[2, col])[8:12])]

    # # #there is a new sheet at column 28
    
    for col in df.columns[29:56]:
        if pd.isna(df[col][0]) == False:
            T7_value = float(str(df[col][0])[8:10])
        if pd.isna(df[col][1]) == False:
            RT_value = float(str(df[col][1])[10:12])
            
            if RT_value == 2.0:
                RT_value = 2.5
            elif RT_value == 0.0:
                RT_value = 0.5
        # print(float(str(df.loc[2, col])[8:12]))
        ndf.at[2, col] = [T7_value, RT_value, float(str(df.loc[2, col])[8:12])]

    ndf.drop([0, 1],0,inplace=True) #removing the first two rows
    ndf.reset_index(inplace=True, drop=True)
    
    num_col = len(list(ndf.columns))
    while num_col > 56:
        drop_col = ndf.columns[-1]
        ndf.drop(drop_col, axis=1, inplace=True)
        num_col = len(list(ndf.columns))
    # print(ndf)
    # print(len(list(ndf.columns)))
    
    num_row = len(list(ndf.index))
    while num_row > 62:
        drop_row = ndf.index[-1]
        ndf.drop(drop_row, axis=0, inplace=True)
        ndf.reset_index(inplace=True, drop=True)
        num_row = len(list(ndf.index))
    # print(len(list(ndf.index)))

    #  #add vRNA dose to header
    for column in ndf:
        if column != 'Unnamed: 0':
            # print(ndf.loc[0, column])
            updatedVals = list(ndf.loc[0, column]) + [vRNA_dose, Cas13_dose]
            # print(updatedVals)
            ndf.at[0, column] = updatedVals
    
    dfExp = ndf.iloc[:, 1:28]
    dfErr = ndf.iloc[:, 29:]
    
    return dfExp, dfErr


def compile_dataset(path, fnames, sheet_names, vRNA_doses, Cas13_doses, out_fname):
    # compile data for excel sheet 1: first Cas13 dose
    dfExp_list1 = []
    dfErr_list1 = []
    for i in range(0, 3):
        dfExp1, dfErr1 = processData(path, fnames[0], sheet_names[i], vRNA_doses[i], Cas13_doses[0])
        dfExp_list1.append(dfExp1)
        dfErr_list1.append(dfErr1)

    dfExp_all1 = pd.concat(dfExp_list1, axis = 1)
    dfErr_all1 = pd.concat(dfErr_list1, axis = 1)

    # compile data for excel sheet 2: second Cas13 dose
    dfExp_list2 = []
    dfErr_list2 = []
    for i in range(0, 3):
        dfExp2, dfErr2 = processData(path, fnames[1], sheetNames[i], vRNA_doses[i], Cas13_doses[1])
        dfExp_list2.append(dfExp2)
        dfErr_list2.append(dfErr2)
        
    dfExp_all2 = pd.concat(dfExp_list2, axis = 1)
    dfErr_all2 = pd.concat(dfErr_list2, axis = 1)

    dfExp_all = pd.concat([dfExp_all1, dfExp_all2], axis = 1)
    dfErr_all = pd.concat([dfErr_all1, dfErr_all2], axis = 1)

    dfExp_all.to_pickle(path + out_fname + 'EXP.pkl')
    dfErr_all.to_pickle(path + out_fname + 'ERR.pkl')


#args for all datasets
path = '/Users/kdreyer/Desktop/Github/COVID_Dx_GAMES/'
sheetNames = ['Processed-0fM_FITC', 'Processed-1fM_FITC', 'Processed-10fM_FITC']
Cas13_doses = [90, 4.5] #nM
vRNA_doses = [0, 1, 10] #fm

#original data (rep1)
fname1_p1 = 'Data processing/202010501_coviddxscreen_processed_FITC.xlsx'
fname1_p2 = 'Data processing/202010506_coviddxscreen_processced_FITC_part2.xlsx'
fnames1 = [fname1_p1, fname1_p2]
saveFilename1 = 'PROCESSED_DATA_rep1_fix_'
# compile_dataset(path, fnames1, sheetNames, vRNA_doses, Cas13_doses, saveFilename1)

#repeat data (rep2)
fname2_p1 = 'Data processing/20221115_covidscreen_processed_FITC_part1.xlsx'
fname2_p2 = 'Data processing/20221117_covidscreen_processed_FITC_part2.xlsx'
fnames2 = [fname2_p1, fname2_p2]
saveFilename2 = 'PROCESSED_DATA_rep2_'
# compile_dataset(path, fnames2, sheetNames, vRNA_doses, Cas13_doses, saveFilename2)

#repeat data pt2 (rep3)
fname3_p1 = 'Data processing/20221209_covidsscreen_processed_FITC_part1.xlsx'
fname3_p2 = 'Data processing/20221213_covidscreen_processed_FITC_part2.xlsx'
fnames3 = [fname3_p1, fname3_p2]
saveFilename3 = 'PROCESSED_DATA_rep3_'
compile_dataset(path, fnames3, sheetNames, vRNA_doses, Cas13_doses, saveFilename3)







