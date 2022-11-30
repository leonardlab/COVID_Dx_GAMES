#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:27:58 2021

@author: kate
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def processData(originalFilename, sheetName, saveFilename, vRNA_dose, Cas13_dose):
    stri = originalFilename
    x1 =pd.ExcelFile(stri)
    df = x1.parse(sheetName)
    ndf = df.copy()
    
    for col in df.columns[1:28]:
        if pd.isna(df[col][0]) == False:
            T7_value = float(str(df[col][0])[8:10])
        if pd.isna(df[col][1]) == False:
            RT_value = float(str(df[col][1])[10:12])
           
            if RT_value == 2.0:
                RT_value = 2.5
            elif RT_value == 0.0:
                RT_value = 0.5
         
        #print(float(str(df[col][2])[8:10]))
        ndf[col][2] = [T7_value, RT_value, float(str(df[col][2])[8:12])]
        
    #there is a new sheet at column 28
    
    for col in df.columns[29:]:
        if pd.isna(df[col][0]) == False:
            T7_value = float(str(df[col][0])[8:10])
        if pd.isna(df[col][1]) == False:
            RT_value = float(str(df[col][1])[10:12])
            
            if RT_value == 2.0:
                RT_value = 2.5
            elif RT_value == 0.0:
                RT_value = 0.5
        #print(float(str(df[col][2])[8:10]))
        ndf[col][2] = [T7_value, RT_value, float(str(df[col][2])[8:12])]
        
    ndf.drop([0, 1],0,inplace=True) #removing the first two rows
    
     #add vRNA dose to header
    for column in ndf:
        if column != 'Unnamed: 0':
            #print(ndf[column].iloc[0])
            updatedVals = list(ndf[column].iloc[0]) + [vRNA_dose] + [Cas13_dose]
            ndf[column].iloc[0] =updatedVals
    
    
    dfExp = ndf.iloc[:, 1:28]
    dfErr = ndf.iloc[:, 29:]
    
    return dfExp, dfErr

#First sheet - Cas13 = 90
originalFilename = '202010501_coviddxscreen_processed_FITC.xlsx'
saveFilename = 'PROCESSED DATA'
Cas13_dose = 90 #nM

sheetNames = ['10fM_FITC', '1fM_FITC', '0fM_FITC']
vRNA_doses = [10, 1, 0] #fm
dfExp_list1 = []
dfErr_list1 = []
for i in range(0, 3):
    dfExp1, dfErr1 = processData(originalFilename, sheetNames[i], saveFilename, vRNA_doses[i], Cas13_dose)
    dfExp_list1.append(dfExp1)
    dfErr_list1.append(dfErr1)
    
dfExp_all1 = pd.concat(dfExp_list1, axis = 1)
dfErr_all1 = pd.concat(dfErr_list1, axis = 1)
#dfExp_all1.to_pickle('./' + saveFilename + 'exp.pkl')
#dfErr_all1.to_pickle('./' + saveFilename + 'err.pkl')

#Second sheet - Cas13 = 4.5
Cas13_dose = 4.5 #nM
originalFilename = '202010506_coviddxscreen_processced_FITC_part2.xlsx'
sheetNames = ['Processed-0fM_FITC', 'Processed-1fM_FITC', 'Processed-10fM_FITC']
dfExp_list2 = []
dfErr_list2 = []
for i in range(0, 3):
    dfExp2, dfErr2 = processData(originalFilename, sheetNames[i], saveFilename, vRNA_doses[i], Cas13_dose)
    dfExp_list2.append(dfExp2)
    dfErr_list2.append(dfErr2)
    
dfExp_all2 = pd.concat(dfExp_list2, axis = 1)
dfErr_all2 = pd.concat(dfErr_list2, axis = 1)

dfExp_all = pd.concat([dfExp_all1, dfExp_all2], axis = 1)
dfErr_all = pd.concat([dfErr_all1, dfErr_all2], axis = 1)

dfExp_all.to_pickle('./' + saveFilename + ' EXP.pkl')
dfErr_all.to_pickle('./' + saveFilename + ' ERR.pkl')

with pd.ExcelWriter('df exp' + '.xlsx') as writer:  # doctest: +SKIP
    dfExp_all.to_excel(writer, sheet_name=' ')
    
with pd.ExcelWriter('df err' + '.xlsx') as writer:  # doctest: +SKIP
    dfErr_all.to_excel(writer, sheet_name=' ')










