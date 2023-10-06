import pandas as pd
import numpy as np

d = {'index': 0,
    'p1': [0.000536104732682781], 'p2': [7115.86356173738], 'p3': [0.892292062716542], 'p4': [0.0665475124360066],
    'p5': [0.114401689315798], 'p6': [42.6583507978646], 'p7': [16.6259078601564]}

df_ig = pd.DataFrame(data = d)

df_ig['initial params'] = [[0.000536104732682781, 7115.86356173738, 0.892292062716542, 0.0665475124360066, 
                            0.114401689315798, 42.6583507978646, 16.6259078601564]]

df_ig['free params'] = [['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']]
print(type(df_ig.at[0, 'free params']))

filename = '/Users/kdreyer/Desktop/ig_from_modelB_rep1_opt'

df_ig.to_pickle(filename + '.pkl')
with pd.ExcelWriter(filename + '.xlsx') as writer: 
    df_ig.to_excel(writer, sheet_name=' ')
