import pandas as pd
import numpy as np

# d = {'index': 0,
#     'p1': [0.000536104732682781], 'p2': [7115.86356173738], 'p3': [0.892292062716542], 'p4': [0.0665475124360066],
#     'p5': [0.114401689315798], 'p6': [42.6583507978646], 'p7': [16.6259078601564]}

d = {'index': 0,
    'p1': [0.466701312025581], 'p2': [10299.0062624589], 'p3': [0.0362576367440956], 'p4': [20.4080261491564],
    'p5': [0.0138079266284545], 'p6': [0], 'p7': [0]}

df_ig = pd.DataFrame(data = d)

df_ig['initial params'] = [[0.466701312025581, 10299.0062624589, 0.0362576367440956, 20.4080261491564,
                            0.0138079266284545, 0.0, 0.0]]

df_ig['free params'] = [['p1', 'p2', 'p3', 'p4', 'p5']]
print(type(df_ig.at[0, 'free params']))

filename = '/Users/kdreyer/Desktop/ig_from_modelA_rep1_opt'

df_ig.to_pickle(filename + '.pkl')
with pd.ExcelWriter(filename + '.xlsx') as writer: 
    df_ig.to_excel(writer, sheet_name=' ')
