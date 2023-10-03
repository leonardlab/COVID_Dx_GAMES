import numpy as np
import pandas as pd
from DefineExpData_COVID_Dx import defineExp

#for both datasets
model = 'model D'
k_CV = ''
k_PEM_evaluation = ''

#rep 1 data
data_rep1 = 'slice drop high error'
_, rep1_exp, rep1_err, _, _ = defineExp(data_rep1, k_PEM_evaluation)
print('rep1 mean error', np.mean(rep1_err))

#rep 2 data
data_rep2 = 'rep2 slice drop high error'
_, rep2_exp, rep2_err, _, _ = defineExp(data_rep2, k_PEM_evaluation)
print('rep2 mean error', np.mean(rep2_err))

#rep 3 data
data_rep3 = 'rep3 slice drop high error'
_, rep3_exp, rep3_err, _, _ = defineExp(data_rep3, k_PEM_evaluation)
# print(rep3_err)
print('rep3 mean error', np.mean(rep3_err))