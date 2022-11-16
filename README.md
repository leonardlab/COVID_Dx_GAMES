# COVID_Dx_GAMES

This code is associated with Chapter 3 of Kate Dray's PhD thesis (Northwestern University, Chemical Engineering) and follows the GAMES conceptual workflow (Dray et al. (2022). ACS Synthetic Biology). Since the completeion of this code, the GAMES code has been significantly refactored; the code presented here uses v1.1 of the GAMES software package (along with Python 3.7). Other package requirements can be found in the documentatation for v1.1 on the GAMES code (can be found on the Leonard Lab GitHub account).

Run_COVID_Dx.py and Test_COVID_Dx.py are the main executable files. Test_COVID_Dx.py solves the ODEs for a single parameter set, while  Run_COVID_Dx.py is used to execute one or more modules of the GAMES workflow (as defined in Settings_COVID_Dx.py).

Analysis_Cross_Validation.py and Analysis_PEM_Evaluation.py are also executable and can be used to analyze cross validation results and PEM evaluation results (requires input of results folder names to run - results folders are currently set to the final results used in the paper). 

SensitivityAnalysis.py is also executable and can be uused to run sensitivity analysis and plot results. 

The ODEs are defined in gen_mechanism.py. 

Settings, including parameter estimation method hyperparameters, definition of free parameters, and choice of model and training data, are set in Settings_COVID_Dx.py). 