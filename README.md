# COVID_Dx_GAMES

This repository contains supporting code for the unpublished article: Jung, JK et al. Model-driven characterization of CRISPR-based point-of-use pathogen diagnostics. This work utilizes the [Dray, et al. GAMES conceptual workflow](https://pubs.acs.org/doi/10.1021/acssynbio.1c00528), and is currently using the [GAMES v1.0.1 framework](https://github.com/leonardlab/GAMES). 
See details below regarding how to setup the code and conduct a test run with Model D to produce plots of the modeling objectives with the calibrated parameters for data set 2 (Figure 4 in the article). <br />

## Overview of main executable files
- Test_COVID_Dx.py solves the ODEs for a single parameter set
- Run_COVID_Dx.py is used to execute one or more modules of the GAMES workflow (as defined in Settings_COVID_Dx.py)<br />
- Analysis_PEM_Evaluation.py generates plots of the PEM evaluation for each dataset, to compare to the PEM evaluation criterion <br />
- SensitivityAnalysis.py runs a sensitivity analysis based on the F_max and t_1/2 values and plots the results
- Sensitivity_Analysis_chi2.py runs a sensitivity analysis based on the chi2 value and plots the results <br />
See required updates [below](#4-update-the-required-paths-in-the-code-files-to-make-them-executable-on-your-machine-) to make these files executable on your machine. <br />

## Notes on other files
- Settings_COVID_Dx.py contains the main settings for each run of the code, including folder name, definition of free parameters, choice of model, choice of training data, run type, and parameter estimation method hyperparameters. Each time the code is run, this file needs to be updated with the appropriate settings. <br />
- gen_mechanism.py contains the model ODEs <br />

## Setup on your device <br />
### macOS: <br />
#### 1. install dependencies with your preferred package manager (I recommend conda--i.e. anaconda or miniconda-- or pip): <br />
- dependencies: <br />
python >=3.7 <br />
lmfit >=0.9.14 <br />
matplotlib >=3.1.3 <br />
*note that to create the plots in Sensitivity_Analysis.py and Sensitivity_Analysis_chi2.py, matplotlib >=3.7.0 <br />
numpy >=1.18.1 <br />
pandas >=1.0.5 <br />
salib >=1.3.8 <br />
scipy >=1.4 <br />
- install with conda: <br />
conda install <package_name> <br /> 
- install with pip: <br /> 
pip install <package_name> <br />

#### 2. clone the repo with the http or ssh url <br />
- using the terminal (on macOS), cd into the directory in which you would like the repo to be cloned. To clone the repo into a directory with the same name as the repo use the following command (for http*):<br />
**git clone https://github.com/leonardlab/COVID_Dx_GAMES.git** <br />
##### *to clone with ssh, use the ssh url instead of the http url. <br />

#### 3. make results directory: <br />
in the directory of your local COVID_Dx_GAMES repo, make a new directory named "Results". Files from each run of the code will be saved there. <br />

#### 4. update the required paths in the code files to make them executable on your machine: <br />
- Change ***results_folder_path*** to correct absolute path to results folder in Saving.py <br />
- Change ***paper.mplstyle.py path*** (in plt.style.use()) to correct absolute path in: <br />
-- Analysis_PEM_Evaluation.py <br />
-- Analysis_Plots.py <br />
-- Run_COVID_Dx.py <br />
-- Sensitivity_Analysis_chi2.py <br />
-- SensitivityAnalysis.py <br />
-- Test_COVID_Dx.py <br />
- Change ***df_data*** and ***df_error*** paths to correct absolute paths in: <br />
-- Analysis_plots.py <br />
-- DefineExpData_COVID_Dx.py <br />
- Change paths to all PEM evaluation results to correct absolute paths in Analysis_PEM_Evaluation.py <br />
- Change path to PEM evaluation data set to correct absolute path in DefineExpData_COVID_Dx.py (only necessary if running PEM evaluation parameter estimation) <br />
- Change path to PEM evaluation global search results to correct absolute path in Run_COVID_Dx.py (only necessary if running PEM evaluation parameter estimation) <br />
- Change paths to experimental data excel files to correct absolute paths in process_raw_training_data.py <br />

### Note for running code on windows: <br />
#### (information based on GAMESv1.0.0 readme): <br />
The parallelization component of this code was written using Python’s multiprocessing package, which has a different implementation on mac vs. windows OS. This code was run and tested on macOS and is *not* currently set up to support any other OS. <br />
However, GAMES v1.0.1 code was successfully run on Linux, and on Windows *without* parallelization. <br /> If you are a Windows user, we recommend setting parallelization = ‘no’ on line XXX in Run.py to use a version of the code that does not use parallelization. However, this particular model is not well suited to run without parallelization and there is a chance it is not feasible for the iterative, parameter estimation code.
 <br />

## Test running the code: <br />
The code is currently set up to run a test for model D (the final model) and data set 2, with the calibrated parameter set from the low tolerance run, using the low error tolerances in the ODE solver. This run will solve the model ODEs and plot each of the 6 modeling objectives and a parity plot between the experimental data and simulation values. <br />
- Update folder_name in Settings_COVID_Dx.py to the desired folder name for the run, which is where the above plots will be saved (this needs to be updated each time the code in run or the previous results will be overwritten) <br />
- Run Test_COVID_Dx.py to execute the test run <br />

## Running other modules <br />
To run other modules (e.g. parameter estimation or generate PEM evaluation data), change conditions_dictionary["run_type"] in Settings_COVID_Dx.py <br />

## Updating the code to run with other model versions or data sets <br />
The code is currently set up to run Model D (the final model) with data set 2. <br />
- To run the code with Model A, B, or C:
-- Change modelID in Settings_COVID_Dx.py, and uncomment p_labels_all (line 76)
-- Update solveForOpt() in Run_COVID_Dx.py: remove p8 and p9 from function arguments and list of p values (line 347), and update result_row and result_row_labels to grab correct indices: [:22] for both
- To run the code with other experimental data sets, change conditions_dictionary["data"] in Settings_COVID_Dx.py to appropriate data set

## Updating the code to run with high solver error tolerances
As we note in the manuscript, the default error tolerances for the ODE solver resulted in negative concentration values for some model states in some simulations. The code is currently set up to run with the low error tolerances. To update the code to run with the default error tolerances:
- Remove the arguments atol = self.abs_tol and rtol = self.rel_tol in the solver function call in gen_mechanism.py (currently set in line 127)
 
