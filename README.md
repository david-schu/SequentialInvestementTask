# SequentialInvestementTask

code of the Master's thesis by David Schultheiß

## Data parsing

The data for each subject is parsed from raw data files into pandas DataFrames with load_data.py in the helperFiles folder. Hereby, market predictions generated with the GaussianProcess.py file. To adjust the Gaussian process details, you can change the kernel parameters, or the Gaussian Process itself. 

## Jupyter Notebooks

Figure of the thesis are created in four juypter notebooks - general data anaylsis, regression analysis, mechanistic model anayslis, neural data anaylsis. 

## Generelized mechanistic model

mechnisticModel contains code for fitting a mechnistic model given subject data. the NLL-function and fitting function of the model are contained in model.py. fit.py is an example of calling the fitting and nll function. The jupyter notebook guides through the fitting of models with different complexities.
