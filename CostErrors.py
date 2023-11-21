import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib, analysisFunctions, plottingFunctions, scipy
import seaborn as sns

matplotlib.rc('savefig', dpi=300)
matplotlib.rc('font', 
              **{'family':'sans-serif',
                 'sans-serif':'Helvetica'})
matplotlib.rcParams['pdf.fonttype'] = 42
sns.set_style('whitegrid')
sns.set_palette('colorblind')
sns.set_context('talk')

# read data
df = pd.read_csv('ExpCurves.csv')

# compute all errors combination for all technologies
dfObsErr = analysisFunctions.computeAllErrors(df)

# transform list to dataframe
dfObsErr = pd.DataFrame(dfObsErr, 
                        columns=['Training horizon',
                                 'Forecast horizon',
                                 'Observation',
                                 'Forecast (Tech)',
                                 'Forecast (Avg)',
                                 'Error (Tech)',
                                 'Error (Avg)',
                                 'Point',
                                 'Tech'])

dfObsErrAll = dfObsErr.copy()

# select only technologies for which at least 
# one order of magnitude of data is available
# for both training and forecast horizons
dfObsErr = dfObsErr.loc[\
    dfObsErr['Tech'].isin(\
        dfObsErr.copy().loc[\
            (dfObsErr['Training horizon']>=1) &\
            (dfObsErr['Forecast horizon']>=1),'Tech'].values)].copy()

for t in dfObsErr['Tech'].unique():
    dfObsErr.loc[dfObsErr['Tech']==t] = \
        dfObsErr.loc[(dfObsErr['Tech']==t) & \
        (dfObsErr.loc[dfObsErr['Tech']==t,'Point'].isin(\
            dfObsErr.loc[(dfObsErr['Tech']==t) & \
                (dfObsErr['Training horizon']>=1) &\
                (dfObsErr['Forecast horizon']>=1) ,'Point'].values))]

# select only data for which 
# at least one order of magnitude
# has been used for training
dfObsErr = dfObsErr.loc[\
    dfObsErr['Training horizon']>=1]

fig, ax = plottingFunctions.plotObsPredErr(dfObsErr)


# for supplementary material
fig, ax = plottingFunctions.plotErrTrFor(dfObsErrAll)  

fig, ax = plottingFunctions.plotErrorTech(dfObsErr)

plt.show()