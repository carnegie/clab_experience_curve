import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib, analysisFunctions, plottingFunctions, os
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
                                 'Max trainingt horizon',
                                 'Max forecast horizon',
                                 'Tech'])

dfObsErrAll = dfObsErr.copy()

dfObsErr = dfObsErr.loc[dfObsErr['Training horizon']>=1]\
                     .loc[dfObsErr['Max forecast horizon']>=1]


# fig, ax = plottingFunctions.plotObsPredErr(dfObsErr)

fig, ax = plottingFunctions.plotErr(dfObsErr)

if not os.path.exists('figs' + os.path.sep + 'costError'):
    os.makedirs('figs' + os.path.sep + 'costError')
fig.savefig('figs' + os.path.sep + 'costError' + \
            os.path.sep + 'ObsPredErr.png')
fig.savefig('figs' + os.path.sep + 'costError' + \
            os.path.sep + 'ObsPredErr.eps')

# for supplementary material
fig, ax = plottingFunctions.plotErrTrFor(dfObsErrAll)  

if not os.path.exists('figs' + os.path.sep + 'SupplementaryFigures'):
    os.makedirs('figs' + os.path.sep + 'SupplementaryFigures')
fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
            os.path.sep + 'ErrorTrainingForecast.png')

fig, ax = plottingFunctions.plotErrorTech(dfObsErr)
fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
            os.path.sep + 'ErrorByTech.png')

# repeat removing technology under exam when computing average slope
# compute all errors combination for all technologies
dfObsErr = analysisFunctions.computeAllErrors(df, removeTechExamined=True)

# transform list to dataframe
dfObsErr = pd.DataFrame(dfObsErr, 
                        columns=['Training horizon',
                                 'Forecast horizon',
                                 'Observation',
                                 'Forecast (Tech)',
                                 'Forecast (Avg)',
                                 'Error (Tech)',
                                 'Error (Avg)',
                                 'Max trainingt horizon',
                                 'Max forecast horizon',
                                 'Tech'])

dfObsErrAll = dfObsErr.copy()

dfObsErr = dfObsErr.loc[dfObsErr['Training horizon']>=1]\
                     .loc[dfObsErr['Max forecast horizon']>=1]


# fig, ax = plottingFunctions.plotObsPredErr(dfObsErr)

fig, ax = plottingFunctions.plotErr(dfObsErr)

fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
            os.path.sep + 'ObsPredErr_removeTechExamined.png')

# for supplementary material
fig, ax = plottingFunctions.plotErrTrFor(dfObsErrAll)  

fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
            os.path.sep + 'ErrorTrainingForecast_removeTechExamined.png')

fig, ax = plottingFunctions.plotErrorTech(dfObsErr)
fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
            os.path.sep + 'ErrorByTech_removeTechExamined.png')

plt.show()