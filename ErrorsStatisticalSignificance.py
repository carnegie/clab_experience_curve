import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib, scipy, analysisFunctions, plottingFunctions, os

sns.set_style('whitegrid')
sns.set_palette('colorblind')
sns.set_context('talk')
matplotlib.rc('savefig', dpi=300)
matplotlib.rc('font',
                **{'family':'sans-serif',
                     'sans-serif':'Helvetica'})
matplotlib.rcParams['pdf.fonttype'] = 42


# read data
df = pd.read_csv('ExpCurves.csv')

# compute all possible combination of errors
dfObsErr = analysisFunctions.computeAllErrors(df)

# transform list to dataframe
dfObsErrAll = pd.DataFrame(dfObsErr, columns=['Training horizon',
                                           'Forecast horizon',
                                           'Observation',
                                           'Forecast (Tech)',
                                           'Forecast (Avg)',
                                           'Error (Tech)',
                                           'Error (Avg)',
                                           'Max training horizon',
                                           'Max forecast horizon',
                                           'Tech'])

# repeat test for different orders of magnitude
rangeslo = [0.0,0.5,1.0]
rangesmed = [0.5,1.0,2.0]
rangeshi = [1.0,1.5,3.0]

# create lists to store results
techTests, allTests = [], []

# iterate over training and forecast ranges
for r1 in enumerate(rangesmed):
    for r2 in enumerate(rangesmed):

        # select data for which:
        # - training horizon is at least the required value
        # - max forecast horizon of the point is at least the required value
        # - forecast horizon is above defined lower range
        # - forecast horizon is below defined higher range
        sel = dfObsErrAll\
            .loc[dfObsErrAll['Training horizon'] >= \
                rangesmed[r1[0]]]\
            .loc[dfObsErrAll['Max forecast horizon'] >= \
                 rangesmed[r2[0]]]\
            .loc[dfObsErrAll['Forecast horizon'] >= \
                 rangeslo[r2[0]]]\
            .loc[dfObsErrAll['Forecast horizon'] < \
                 rangeshi[r2[0]]].copy()

        # create lists to store errors
        errTech, errAvg = [], []

        # iterate over technologies
        for t in sel['Tech'].unique():

            # select data for technology
            errTech.append(sel.loc[\
                sel['Tech']==t, 'Error (Tech)'].values)
            errAvg.append(sel.loc[\
                sel['Tech']==t, 'Error (Avg)'].values)

            # for each technology append:
            # - the technology
            # - the training horizon
            # - the forecast horizon
            # - the p value of the paired t-test
            # - the p value of the mann whitney test
            # - the RMSE of the tech using tech specific method
            # - the RMSE of the tech using average method
            techTests.append([t, r1[1], r2[1],
                scipy.stats.ttest_rel(\
                    errTech[-1], errAvg[-1])[1],
                scipy.stats.mannwhitneyu(  
                    errTech[-1], errAvg[-1])[1],
                np.mean([x**2 for x in errTech[-1]])**0.5,  
                np.mean([x**2 for x in errAvg[-1]])**0.5])
            
        # append:
        # - the training horizon
        # - the forecast horizon
        # - the p value of the paired t-test
        # - the p value of the mann whitney test
        allTests.append([r1[1], r2[1], 
            scipy.stats.ttest_rel(\
                [np.mean([x**2 for x in a])**0.5 for a in errTech],
                [np.mean([x**2 for x in a])**0.5 for a in errAvg])[1],
            scipy.stats.wilcoxon(\
                [np.mean([x**2 for x in a])**0.5 for a in errTech],
                [np.mean([x**2 for x in a])**0.5 for a in errAvg])[1]])

# transform lists to dataframes   
techTests = pd.DataFrame(techTests, columns=['Tech', 'Training horizon', 
    'Forecast horizon', 'Paired t-test', 'Wilcoxon signed-ranks test',
    'Mean Error (Tech)', 'Mean Error (Avg)'])
allTests = pd.DataFrame(allTests, columns=['Training horizon',
    'Forecast horizon', 'Paired t-test', 'Wilcoxon signed-ranks test'])

# save dataframes to csv
techTests.to_csv('StatisticalTests' + os.path.sep + 'techTests.csv')
allTests.to_csv('StatisticalTests' + os.path.sep + 'AllTests.csv')

# plot tech results
fig, ax = plottingFunctions.plotStatisticalTestTech(techTests)


# repeat selecting only techs that cover all the ranges examined
techs = sel['Tech'].unique()

# create lists to store results
techTests, allTests = [], []

# iterate over training and forecast ranges
for r1 in enumerate(rangesmed):
    for r2 in enumerate(rangesmed):

        # select data for which:
        # - training horizon is at least the required value
        # - max forecast horizon of the point is at least the required value
        # - forecast horizon is above defined lower range
        # - forecast horizon is below defined higher range
        sel = dfObsErrAll\
            .loc[dfObsErrAll['Training horizon'] >= \
                rangesmed[r1[0]]]\
            .loc[dfObsErrAll['Max forecast horizon'] >= \
                 rangesmed[r2[0]]]\
            .loc[dfObsErrAll['Forecast horizon'] >= \
                 rangeslo[r2[0]]]\
            .loc[dfObsErrAll['Forecast horizon'] < \
                 rangeshi[r2[0]]].copy()

        # create lists to store errors
        errTech, errAvg = [], []

        # iterate over technologies
        for t in sel['Tech'].unique():

            if t not in techs:
                continue

            # select data for technology
            errTech.append(sel.loc[\
                sel['Tech']==t, 'Error (Tech)'].values)
            errAvg.append(sel.loc[\
                sel['Tech']==t, 'Error (Avg)'].values)

            # for each technology append:
            # - the technology
            # - the training horizon
            # - the forecast horizon
            # - the p value of the paired t-test
            # - the p value of the wilcoxon signed rank test
            # - the RMSE of the tech using tech specific method
            # - the RMSE of the tech using average method
            techTests.append([t, r1[1], r2[1],
                scipy.stats.ttest_rel(\
                    errTech[-1], errAvg[-1])[1],
                scipy.stats.wilcoxon(  
                    errTech[-1], errAvg[-1])[1],
                np.mean([x**2 for x in errTech[-1]])**0.5,  
                np.mean([x**2 for x in errAvg[-1]])**0.5])

        # append:
        # - the training horizon
        # - the forecast horizon
        # - the p value of the paired t-test
        # - the p value of the mann whitney test
        allTests.append([r1[1], r2[1], 
            scipy.stats.ttest_rel(\
                [np.mean([x**2 for x in a])**0.5 for a in errTech],
                [np.mean([x**2 for x in a])**0.5 for a in errAvg])[1],
            scipy.stats.wilcoxon(\
                [np.mean([x**2 for x in a])**0.5 for a in errTech],
                [np.mean([x**2 for x in a])**0.5 for a in errAvg])[1]])
        
techTests = pd.DataFrame(techTests, columns=['Tech', 'Training horizon', 
    'Forecast horizon', 'Paired t-test', 'Wilcoxon signed-ranks test',
    'Mean Error (Tech)', 'Mean Error (Avg)'])
allTests = pd.DataFrame(allTests, columns=['Training horizon',
    'Forecast horizon', 'Paired t-test', 'Wilcoxon signed-ranks test'])

techTests.to_csv('StatisticalTests' +os.path.sep + 'techTests_sameTechs.csv')
allTests.to_csv('StatisticalTests' +os.path.sep + 'AllTests_sameTechs.csv')

if not os.path.exists('figs' + os.path.sep + 'SupplementaryFigures'):
    os.makedirs('figs' + os.path.sep + 'SupplementaryFigures')
fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
            os.path.sep + 'StatisticalTestTechs.png')

plt.show()
