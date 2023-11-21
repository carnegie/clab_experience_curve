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

# select only technologies for which at least 
# one order of magnitude of data is available
# for both training and forecast horizons
dfObsErr = dfObsErr.loc[\
    dfObsErr['Tech'].isin(\
        dfObsErr.copy().loc[\
            (dfObsErr['Training horizon']>=1) &\
            (dfObsErr['Forecast horizon']>=1),'Tech'].values)]

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
fig, ax = plottingFunctions.plotErrorTech(dfObsErr)

plt.show()
exit()

exit()

# set training and forecast orders of magnitude
trOrds = [0.5,1,2]
forOrds = [0.5,1,2]

# set sampling points per order of magnitude 
# used for plotting functions
samplingPoints = 3

# initialize lists to be used to store data
fErrsTech, fErrsAvg = [], []
Ranges = []
obs, predTech ,predAvg = [], [], []
testRes, testResDetail = [], []

# iterate over the selected 
# training and forecast orders of magnitudes
for tOrd in trOrds:
    for fOrd in forOrds:

        # compute errors for each training and forecast range 
        _, dferrTech, dferrAvg, obspred = \
            analysisFunctions.computeErrors(df, tOrd, fOrd)

        # store data in a dataframe
        columns = ['Forecast horizon', 'Error', 'Tech']
        dferrTech = pd.DataFrame(dferrTech, 
                        columns = ['Forecast horizon', 
                                    'Error', 
                                    'Tech'])
        dferrAvg = pd.DataFrame(dferrAvg, columns=columns)

        # store in dataframe and rename columns
        obspred = pd.DataFrame(obspred, 
                        columns=['Forecast horizon',
                                    'Observed', 'PredTech',
                                    'PredAvg','Tech'])
        o = obspred[['Forecast horizon', 'Observed','Tech']]
        o.columns = ['Forecast horizon', 'Error', 'Tech']
        pTech = obspred[['Forecast horizon', 'PredTech','Tech']]
        pTech.columns = ['Forecast horizon', 'Error', 'Tech']
        pAvg = obspred[['Forecast horizon', 'PredAvg','Tech']]
        pAvg.columns = ['Forecast horizon', 'Error', 'Tech']

        # store data in lists for later use
        fErrsTech.append(dferrTech)
        fErrsAvg.append(dferrAvg)
        obs.append(o)
        predTech.append(pTech)
        predAvg.append(pAvg)
        Ranges.append([tOrd, fOrd])

        # initialize counters to store statistical test results 
        ttech, aavg = 0, 0

        # iterate over all technologies
        for tech in dferrTech['Tech'].unique():

            # perform kolmogorov smirnov statistical test
            # to check if error differences are significant
            pvalue = scipy.stats.ks_2samp(\
                dferrTech.loc[\
                    dferrTech['Tech']==tech,'Error'], 
                dferrAvg.loc[\
                    dferrAvg['Tech']==tech,'Error']
                    ).pvalue

            # if errors are significantly 
            # save which method has lowest error
            if pvalue < 10:
                eTech = np.mean(dferrTech.loc[\
                    dferrTech['Tech']==tech,'Error'].values**2)
                eAvg = np.mean(dferrAvg.loc[\
                    dferrAvg['Tech']==tech,'Error'].values**2)
                if eTech < eAvg:
                    ttech += 1
                else:
                    aavg += 1
            
            # append information to list
            testResDetail.append([tOrd, fOrd, 
                                  tech, pvalue, 
                                  eTech - eAvg])

        testRes.append([ttech, aavg, dferrTech['Tech'].nunique() - ttech - aavg ])

# plot observations, and forecasts
# then compute errors and plot them
# finally report aggregated errors

fig, ax = plt.subplots(2, 4, 
                       sharex=True, sharey='row',
                       figsize = (12,6),
                       subplot_kw=dict(box_aspect=1),)
ax[0][0].scatter(10**obs[Ranges.index([1,1])]['Forecast horizon'].values,
                 10**obs[Ranges.index([1,1])]['Error'].values,
                 marker = '.',
                 color=sns.color_palette('colorblind')[3])
ax[0][1].scatter(10**predTech[Ranges.index([1,1])]['Forecast horizon'].values,
                 10**predTech[Ranges.index([1,1])]['Error'].values,
                 marker = '.',
                 color=sns.color_palette('colorblind')[0])
ax[0][2].scatter(10**predAvg[Ranges.index([1,1])]['Forecast horizon'].values,
                 10**predAvg[Ranges.index([1,1])]['Error'].values,
                 marker = '.',
                 color=sns.color_palette('colorblind')[2])
ax[0][3].axis('off')
ax[1][0].axis('off')
ax[1][1].scatter(10**fErrsTech[Ranges.index([1,1])]['Forecast horizon'].values,
                 10**fErrsTech[Ranges.index([1,1])]['Error'].values,
                 marker = '.',
                 color=sns.color_palette('colorblind')[0])
ax[1][2].scatter(10**fErrsAvg[Ranges.index([1,1])]['Forecast horizon'].values,
                 10**fErrsAvg[Ranges.index([1,1])]['Error'].values,
                 marker = '.',
                 color=sns.color_palette('colorblind')[2])
ax[1][1].set_ylim(0.1, 10)
ax[1][1].set_yscale('log', base=10)
ax[0][0].annotate('Cumulative production / current cumulative production', xy=(0.5, 0.05), 
             xycoords='figure fraction',
             ha='center', va='center',)
fig.subplots_adjust(bottom=0.125, top=0.95,
                    left=.1, right=.95,)
plt.show()
exit()




plottingFunctions.plotObsPred(obs, predTech, predAvg,
                              [1,1], Ranges, 
                            samplingPoints, )

plt.show()
exit()

fig, ax = plt.subplots(3,3, sharex=True, sharey=True, figsize=(11,6))
count = 0
for item in testRes:
    ax[int(count/3)][count%3].bar([0],item[0], color=['royalblue'])
    ax[int(count/3)][count%3].bar([0],item[2], bottom=item[0], color=['grey'])
    ax[int(count/3)][count%3].bar([0],item[1], bottom=item[0]+item[2], color=['forestgreen'])
    ax[int(count/3)][count%3].set_xlim(-1,1)
    count+=1
# plt.show()

testResDetail = pd.DataFrame(testResDetail, columns=['tOrd','fOrd','Tech','pvalue','diff'])

testResDetail['Sector'] = [analysisFunctions.sectorsinv[tech] for tech in testResDetail['Tech']]

testResDetail = testResDetail.sort_values(by=['Sector','Tech'])
Ranges_ = [Ranges[x] for x in [0,1,3,4,2,6,5,7,8]]

plottingFunctions.plotSignificance(testResDetail,Ranges_)
# plt.show()

fig, ax = plottingFunctions.plotForecastSlopeError(fErrsTech, fErrsAvg, 
                                                [1,1], Ranges, 
                                                samplingPoints, vert=True)
plt.show()

fig, ax = plottingFunctions.plotRankTechnologies(fErrsTech, fErrsAvg,
                                                 [1,1], Ranges, samplingPoints)
plt.show()

exit()
# Figure 3: summary boxplots
# select orders of magnitude (training and forecast) to be included
trForOrds = [[0.5, 0.5],
             [0.5, 1],
             [1, 0.5],
             [1, 1] ]
fig, ax = plottingFunctions.summaryBoxplots(trForOrds, fErrsTech, fErrsAvg, Ranges)

# for supplementary material (extend the above until 2 orders of magnitudes)
trForOrds = [[0.5, 0.5],
             [0.5, 1],
             [0.5, 2],
             [1, 0.5],
             [1, 1] ,
             [1, 2],
             [2, 0.5],
             [2, 1],
             [2, 2]]
fig, ax = plottingFunctions.summaryBoxplots(trForOrds, fErrsTech, fErrsAvg, Ranges)

plt.show()
