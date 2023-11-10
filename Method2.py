import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib, analysisFunctions, plottingFunctions
import scipy
from scipy.stats import distributions

matplotlib.rc('savefig', dpi=300)

matplotlib.rc('font', 
              **{'family':'sans-serif','sans-serif':'Helvetica'})
matplotlib.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('ExpCurves.csv')

df = df.loc[~(df['Tech'].str.contains('Nuclear'))]

# plot orders of magnitudes available for each technology
plottingFunctions.plotOrdersOfMagnitude(df)

# set training and forecast orders of magnitude
trOrds = [0.5,1,2]
forOrds = [0.5,1,2]
# set sampling points per order of magnitude 
# used for plotting functions
samplingPoints = 3

# supplementary figures : plots of forecast errors

# initialize lists to be used to store data
tErrs = []
fErrsTech = []
fErrsAvg = []
fSlErrTech = []
fSlErrAvg = []
Ranges = []
obsErr = []
predErr = []
pred2Err = []
slopeVals = []
testRes = []
testResDetail = []
for tOrd in trOrds:
    for fOrd in forOrds:

        # compute points errors for each training and forecast range 
        trainErr, dferrTech, dferrAvg, \
            slopeErrTech, slopeErrAvg, slopevals, obspred = \
            analysisFunctions.computeErrors(df, tOrd, fOrd)

        # store data in a dataframe
        columns = ['Forecast horizon', 'Error', 'Tech']
        trainErr = pd.DataFrame(trainErr, columns = columns)
        dferrTech = pd.DataFrame(dferrTech, columns = ['Forecast horizon', 'Error', 'Tech','F p-value'])
        dferrAvg = pd.DataFrame(dferrAvg, columns=columns)
        slopeErrTech = pd.DataFrame(slopeErrTech, columns=columns)
        slopeErrAvg = pd.DataFrame(slopeErrAvg, columns=columns)
        slopevals = pd.DataFrame(slopevals, columns=['Slope val','Slope training','Average slope','Tech'])
        slopevals['Average slope'] = slopevals['Average slope'].mean()
        # slopevals = slopevals.groupby('Tech').mean().reset_index()
        obspred = pd.DataFrame(obspred, columns=['Forecast horizon',
                                                 'Observed', 'PredTech',
                                                 'PredAvg','Tech'])
        obs = obspred[['Forecast horizon', 'Observed','Tech']]
        obs.columns = ['Forecast horizon', 'Error', 'Tech']
        pred = obspred[['Forecast horizon', 'PredTech','Tech']]
        pred.columns = ['Forecast horizon', 'Error', 'Tech']
        pred2 = obspred[['Forecast horizon', 'PredAvg','Tech']]
        pred2.columns = ['Forecast horizon', 'Error', 'Tech']
        # # uncomment to print number of technologies and their names
        # print('Count of technologies ' + \
        #       'with enough orders of magnitude: ', 
        #       dferrTech['Tech'].nunique())
        # print(trainErr['Tech'].unique())

        # # # # plot forecast errors (lines and boxplots)
        # plottingFunctions.plotForecastErrors(dferrTech, dferrAvg, 
        #                                      fOrd, samplingPoints,
        # # for training errors to be included, uncomment the following line
        #                                      trainErr, tOrd
        #                                      )

        # store data in lists for later use
        tErrs.append(trainErr)
        fErrsTech.append(dferrTech)
        fErrsAvg.append(dferrAvg)
        fSlErrTech.append(slopeErrTech)
        fSlErrAvg.append(slopeErrAvg)
        obsErr.append(obs)
        predErr.append(pred)
        pred2Err.append(pred2)
        slopeVals.append(slopevals)
        Ranges.append([tOrd, fOrd])

        # print('testing distributions for : ', tOrd, ' ', fOrd,)
        ttech = 0
        aavg = 0
        for tech in dferrTech['Tech'].unique():
            try:
                pvalue = scipy.stats.ks_2samp(dferrTech.loc[dferrTech['Tech']==tech,'Error'], dferrAvg.loc[dferrAvg['Tech']==tech,'Error']).pvalue
                # plt.figure()
                # at = dferrTech.loc[dferrTech['Tech']==tech,'Error'].sort_values().values.tolist().copy()
                # av = dferrAvg.loc[dferrAvg['Tech']==tech,'Error'].sort_values().values.tolist().copy()
                # plt.plot(at, 
                #             [x/len(at) for x in range(len(at))], color='royalblue')
                # plt.plot(av, 
                #             [x/len(av) for x in range(len(av))], color='forestgreen')
                # plt.xlabel('Error values')
                # plt.ylabel('Cumulative probability distribution')
                # plt.title(tech+' - pvalue: '+str(pvalue))
                # plt.show()

                
                # pvalue = scipy.stats.ttest_ind(dferrTech.loc[dferrTech['Tech']==tech,'Error'], dferrAvg.loc[dferrAvg['Tech']==tech,'Error']).pvalue
            except ValueError:
                pvalue = 1
            if pvalue < 0.05:
                if np.mean(dferrTech.loc[dferrTech['Tech']==tech,'Error'].values**2) < np.mean(dferrAvg.loc[dferrAvg['Tech']==tech,'Error'].values**2):
                    ttech += 1
                else:
                    aavg += 1
            testResDetail.append([tOrd, fOrd, tech, pvalue, np.mean(dferrTech.loc[dferrTech['Tech']==tech,'Error'].values**2) - np.mean(dferrAvg.loc[dferrAvg['Tech']==tech,'Error'].values**2)])
        print('\n',tOrd, fOrd, ttech, aavg)
        testRes.append([ttech, aavg, dferrTech['Tech'].nunique() - ttech - aavg ])
        pvalue = scipy.stats.ks_2samp(dferrTech['Error'].values, dferrAvg['Error']).pvalue
        print(pvalue)
plottingFunctions.plotObsPred(obsErr, predErr, pred2Err,
                              [1,1], Ranges, 
                            samplingPoints, )

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
# sectorTech = [analysisFunctions\
#               .sectorsinv[tech] for tech in df['Tech'].unique()]
testResDetail = testResDetail.sort_values(by=['Sector','Tech'])
Ranges_ = [Ranges[x] for x in [0,1,3,4,2,6,5,7,8]]

# plottingFunctions.plotSignificance(testResDetail,Ranges_)

# plt.show()

# exit()
# Figure 2
# trForOrds = [[0.5, 0.5],
#              [0.5, 1],
#              [1, 0.5],
            #  [1, 1] ]
# trForOrds = [[0.5, 0.5],
#              [0.5, 1],
#              [0.5, 2],
#              [1, 0.5],
#              [1, 1] ,
#              [1, 2],
#              [2, 0.5],
#              [2, 1],
#              [2, 2]]
# fig, ax = plottingFunctions.plotForecastErrorGrid(fErrsTech, fErrsAvg, 
#                                          Ranges, trForOrds, samplingPoints,)
# plt.show()
# trForOrds = [[0.5, 0.5],
#              [0.5, 1],
#              [0.5, 2],
#              [1, 0.5],
#              [1, 1] ,
#              [1, 2],
#              [2, 0.5],
#              [2, 1],
#              [2, 2]]
# fig, ax = plottingFunctions.plotR2Grid(slopeVals, Ranges)
# fig, ax = plottingFunctions.plotR2Contour(slopeVals, Ranges)

fig, ax = plottingFunctions.plotForecastSlopeError(fErrsTech, fErrsAvg, 
                                                [1,1], Ranges, 
                                                samplingPoints, vert=True)
plt.show()
# fig, ax = plottingFunctions.plotSlopeErrorGrid(fSlErrTech, fSlErrAvg, 
#                                                trForOrds, Ranges, vert=True)
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
