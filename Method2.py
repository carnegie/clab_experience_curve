import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib, analysisFunctions, plottingFunctions

matplotlib.rc('font', 
              **{'family':'sans-serif','sans-serif':['Helvetica']})
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
samplingPoints = 5

# supplementary figures : plots of forecast errors

# initialize lists to be used to store data
tErrs = []
fErrsTech = []
fErrsAvg = []
fSlErrTech = []
fSlErrAvg = []
Ranges = []
for tOrd in trOrds:
    for fOrd in forOrds:

        # compute points errors for each training and forecast range 
        trainErr, dferrTech, dferrAvg, \
            slopeErrTech, slopeErrAvg = \
            analysisFunctions.computeErrors(df, tOrd, fOrd)

        # store data in a dataframe
        columns = ['Forecast horizon', 'Error', 'Tech']
        trainErr = pd.DataFrame(trainErr, columns = columns)
        dferrTech = pd.DataFrame(dferrTech, columns = columns)
        dferrAvg = pd.DataFrame(dferrAvg, columns=columns)
        slopeErrTech = pd.DataFrame(slopeErrTech, columns=columns)
        slopeErrAvg = pd.DataFrame(slopeErrAvg, columns=columns)

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
        Ranges.append([tOrd, fOrd])
# exit()
# Figure 2
trForOrds = [[0.5, 0.5],
             [0.5, 1],
             [1, 0.5],
             [1, 1] ]
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

# trForOrds = [[0.5, 0.5],
#              [0.5, 1],
#              [0.5, 2],
#              [1, 0.5],
#              [1, 1] ,
#              [1, 2],
#              [2, 0.5],
#              [2, 1],
#              [2, 2]]
fig, ax = plottingFunctions.plotForecastSlopeError(fErrsTech, fErrsAvg, 
                                                fSlErrTech, fSlErrAvg,
                                                [1,1], Ranges, 
                                                samplingPoints, vert=True)
# fig, ax = plottingFunctions.plotSlopeErrorGrid(fSlErrTech, fSlErrAvg, 
#                                                trForOrds, Ranges, vert=True)
fig, ax = plottingFunctions.plotRankTechnologies(fErrsTech, fErrsAvg,
                                                 fSlErrTech, fSlErrAvg,
                                                 [1,1], Ranges)
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
