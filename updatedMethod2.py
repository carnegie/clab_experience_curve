import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib, analysisFunctions, plottingFunctions

matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('ExpCurves.csv')

# plot orders of magnitudes available for each technology
plottingFunctions.plotOrdersOfMagnitude(df)
plt.show()

# set training and forecast orders of magnitude
trOrds = [0.5,1,2]
forOrds = [0.5,1,2]
# set sampling points per order of magnitude used for plotting functions
samplingPoints = 5

tErrs = []
fErrsTech = []
fErrsAvg = []
Ranges = []
for tOrd in trOrds:
    for fOrd in forOrds:

        # compute points errors for each training and forecast range 
        trainErr, dferrTech, dferrAvg = analysisFunctions.computeErrors(df, tOrd, fOrd)

        # store data in a dataframe
        columns = ['Forecast horizon', 'Error', 'Tech']
        trainErr = pd.DataFrame(trainErr, columns = columns)
        dferrTech = pd.DataFrame(dferrTech, columns = columns)
        dferrAvg = pd.DataFrame(dferrAvg, columns=columns)

        print('Count of technologies with enough orders of magnitude: ', dferrTech['Tech'].nunique())
        print(trainErr['Tech'].unique())

        # plot errors
        plottingFunctions.plotForecastErrors(dferrTech, dferrAvg, trainErr,
                                             tOrd, fOrd, samplingPoints)
        
        # store data in lists for later use
        tErrs.append(trainErr)
        fErrsTech.append(dferrTech)
        fErrsAvg.append(dferrAvg)
        Ranges.append([tOrd, fOrd])

plt.show()
exit()

## boxplots
fig, ax = plt.subplots(3,3, sharey=True, sharex=True, figsize=(10,8))
tOrd = [0.5,1,2]
count = 0
props1 = dict(color=cmapp(0.7), lw=2)
props2 = dict(color=cmapg(0.7), lw=2)

for tOrd in trOrds:
    if tOrd == 0.5:
        forOrds = [0.5,1,2]
    elif tOrd == 1:
        forOrds = [0.5,1,]
    elif tOrd == 2:
        forOrds = [0.5,]
    for fOrd in forOrds:
        trainingOrdMag = np.log10(10**tOrd)
        forecastOrdMag = np.log10(10**fOrd)
        samplingPoints = 5
        counterr = 0
        # count = 0
        countTot = 0
        trainErr, dferrTech, dferrAvg = [], [], []
        for tech in df['Tech'].unique():
            # if 'Offshore' in tech:
            #     sel = df.loc[df['Tech']==tech].copy()
            #     x = np.log10(sel['Cumulative production'].values)
            #     y = np.log10(sel['Unit cost'].values)
            #     plt.figure()

            #     plt.scatter(x,y)
            #     plt.show()

            #     continue  
            # computing average technological slope based on all other technologies
            slopeall = np.mean(slopes.loc[slopes['Tech'] != tech,'Slope'].values)
            # computing technology specific slope
            sel = df.loc[df['Tech']==tech].copy()
            # if x[-1] - x[0] > trainingOrdMag+forecastOrdMag:
                # count += 1
            x = np.log10(sel['Cumulative production'].values)
            y = np.log10(sel['Unit cost'].values)
            H = len(x)
            print(tech, x[-1] - x[0], x[-1] - x[0]>trainingOrdMag+forecastOrdMag )
            # select N points before midpoint and compute slope
            for i in range(H):
                flag = 0
                for N in range(i-1, -1, -1):
                    if x[i] - x[N] >= trainingOrdMag and flag == 0:
                        slope = (y[i] - y[N]) /\
                            (x[i] - x[N])
                        # add linear regression method and compute training error
                        for M in range(i+1, H):
                            if x[M] - x[i] >= forecastOrdMag and flag == 0:
                                model = sm.OLS(y[N:i+1], sm.add_constant(x[N:i+1]))
                                result = model.fit()
                                terr = y[N:i+1] - result.predict(sm.add_constant(x[N:i+1]))
                                slope = result.params[1]
                                for idx in range(len(terr)):
                                    trainErr.append([x[N+idx] - x[i], terr[idx], tech])
                            # compute error associated using slope M points after midpoint
                                pred =  y[i] + slope * (x[i:M+1] - x[i])
                                pred2 =  y[i] + slopeall * (x[i:M+1] - x[i])
                                error = (y[i:M+1] - (pred)) 
                                error2 = (y[i:M+1] - (pred2)) 
                                for idx in range(len(error)):
                                    dferrTech.append([x[i+idx] - x[i], error[idx], tech])
                                    dferrAvg.append([x[i+idx] - x[i], error2[idx], tech])
                                flag = 1
                                countTot += 1

        print('Total points found with selected orders of magnitude: ', countTot)

        trainErr = pd.DataFrame(trainErr, 
                            columns = ['Forecast horizon',
                                        #'Log of ratios for prediction',
                                        'Error', 'Tech'])
        dferrTech = pd.DataFrame(dferrTech, 
                            columns = ['Forecast horizon',
                                        #'Log of ratios for prediction',
                                        'Error', 'Tech'])
        dferrAvg = pd.DataFrame(dferrAvg, 
                            columns = [#'Log of ratios for predictor',
                                        'Forecast horizon',
                                        'Error', 'Tech'])
        # print('Count of technologies with enough orders of magnitude: ', count)
        # print(trainErr.shape, dferrTech.shape, dferrAvg.shape)
        # print(trainErr['Tech'].nunique(), dferrTech['Tech'].nunique(), dferrAvg['Tech'].nunique())
        # print(trainErr['Tech'].unique())

        # trainErr['Weights'] = 0.0
        # if trainErr.shape[0] == 0:
        #     pctTrain.append([trainInt[i],np.nan,np.nan,np.nan,np.nan,np.nan])
        #     countPoints.append(0)
        #     countTechs.append(0)
        #     continue
        # for tt in trainErr['Tech'].unique():
        #     trainErr.loc[trainErr['Tech']==tt,'Weights'] = 1/trainErr.loc[trainErr['Tech']==tt].count()[0]
        # trainErr = trainErr.sort_values(by='Error', ascending=True)
        # cumsum = trainErr['Weights'].cumsum().round(4)
        # pt = []
        # stats = {}
        # labs = {0: 'whislo', 25: 'q1', 50: 'med', 75: 'q3', 100: 'whishi'}
        # for q in [0,25,50,75,100]:
        # # for q in [0,10,20,30,40,50,60,70,80,90,100]:
        #     cutoff = sel['Weights'].sum() * q/100
        #     pt.append(trainErr['Error'][cumsum >= cutoff.round(4)].iloc[0])
        #     stats[labs[q]] = trainErr['Error'][cumsum >= cutoff.round(4)].iloc[0]
        

        # plot forecast error
        dferrTech['Weights'] = 0.0
        dferrAvg['Weights'] = 0.0
        for tt in dferrTech['Tech'].unique():
            dferrTech.loc[dferrTech['Tech']==tt,'Weights'] = 1/dferrTech.loc[dferrTech['Tech']==tt].count()[0]
        for tt in dferrAvg['Tech'].unique():
            dferrAvg.loc[dferrAvg['Tech']==tt,'Weights'] = 1/dferrAvg.loc[dferrAvg['Tech']==tt].count()[0]
        dferrTech = dferrTech.sort_values(by='Error', ascending=True)
        dferrAvg = dferrAvg.sort_values(by='Error', ascending=True)
        cumsum1 = dferrTech['Weights'].cumsum().round(4)
        cumsum2 = dferrAvg['Weights'].cumsum().round(4)
        statsTech = {}
        statsAvg = {}
        labs = {0: 'whislo', 25: 'q1', 50: 'med', 75: 'q3', 100: 'whishi'}
        for q in [0,10,25,50,75,90,100]:
        # for q in [0,10,20,30,40,50,60,70,80,90,100]:
            cutoff1 = dferrTech['Weights'].sum() * q/100
            cutoff2 = dferrAvg['Weights'].sum() * q/100
            # pt1.append(dferrTech['Error'][cumsum1 >= cutoff1.round(4)].iloc[0])
            # pt2.append(dferrAvg['Error'][cumsum2 >= cutoff2.round(4)].iloc[0])
            if q in labs.keys():
                statsTech[labs[q]] = 10**dferrTech['Error'][cumsum1 >= cutoff1.round(4)].iloc[0]
                statsAvg[labs[q]] = 10**dferrAvg['Error'][cumsum2 >= cutoff2.round(4)].iloc[0]
        print(count)
        ax[int(count/3)][count%3].bxp([statsTech], positions = [0], showfliers=False, boxprops=props1)
        ax[int(count/3)][count%3].bxp([statsAvg], positions = [1], showfliers=False, boxprops=props2)
        ax[int(count/3)][count%3].set_yscale('log', base=10)
        ax[int(count/3)][count%3].set_ylim(0.1,10)
        ax[int(count/3)][count%3].plot([-1,3],[1,1],'k', zorder=-10, alpha=0.5)
        ax[int(count/3)][count%3].set_xlim(-0.5,1.5)
        ax[int(count/3)][count%3].annotate('Techs = '+str(dferrAvg['Tech'].nunique()) +  
                                            '\n N = '+str(dferrAvg['Error'].count()), xy=(0.5, 4), xycoords='data', ha='center', va='bottom', fontsize=12)
        count += 1
        if tOrd == 1 and fOrd == 1:
            count += 1
        if tOrd == 2 and fOrd > 0.5:
            count +=1
[ax[2][x].set_xticks([0,1],['Technology-specific','Average slope']) for x in range(3)]
ax[1][0].set_ylabel('Error (Actual/Predicted)')
ax[1][0].annotate('Training interval',
            xy=(-.6, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            fontsize=12,
            rotation=90)
count = 0
for l in trOrds:
    ax[count][0].annotate("$10^{{{}}}$".format(l),
            xy=(-.4, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            # fontsize=20
            )
    count += 1
ax[0][1].annotate('Forecast interval',
            xy=(.5, 1.3), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            fontsize=12
            )
count = 0
for l in [0.5,1,2]:
    ax[0][count].annotate("$10^{{{}}}$".format(l),
            xy=(.5, 1.1), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            # fontsize=20
            )
    count += 1
plt.subplots_adjust(bottom=0.05, top=0.9, left=0.175, right=0.975)
plt.show()