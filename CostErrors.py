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

# create figure
fig, ax = plt.subplots(2, 4, 
                       figsize = (15,8),
                       subplot_kw=dict(box_aspect=1),)

# scatter observations and forecasts
ax[0][0].scatter(10**dfObsErr['Forecast horizon'].values,
                 10**dfObsErr['Observation'].values,
                 marker = '.', s=2, alpha=.1,
                 color=sns.color_palette('colorblind')[3])
ax[0][1].scatter(10**dfObsErr['Forecast horizon'].values,
                 10**dfObsErr['Forecast (Tech)'].values,
                 marker = '.', s=2, alpha=.1,
                 color=sns.color_palette('colorblind')[0])
ax[0][2].scatter(10**dfObsErr['Forecast horizon'].values,
                 10**dfObsErr['Forecast (Avg)'].values,
                 marker = '.', s=2, alpha=.1,
                 color=sns.color_palette('colorblind')[2])

# remove empty panels
ax[0][3].axis('off')
ax[1][0].axis('off')

# scatter errors
ax[1][1].scatter(10**dfObsErr['Forecast horizon'].values,
                 10**dfObsErr['Error (Tech)'].values,
                 marker = '.', s=2, alpha=.1,
                 color=sns.color_palette('colorblind')[0])
ax[1][2].scatter(10**dfObsErr['Forecast horizon'].values,
                 10**dfObsErr['Error (Avg)'].values,
                 marker = '.', s=2, alpha=.1,
                 color=sns.color_palette('colorblind')[2])

# adjust axes limits and labels
ax[0][0].set_ylabel('Unit cost')
ax[1][1].set_ylim(0.1, 10)
ax[1][2].set_ylim(0.1, 10)
ax[1][1].set_xlim(1, 12)
ax[1][1].set_yscale('log', base=10)
ax[1][2].set_yscale('log', base=10)
ax[0][0].annotate('Cumulative production'
                  ' / current cumulative production', 
                  xy=(0.5, 0.05), 
             xycoords='figure fraction',
             ha='center', va='center',)



# add boxplots on top

# select upper and lower bound for bins
lower = [10**0, 10**(1/3), 10**(2/3), 10**(1)]
medium = [10**(1/6), 10**(3/6), 10**(5/6), 10**(7/6)]
upper = [10**(1/3), 10**(2/3), 10**(1), 10**(4/3)]

# iterate over variables for which to plot boxplots
for var in [['Observation',0,0],
            ['Forecast (Tech)',0,1],
            ['Error (Tech)',1,1],
            ['Error (Avg)',1,2]]:

    # create lists to store data
    upp, low = [0], [0]
    p75, p25, med = [0], [0], [0]

    # iterate over bins
    for i in range(len(lower)):

        # select data for current bin
        sel = dfObsErr.loc[\
            (dfObsErr['Forecast horizon']>=np.log10(lower[i])) &\
            (dfObsErr['Forecast horizon']<np.log10(upper[i]))].copy()
        
        # compute weights for each technology
        sel['weights'] = 1.0
        for t in sel['Tech'].unique():
            sel.loc[sel['Tech']==t, 'weights'] = 1.0 / \
                sel.loc[sel['Tech']==t].shape[0] / \
                sel['Tech'].nunique()
        
        # compute weighted quantiles
        q5, q25, q50, q75, q95 = \
            sm.stats.DescrStatsW(\
                sel[var[0]], weights=sel['weights'])\
                    .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        
        # plot boxplots
        ax[var[1]][var[2]].plot(\
            [medium[i]-0.25, medium[i]+0.25], 
            10**np.array([q50, q50]), 
            color='black')
        ax[var[1]][var[2]].plot(\
            [medium[i]-0.25, medium[i]+0.25, 
                medium[i]+0.25, medium[i]-0.25, medium[i]-0.25], 
            10**np.array([q25, q25, q75, q75, q25]), 
            color='black',)
        
        # append data to lists
        upp.append(q95)
        low.append(q5)
        p75.append(q75)
        p25.append(q25)
        med.append(q50)
    
    # plot shaded area and median line
    ax[var[1]][var[2]].fill_between(\
        [1,*medium], 10**np.array(low), 10**np.array(upp), 
        color='black', alpha=.2)    
    ax[var[1]][var[2]].fill_between(\
        [1,*medium], 10**np.array(p25), 10**np.array(p75), 
        color='black', alpha=.2)    
    ax[var[1]][var[2]].plot(\
        [1,*medium], 10**np.array(med), color='black')    

ax[0][1].set_yticklabels([])
ax[0][2].set_yticklabels([])
ax[1][2].set_yticklabels([])
ax[1][1].set_ylabel('Observed / forecasted cost')

# annotate figure
ax[0][0].annotate('Observations', xy=(6.5,1.6),
                  ha='center', va='center',
                  xycoords='data', 
                  color=sns.color_palette('colorblind')[3])
ax[0][1].annotate('Technology-specific\nslope', 
                  xy=(6.5,1.5),
                  ha='center', va='center',
                  xycoords='data',
                  color=sns.color_palette('colorblind')[0])
ax[0][2].annotate('Average slope',
                    xy=(6.5,1.6),
                    ha='center', va='center',
                    xycoords='data',
                    color=sns.color_palette('colorblind')[2])

ax[1][1].annotate('Technology-specific\nslope', 
                  xy=(6.5,5),
                  ha='center', va='center',
                  xycoords='data',
                  color=sns.color_palette('colorblind')[0])
ax[1][2].annotate('Average slope',
                    xy=(6.5,6),
                    ha='center', va='center',
                    xycoords='data',
                    color=sns.color_palette('colorblind')[2])

# add boxplot legend
ax[1][0].plot([0,.5], [1,1], color='black')
ax[1][0].plot([0,.5,.5,0,0], [0.5,0.5,1.5,1.5,0.5], color='black')
ax[1][0].fill_between([0,.5], [0,0], [2,2], color='black', alpha=.2)
ax[1][0].fill_between([0,.5], [.5,.5], [1.5,1.5], color='black', alpha=.2)
ax[1][0].set_ylim(-1,3)
ax[1][0].set_xlim(-1.8,3)
ax[1][0].annotate('50%', xy=(-.5,1),
                    ha='center', va='center',
                    xycoords='data')
ax[1][0].annotate('90%', xy=(-1.5,1),
                    ha='center', va='center',
                    xycoords='data')
ax[1][0].annotate('Median', xy=(.6,1),
                  ha='left', va='center',
                    xycoords='data')
ax[1][0].plot([-.1,-.5,-.5], [1.5,1.5,1.25], color='black')
ax[1][0].plot([-.1,-.5,-.5], [.5,.5,.75], color='black')
ax[1][0].plot([-.1,-1.5,-1.5], [2,2,1.25], color='silver')
ax[1][0].plot([-.1,-1.5,-1.5], [0,0,.75], color='silver')

ax[0][0].set_xlim(1,12)
ax[0][0].set_xticklabels([])
ax[0][1].set_xlim(1,12)
ax[0][1].set_xticklabels([])
ax[0][2].set_xlim(1,12)
ax[0][2].set_xticklabels([])
ax[1][1].set_xlim(1,12)
# ax[1][1].set_xticklabels([])
ax[1][2].set_xlim(1,12)
# ax[1][2].set_xticklabels([])
ax[0][0].set_ylim(0,1.75)
ax[0][1].set_ylim(0,1.75)
ax[0][2].set_ylim(0,1.75)

## plot error boxplot comparison

sel = dfObsErr.loc[\
    (dfObsErr['Forecast horizon']>=np.log10(10**0.5)) &\
    (dfObsErr['Forecast horizon']<np.log10(10**1.5))].copy()

# compute weights for each technology
sel['weights'] = 1.0
for t in sel['Tech'].unique():
    sel.loc[sel['Tech']==t, 'weights'] = 1.0 / \
        sel.loc[sel['Tech']==t].shape[0] / \
        sel['Tech'].nunique()

for var in ['Error (Tech)', 'Error (Avg)']:
    # compute weighted quantiles
    q5, q25, q50, q75, q95 = \
        sm.stats.DescrStatsW(\
            sel[var], weights=sel['weights'])\
                .quantile([0.05, 0.25, 0.5, 0.75, 0.95])

    if var=='Error (Tech)':
        color = sns.color_palette('colorblind')[0]
    else:
        color = sns.color_palette('colorblind')[2]

    # plot boxplots
    ax[1][3].plot(\
        [1*(var=='Error (Avg)')-0.25, 1*(var=='Error (Avg)')+0.25], 
        10**np.array([q50, q50]), 
        color=color)
    ax[1][3].plot(\
        [1*(var=='Error (Avg)')-0.25, 1*(var=='Error (Avg)')+0.25, 
            1*(var=='Error (Avg)')+0.25, 1*(var=='Error (Avg)')-0.25, 1*(var=='Error (Avg)')-0.25], 
        10**np.array([q25, q25, q75, q75, q25]), 
        color=color)

    # plot shaded area and median line
    ax[1][3].fill_between(\
        [1*(var=='Error (Avg)')-.25, 1*(var=='Error (Avg)')+.25], 
        10**np.array([q5,q5]), 10**np.array([q95,q95]), 
        color=color, alpha=.2)    
    ax[1][3].fill_between(\
        [1*(var=='Error (Avg)')-.25, 1*(var=='Error (Avg)')+.25], 
        10**np.array([q25,q25]), 10**np.array([q75,q75]), 
        color=color, alpha=.2)  

ax[1][3].set_ylim(0.1,10)
ax[1][3].set_yscale('log', base=10)
ax[1][3].set_yticklabels([])
ax[1][3].set_xticks([0,1], labels=['Technology-specific\nslope', 'Average\nslope'])
ax[1][1].yaxis.grid(which='minor', linewidth=0.5)
ax[1][2].yaxis.grid(which='minor', linewidth=0.5)
ax[1][3].yaxis.grid(which='minor', linewidth=0.5)

fig.subplots_adjust(bottom=0.15, top=0.95,
                    left=.06, right=.98,)
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
