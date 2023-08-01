import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import cmcrameri as cm
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('ExpCurves.csv')

cmapp = matplotlib.colormaps['Purples']
cmapg = matplotlib.colormaps['Greens']

method = 'regression'

# get slope for all technologies
slopes = []
# slopes_pl = []
for tech in df['Tech'].unique():
    sel = df.loc[df['Tech']==tech]
    x = np.log10(sel['Cumulative production'].values)
    y = np.log10(sel['Unit cost'].values)
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()
    slopes.append([tech, result.params[1]])
slopes = pd.DataFrame(slopes, columns=['Tech', 'Slope'])

count = 0
labs = []
oOfM = []
fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [8,1]})
for tech in df['Tech'].unique():
    sel = df.loc[df['Tech']==tech]
    x = np.log10(sel['Cumulative production'].values)
    x0 = x[0]
    x1 = x[-1]
    oOfM.append([x1-x0])
    ax[0].plot([0, x1-x0], [count, count], color='k', lw=0.5)
    labs.append(tech + ' (' + str(sel.shape[0]) + ')')
    count += 1
ax[0].set_yticks([x for x in range(len(df['Tech'].unique()))], labs, fontsize=5)
ax[0].set_ylabel('Technology')
oOfM.sort()
density = [[0,86]]
for el in oOfM:
    density.append([el[0], density[-1][1]-1])
density = np.array(density)
ax[1].plot(density[:,0], density[:,1], color='k', lw=2)
ax[1].set_xlabel('Orders of magnitude of cumulative production')
ax[1].set_ylabel('Number of technologies available')
plt.subplots_adjust(top=0.98, bottom=0.09, hspace=0.15)
# plt.show()

trOrds = [0.5,1,2]
forOrds = [0.5,1,2]

for tOrd in trOrds:
    for fOrd in forOrds:
        trainingOrdMag = np.log10(10**tOrd)
        forecastOrdMag = np.log10(10**fOrd)
        samplingPoints = 5
        counterr = 0
        count = 0
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
            x = np.log10(sel['Cumulative production'].values)
            y = np.log10(sel['Unit cost'].values)
            if x[-1] - x[0] > trainingOrdMag+forecastOrdMag:
                count += 1
            print(tech, x[-1] - x[0], x[-1] - x[0]>trainingOrdMag+forecastOrdMag )
            H = len(x)
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
                                if method=='regression':
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
        print('Count of technologies with enough orders of magnitude: ', count)
        print(trainErr.shape, dferrTech.shape, dferrAvg.shape)
        print(trainErr['Tech'].nunique(), dferrTech['Tech'].nunique(), dferrAvg['Tech'].nunique())
        print(trainErr['Tech'].unique())

        fig, ax = plt.subplots(3,1, sharex=True, figsize=(9,8))
        figb, axb = plt.subplots(3,1, sharex=True, figsize=(9,8))
        # plot training error
        npoints = int(samplingPoints * trainingOrdMag)
        trainInt = np.linspace(-trainingOrdMag-trainingOrdMag/npoints/2, 0+ trainingOrdMag/npoints/2, npoints+2)
        trainInt = np.linspace(-trainingOrdMag-trainingOrdMag/npoints, 0, npoints+2)
        trainIntAxis = [trainInt[1]]
        for idx in range(1,len(trainInt)-1):
            trainIntAxis.append(trainInt[idx]+\
                                   (trainInt[idx+1]-trainInt[idx])/2)
        trainIntAxis.append(0)
        countPoints = []
        countTechs = []
        pctTrain = []
        for i in range(len(trainInt)):
            if i == 0:
                sel = trainErr.loc[(trainErr['Forecast horizon']<=trainInt[i+1])].copy()
            elif i == len(trainInt)-1:
                sel = trainErr.loc[(trainErr['Forecast horizon']==trainInt[i])].copy()
            else:
                sel = trainErr.loc[(trainErr['Forecast horizon']>trainInt[i]) &\
                            (trainErr['Forecast horizon']<=trainInt[i+1])].copy()
            if sel.shape[0] == 0:
                pctTrain.append([trainInt[i],np.nan,np.nan,np.nan,np.nan,np.nan])
                countPoints.append(0)
                countTechs.append(0)
                continue
            countPoints.append(sel.shape[0])
            countTechs.append(sel['Tech'].nunique())
            for tt in sel['Tech'].unique():
                sel.loc[sel['Tech']==tt,'Weights'] = 1/sel.loc[sel['Tech']==tt].count()[0]
            sel = sel.sort_values(by='Error', ascending=True)
            cumsum = sel['Weights'].cumsum().round(4)
            pt = []
            for q in [0,10,25,50,75,90,100]:
            # for q in [0,10,20,30,40,50,60,70,80,90,100]:
                cutoff = sel['Weights'].sum() * q/100
                pt.append(sel['Error'][cumsum >= cutoff.round(4)].iloc[0])
            pctTrain.append([trainInt[i],*pt])
        pctTrain = np.array(pctTrain)
        ax[0].plot([10**x for x in trainIntAxis],10**pctTrain[:,4], color='b', lw=2)
        for r in range(2,-1,-1):
            ax[0].fill_between([10**x for x in trainIntAxis], 10**pctTrain[:,1+r], 10**pctTrain[:,-r-1], alpha=0.1+0.2*r, color='b', zorder=-2-r, lw=0)
        ax[2].plot([1,1],[0,100**100],'k')
        ax[2].plot([10**x for x in trainIntAxis],countPoints, color='k', lw=2)
        ax[2].set_ylim(0,max(countPoints)*1.1)
        ax2 = ax[2].twinx()
        ax2.plot([10**x for x in trainIntAxis],countTechs, color='red', lw=2)

        for x in trainIntAxis:
            stats = {}
            labs = ['whislo', 'q1', 'med', 'q3', 'whishi']
            count_ = 1
            for l in labs:
                stats[l] = 10**pctTrain[trainIntAxis.index(x),count_]
                count_ += 1
                if count_ == 2:
                    count_ += 1
            axb[0].bxp([stats], positions = [10**x], widths = (10**x)/8, showfliers=False, boxprops=dict(color='b', lw=2), manage_ticks=False)
        axb[0].set_yscale('log', base=10)
        axb[2].plot([1,1],[0,100**100],'k')
        axb[2].plot([10**x for x in trainIntAxis],countPoints, color='k', lw=2)
        axb[2].set_ylim(0,max(countPoints)*1.1)
        axb[0].set_xscale('log', base=10)
        ax2b = axb[2].twinx()
        ax2b.plot([10**x for x in trainIntAxis],countTechs, color='red', lw=2)

        # plot forecast error
        npoints = int(samplingPoints * forecastOrdMag)
        forecastInt = np.linspace(0-forecastOrdMag/npoints, forecastOrdMag, npoints+2)
        forecastIntAxis = [0]
        for idx in range(1,len(forecastInt)-1):
            forecastIntAxis.append(forecastInt[idx]+\
                                   (forecastInt[idx+1]-forecastInt[idx])/2)
        forecastIntAxis.append(forecastOrdMag)
        pctTech, pctAvg = [], []
        countPoints = []
        countTechs = []
        for i in range(len(forecastInt)):
            if i == 0:
                sel1 = dferrTech.loc[(dferrTech['Forecast horizon']==0)].copy()
                sel2 = dferrAvg.loc[(dferrAvg['Forecast horizon']==0)].copy()
            elif i == len(forecastInt)-1:
                sel1 = dferrTech.loc[(dferrTech['Forecast horizon']>=forecastInt[i])].copy()
                sel2 = dferrAvg.loc[(dferrAvg['Forecast horizon']>=forecastInt[i])].copy()
            else:
                sel1 = dferrTech.loc[(dferrTech['Forecast horizon']>forecastInt[i]) &\
                            (dferrTech['Forecast horizon']<=forecastInt[i+1])].copy()
                sel2 = dferrAvg.loc[(dferrAvg['Forecast horizon']>forecastInt[i]) &\
                            (dferrAvg['Forecast horizon']<=forecastInt[i+1])].copy()
            if sel1.shape[0] == 0:
                pctTech.append([forecastInt[i],np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
                pctAvg.append([forecastInt[i],np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
                countPoints.append([0,0])
                countTechs.append([0,0])
                continue
            countPoints.append([sel1.shape[0],sel2.shape[0]])
            countTechs.append([sel1['Tech'].nunique(),sel2['Tech'].nunique()])
            for tt in sel1['Tech'].unique():
                sel1.loc[sel1['Tech']==tt,'Weights'] = 1/sel1.loc[sel1['Tech']==tt].count()[0]
            for tt in sel2['Tech'].unique():
                sel2.loc[sel2['Tech']==tt,'Weights'] = 1/sel2.loc[sel2['Tech']==tt].count()[0]
            sel1 = sel1.sort_values(by='Error', ascending=True)
            sel2 = sel2.sort_values(by='Error', ascending=True)
            cumsum1 = sel1['Weights'].cumsum().round(4)
            cumsum2 = sel2['Weights'].cumsum().round(4)
            pt1, pt2 = [], []
            for q in [0,10,25,50,75,90,100]:
            # for q in [0,10,20,30,40,50,60,70,80,90,100]:
                cutoff1 = sel1['Weights'].sum() * q/100
                cutoff2 = sel2['Weights'].sum() * q/100
                pt1.append(sel1['Error'][cumsum1 >= cutoff1.round(4)].iloc[0])
                pt2.append(sel2['Error'][cumsum2 >= cutoff2.round(4)].iloc[0])
            pctTech.append([forecastInt[i],*pt1])
            pctAvg.append([forecastInt[i],*pt2])
        pctTech = np.array(pctTech)
        pctAvg = np.array(pctAvg)
        ax[0].plot([10**x for x in forecastIntAxis],10**pctTech[:,4], color=cmapp(0.7), lw=2)
        ax[1].plot([10**x for x in forecastIntAxis],10**pctAvg[:,4], color=cmapg(0.7), lw=2)
        for r in range(2,-1,-1):
            ax[0].fill_between([10**x for x in forecastIntAxis], 10**pctTech[:,1+r], 10**pctTech[:,-r-1], alpha=0.1+0.2*r, color=cmapp(0.7), zorder=-2-r, lw=0)
            ax[1].fill_between([10**x for x in forecastIntAxis], 10**pctAvg[:,1+r], 10**pctAvg[:,-r-1], alpha=0.1+0.2*r, color=cmapg(0.7), zorder=-2-r, lw=0)
        ax[2].plot([10**x for x in forecastIntAxis], np.asarray(countPoints)[:,0], color='k', lw=2)
        ax2.plot([10**x for x in forecastIntAxis], np.asarray(countTechs)[:,1], color='red', lw=2)
        ax2.set_ylabel('Number of technologies available', color='red')
        ax2b.set_ylabel('Number of technologies available', color='red')
        ax[0].plot([1,1],[0,100],'k')
        ax[1].plot([1,1],[0,100],'k')
        axb[0].plot([1,1],[0,100],'k')
        axb[1].plot([1,1],[0,100],'k')
        ax[0].annotate('Training', xy=(10**(-trainingOrdMag/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
        ax[0].annotate('Forecast', xy=(10**(+forecastOrdMag/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
        axb[0].annotate('Training', xy=(10**(-trainingOrdMag/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
        axb[0].annotate('Forecast', xy=(10**(+forecastOrdMag/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
        ax[0].set_ylim(0.1,10)
        ax[0].set_yscale('log', base=10)
        ax[0].set_xscale('log', base=10)
        ax[0].set_ylabel('Error (Actual/Predicted)')
        ax[0].set_title('Technologies available: ' + str(trainErr['Tech'].nunique())+
                        '\n Total points with '+ str(trainingOrdMag)+ ' orders of magnitude for training '+
                        ' and '+ str(forecastOrdMag)+' orders of magnitude for forecast: '+ str(countTot))
        axb[0].set_ylabel('Error (Actual/Predicted)')
        axb[0].set_title('Technologies available: ' + str(trainErr['Tech'].nunique())+
                        '\n Total points with '+ str(trainingOrdMag)+ ' orders of magnitude for training '+
                        ' and '+ str(forecastOrdMag)+' orders of magnitude for forecast: '+ str(countTot))
        ax[0].plot([0,10**10],[1,1],'k', zorder=-10)
        ax[0].set_xlim(10**-trainingOrdMag, 10**forecastOrdMag)
        ax[1].annotate('Forecast', xy=(10**(+forecastOrdMag/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
        axb[1].annotate('Forecast', xy=(10**(+forecastOrdMag/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
        ax[1].set_ylim(0.1,10)
        ax[1].set_yscale('log', base=10)
        ax[1].set_xscale('log', base=10)
        ax[1].set_ylabel('Error (Actual/Predicted)')
        axb[1].set_ylabel('Error (Actual/Predicted)')
        ax[1].plot([0,10**10],[1,1],'k', zorder=-10)
        ax[1].set_xlim(10**-trainingOrdMag, 10**forecastOrdMag)

        ax[2].annotate('Training', xy=(10**(-trainingOrdMag/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
        ax[2].annotate('Forecast', xy=(10**(+forecastOrdMag/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
        axb[2].annotate('Training', xy=(10**(-trainingOrdMag/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
        axb[2].annotate('Forecast', xy=(10**(+forecastOrdMag/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
        ax[2].set_xscale('log', base=10)
        ax[2].set_xlabel('Predicted cumulative production / Current cumulative production')
        ax[2].set_ylabel('Number of points to estimate error')
        axb[2].set_xlabel('Predicted cumulative production / Current cumulative production')
        axb[2].set_ylabel('Number of points to estimate error')
        ax[2].set_xlim(10**-trainingOrdMag,10**forecastOrdMag)
        countPoints = np.array(countPoints)
        ax[2].set_ylim(0,max(max(countPoints[:,0])*1.1, ax[2].get_ylim()[1]))

        legend_elements = [
                            matplotlib.lines.Line2D([0], [0], color='b', lw=2, label='Training error'),
                            matplotlib.lines.Line2D([0], [0], color=cmapp(0.7), lw=2, label='Forecast error - Technology-specific'),
                            matplotlib.lines.Line2D([0], [0], color=cmapg(0.7), lw=2, label='Forecast error - Average slope'),
                            ]

        fig.legend(handles=legend_elements, loc='lower center', ncol=3)

        axes1=fig.add_axes([0.875,0.415,0.12,0.2])
        axes1.plot([0,1],[0.5,0.5],'k', lw=2)
        axes1.fill_between([0,1],[0.25,0.25],[0.75,0.75], color='k', alpha=0.3)
        axes1.fill_between([0,1],[0.1,0.1],[0.9,0.9], color='k', alpha=0.3)
        axes1.fill_between([0,1],[0,0],[1.0,1.0], color='k', alpha=0.1)
        axes1.annotate('10th percentile', xy=(3.0, 0.1), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.annotate('25th percentile', xy=(3.0, 0.25), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.annotate('Median', xy=(3.0, 0.5), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.annotate('75th percentile', xy=(3.0, 0.75), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.annotate('90th percentile', xy=(3.0, 0.9), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.annotate('Max', xy=(3.0, 1.0), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.annotate('Min', xy=(3.0, 0.0), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.set_xlim(-1,5)
        axes1.set_ylim(-0.2,1.2)
        axes1.set_xticks([])
        axes1.set_yticks([])
        axes1.axis('off')

        fig.subplots_adjust(top=0.92, bottom=0.11, right=0.85)
        
        for x in forecastIntAxis[1:]:
            stats1 = {}
            labs = ['whislo', 'q1', 'med', 'q3', 'whishi']
            count_ = 1
            for l in labs:
                stats1[l] = 10**pctTech[forecastIntAxis.index(x),count_]
                count_ += 1
                if count_ == 2 or count_ == 5:
                    count_ += 1
            axb[0].bxp([stats1], positions = [10**x], widths = (10**x)/8, showfliers=False, boxprops=dict(color=cmapp(0.7), lw=2), manage_ticks=False)
            stats2 = {}
            count_ = 1
            for l in labs:
                stats2[l] = 10**pctAvg[forecastIntAxis.index(x),count_]
                count_ += 1
                if count_ == 2:
                    count_ += 1
            axb[1].bxp([stats2], positions = [10**x], widths = (10**x)/8, showfliers=False, boxprops=dict(color=cmapg(0.7), lw=2), manage_ticks=False)
        axb[1].set_yscale('log', base=10)
        axb[1].set_ylim(0.1,10)
        axb[0].set_ylim(0.1,10)
        axb[0].plot([0,10**10],[1,1],'k', zorder=-10)
        axb[1].plot([0,10**10],[1,1],'k', zorder=-10)
        axb[2].plot([1,1],[0,100**100],'k')
        axb[2].plot([10**x for x in forecastIntAxis],countPoints, color='k', lw=2)
        axb[2].set_ylim(0,max(np.asarray(countPoints)[:,0])*1.1)
        ax2b.plot([10**x for x in forecastIntAxis],countTechs, color='red', lw=2)
        axb[1].set_xlim(10**-trainingOrdMag, 10**forecastOrdMag)
        axb[2].set_ylim(0,max(max(countPoints[:,0])*1.1, ax[2].get_ylim()[1]))

        figb.legend(handles=legend_elements, loc='lower center', ncol=3)

        axes1=figb.add_axes([0.875,0.415,0.12,0.2])
        axes1.plot([0,1],[0.5,0.5],'k', lw=1)
        axes1.plot([0,1,1,0,0],[0.25,0.25,0.75,0.75,0.25], lw=2, color='k')
        # axes1.fill_between([0,1],[0.25,0.25],[0.75,0.75], color='k', alpha=0.3)
        # axes1.fill_between([0,1],[0.1,0.1],[0.9,0.9], color='k', alpha=0.3)
        axes1.plot([0,1],[0,0], color='k', lw=1)
        axes1.plot([0,1],[1,1], color='k', lw=1)
        axes1.plot([0.5,0.5],[0,0.25], color='k', lw=1)
        axes1.plot([0.5,0.5],[0.75,1], color='k', lw=1)
        axes1.annotate('25th percentile', xy=(3.0, 0.25), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.annotate('Median', xy=(3.0, 0.5), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.annotate('75th percentile', xy=(3.0, 0.75), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.annotate('Max', xy=(3.0, 1.0), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.annotate('Min', xy=(3.0, 0.0), xycoords='data', ha='center', va='center', fontsize=7)
        axes1.set_xlim(-1,5)
        axes1.set_ylim(-0.2,1.2)
        axes1.set_xticks([])
        axes1.set_yticks([])
        axes1.axis('off')

        figb.subplots_adjust(top=0.92, bottom=0.11, right=0.85)


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
                                if method=='regression':
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