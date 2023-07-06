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


trainingOrdMag = np.log10(10)
forecastOrdMag = np.log10(100)
samplingPoints = 10
counterr = 0
count = 0
countTot = 0
trainErr, dferrTech, dferrAvg = [], [], []
for tech in df['Tech'].unique():
    # computing average technological slope based on all other technologies
    slopeall = np.mean(slopes.loc[slopes['Tech'] != tech,'Slope'].values)
    # computing technology specific slope
    sel = df.loc[df['Tech']==tech].copy()
    print(tech, x[-1] - x[0], x[-1] - x[0]>trainingOrdMag+forecastOrdMag )
    if x[-1] - x[0] > trainingOrdMag+forecastOrdMag:
        count += 1
    x = np.log10(sel['Cumulative production'].values)
    y = np.log10(sel['Unit cost'].values)
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
                        pred =  y[i] + slope * (x[i+1:M+1] - x[i])
                    # pred =  y[i] + slope * (x[M] - x[i])
                    # if method=='regression':
                    # 	pred = result.params[0] + slope * x[M]
                        pred2 =  y[i] + slopeall * (x[i+1:M+1] - x[i])
                    # pred2 =  y[i] + slopeall * (x[M] - x[i])
                        error = (y[i+1:M+1] - (pred)) 
                        error2 = (y[i+1:M+1] - (pred2)) 
                    # error = (y[M] - (pred)) 
                    # error2 = (y[M] - (pred2)) 
                    # dferrTech.append([x[M] - x[i], error, tech])
                    # dferrAvg.append([x[M] - x[i], error2, tech])
                        for idx in range(len(error)):
                            dferrTech.append([x[i+1+idx] - x[i], error[idx], tech])
                            dferrAvg.append([x[i+1+idx] - x[i], error2[idx], tech])
                        flag = 1
                        countTot += 1
                            # plt.scatter(x[N:M+1], y[N:M+1], color='r', zorder=10)
                            # plt.scatter(x[N:i+1], result.predict(sm.add_constant(x[N:i+1])), color='b')
                            # plt.scatter(x[i+1:M+1], pred, color='purple')
                            # plt.scatter(x[i+1:M+1], pred2, color='green')
                            # plt.show()

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

plt.plot(trainErr['Forecast horizon'],'o')
plt.plot(dferrTech['Forecast horizon'],'o')
plt.plot(dferrAvg['Forecast horizon'],'o')
plt.show()

fig, ax = plt.subplots(3,1, sharex=True)
# plot training error
npoints = int(samplingPoints * trainingOrdMag)
trainInt = np.linspace(-trainingOrdMag-trainingOrdMag/npoints/2, 0+ trainingOrdMag/npoints/2, npoints+2)
countPoints = []
pctTrain = []
for i in range(len(trainInt)-1):
    if i == 0:
        sel = trainErr.loc[(trainErr['Forecast horizon']<=trainInt[i+1])].copy()
    else:
        sel = trainErr.loc[(trainErr['Forecast horizon']>trainInt[i]) &\
                       (trainErr['Forecast horizon']<=trainInt[i+1])].copy()
    if sel.shape[0] == 0:
        pctTrain.append([trainInt[i],np.nan,np.nan,np.nan,np.nan,np.nan])
        countPoints.append(0)
        continue
    countPoints.append(sel.shape[0])
    for tt in sel['Tech'].unique():
        sel.loc[sel['Tech']==tt,'Weights'] = 1/sel.loc[sel['Tech']==tt].count()[0]
    sel = sel.sort_values(by='Error', ascending=True)
    cumsum = sel['Weights'].cumsum().round(4)
    pt = []
    for q in [0,25,50,75,100]:
    # for q in [0,10,20,30,40,50,60,70,80,90,100]:
        cutoff = sel['Weights'].sum() * q/100
        pt.append(sel['Error'][cumsum >= cutoff.round(4)].iloc[0])
    pctTrain.append([trainInt[i],*pt])
pctTrain = np.array(pctTrain)
ax[0].plot([10**(x+(trainInt[1]-trainInt[0])/2) for x in trainInt[:-1]],10**pctTrain[:,3], color='b', lw=2)
ax[1].plot([10**(x+(trainInt[1]-trainInt[0])/2) for x in trainInt[:-1]],10**pctTrain[:,3], color='b', lw=2)
for r in range(1,-1,-1):
    ax[0].fill_between([10**(x+(trainInt[1]-trainInt[0])/2)  for x in trainInt[:-1]], 10**pctTrain[:,1+r], 10**pctTrain[:,-r-1], alpha=0.1+0.2*r, color='b', zorder=-2-r, lw=0)
    ax[1].fill_between([10**(x+(trainInt[1]-trainInt[0])/2)  for x in trainInt[:-1]], 10**pctTrain[:,1+r], 10**pctTrain[:,-r-1], alpha=0.1+0.2*r, color='b', zorder=-2-r, lw=0)
ax[2].plot([10**(x+(trainInt[1]-trainInt[0])/2)  for x in trainInt[:-1]],countPoints, color='k', lw=2)

# plot forecast error
npoints = int(samplingPoints * forecastOrdMag)
forecastInt = np.linspace(0-forecastOrdMag/npoints/2, forecastOrdMag+forecastOrdMag/npoints/2, npoints+2)
pctTech, pctAvg = [], []
countPoints = []
for i in range(len(forecastInt)-1):
    if i == 0:
        sel1 = dferrTech.loc[(dferrTech['Forecast horizon']<=forecastInt[i+1])].copy()
        sel2 = dferrAvg.loc[(dferrAvg['Forecast horizon']<=forecastInt[i+1])].copy()
    else:
        sel1 = dferrTech.loc[(dferrTech['Forecast horizon']>forecastInt[i]) &\
                       (dferrTech['Forecast horizon']<=forecastInt[i+1])].copy()
        sel2 = dferrAvg.loc[(dferrAvg['Forecast horizon']>forecastInt[i]) &\
                       (dferrAvg['Forecast horizon']<=forecastInt[i+1])].copy()
    if sel1.shape[0] == 0:
        pctTech.append([forecastInt[i],np.nan,np.nan,np.nan,np.nan,np.nan])
        pctAvg.append([forecastInt[i],np.nan,np.nan,np.nan,np.nan,np.nan])
        countPoints.append([0,0])
        continue
    countPoints.append([sel1.shape[0],sel2.shape[0]])
    for tt in sel1['Tech'].unique():
        sel1.loc[sel1['Tech']==tt,'Weights'] = 1/sel1.loc[sel1['Tech']==tt].count()[0]
    for tt in sel2['Tech'].unique():
        sel2.loc[sel2['Tech']==tt,'Weights'] = 1/sel2.loc[sel2['Tech']==tt].count()[0]
    sel1 = sel1.sort_values(by='Error', ascending=True)
    sel2 = sel2.sort_values(by='Error', ascending=True)
    cumsum1 = sel1['Weights'].cumsum().round(4)
    cumsum2 = sel2['Weights'].cumsum().round(4)
    pt1, pt2 = [], []
    for q in [0,25,50,75,100]:
    # for q in [0,10,20,30,40,50,60,70,80,90,100]:
        cutoff1 = sel1['Weights'].sum() * q/100
        cutoff2 = sel2['Weights'].sum() * q/100
        pt1.append(sel1['Error'][cumsum1 >= cutoff1.round(4)].iloc[0])
        pt2.append(sel2['Error'][cumsum2 >= cutoff2.round(4)].iloc[0])
    pctTech.append([forecastInt[i],*pt1])
    pctAvg.append([forecastInt[i],*pt2])
pctTech = np.array(pctTech)
pctAvg = np.array(pctAvg)
ax[0].plot([10**(x+(forecastInt[1]-forecastInt[0])/2) for x in forecastInt[:-1]],10**pctTech[:,3], color=cmapp(0.7), lw=2)
ax[1].plot([10**(x+(forecastInt[1]-forecastInt[0])/2) for x in forecastInt[:-1]],10**pctAvg[:,3], color=cmapg(0.7), lw=2)
for r in range(1,-1,-1):
    ax[0].fill_between([10**(x+(forecastInt[1]-forecastInt[0])/2)  for x in forecastInt[:-1]], 10**pctTech[:,1+r], 10**pctTech[:,-r-1], alpha=0.1+0.2*r, color=cmapp(0.7), zorder=-2-r, lw=0)
    ax[1].fill_between([10**(x+(forecastInt[1]-forecastInt[0])/2)  for x in forecastInt[:-1]], 10**pctAvg[:,1+r], 10**pctAvg[:,-r-1], alpha=0.1+0.2*r, color=cmapg(0.7), zorder=-2-r, lw=0)
ax[2].plot([10**(x+(forecastInt[1]-forecastInt[0])/2)  for x in forecastInt[:-1]], np.asarray(countPoints)[:,0], color='k', lw=2)

ax[0].plot([1,1],[0,100],'k')
ax[1].plot([1,1],[0,100],'k')
ax[2].plot([1,1],[0,100],'k')
# ax.fill_between([10**-trainingOrdMag,10**0],[0,0],[100,100], color='grey', alpha=0.1, zorder=-10)
ax[0].annotate('Training', xy=(10**(-trainingOrdMag/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
ax[0].annotate('Forecast', xy=(10**(+forecastOrdMag/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
ax[0].set_ylim(0.1,10)
ax[0].set_yscale('log', base=10)
ax[0].set_xscale('log', base=10)
# ax[0].set_xlabel('Predicted cumulative production / Current cumulative production')
ax[0].set_ylabel('Error (Actual/Predicted)')
ax[0].set_title('Technologies available: ' + str(trainErr['Tech'].nunique())+
                '\n Total points with '+ str(trainingOrdMag)+ ' orders of magnitude for training '+
                 ' and '+ str(forecastOrdMag)+' orders of magnitude for forecast: '+ str(countTot))
ax[0].plot([0,10**10],[1,1],'k', zorder=-10)
ax[0].set_xlim(10**-trainingOrdMag, 10**forecastOrdMag)

ax[1].annotate('Training', xy=(10**(-trainingOrdMag/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
ax[1].annotate('Forecast', xy=(10**(+forecastOrdMag/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
ax[1].set_ylim(0.1,10)
ax[1].set_yscale('log', base=10)
ax[1].set_xscale('log', base=10)
# ax[1].set_xlabel('Predicted cumulative production / Current cumulative production')
ax[1].set_ylabel('Error (Actual/Predicted)')
# ax[1].set_title('Number of technologies available: ' + str(trainErr['Tech'].nunique()))
ax[1].plot([0,10**10],[1,1],'k', zorder=-10)
ax[1].set_xlim(10**-trainingOrdMag, 10**forecastOrdMag)

ax[2].annotate('Training', xy=(10**(-trainingOrdMag/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
ax[2].annotate('Forecast', xy=(10**(+forecastOrdMag/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
ax[2].set_xscale('log', base=10)
ax[2].set_xlabel('Predicted cumulative production / Current cumulative production')
ax[2].set_ylabel('Number of points to estimate error')
ax[2].set_xlim(10**-trainingOrdMag,10**forecastOrdMag)

legend_elements = [
                    matplotlib.lines.Line2D([0], [0], color='b', lw=2, label='Training error'),
                    matplotlib.lines.Line2D([0], [0], color=cmapp(0.7), lw=2, label='Forecast error - Technology-specific)'),
                    matplotlib.lines.Line2D([0], [0], color=cmapg(0.7), lw=2, label='Forecast error - Average slope'),
                    ]

fig.legend(handles=legend_elements, loc='lower center', ncol=3)

axes1=fig.add_axes([0.91,0.4,0.08,0.2])
axes1.plot([0,1],[0.5,0.5],'k', lw=2)
axes1.fill_between([0,1],[0.25,0.25],[0.75,0.75], color='k', alpha=0.3)
axes1.fill_between([0,1],[0,0],[1.0,1.0], color='k', alpha=0.1)
axes1.annotate('25th percentile', xy=(3.0, 0.25), xycoords='data', ha='center', va='center', fontsize=7)
axes1.annotate('Median', xy=(3.0, 0.5), xycoords='data', ha='center', va='center', fontsize=7)
axes1.annotate('75th percentile', xy=(3.0, 0.75), xycoords='data', ha='center', va='center', fontsize=7)
axes1.annotate('Max', xy=(3.0, 1.0), xycoords='data', ha='center', va='center', fontsize=7)
axes1.annotate('Min', xy=(3.0, 0.0), xycoords='data', ha='center', va='center', fontsize=7)
axes1.set_xlim(-1,5)
axes1.set_ylim(-0.2,1.2)
axes1.set_xticks([])
axes1.set_yticks([])
# axes1.axis('off')

plt.subplots_adjust(top=0.92, bottom=0.11)
plt.show()