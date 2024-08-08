import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib, os, scipy
import statsmodels.api as sm
import piecewise_regression as pw

sns.set_style("ticks")
sns.set_context("talk")
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300

boxplot = True

# read original li-ion battery data 
price = pd.read_excel(\
    '..' + os.path.sep + \
    'MicahTrancik' + os.path.sep + \
    'LiIonDataSeries_represonly_withcover.xlsx',
                   sheet_name='RepreSeries_Price_All_Cells')
price = price[['IndependentAxisData','DependentAxisData']]
price.columns = ['Time (Year)','Unit cost (2018 USD/kWh)']
prod = pd.read_excel(\
    '../MicahTrancik/LiIonDataSeries_represonly_withcover.xlsx',
                   sheet_name='RepreSeries_MarketSize_All_MWh')
prod = prod[['IndependentAxisData','DependentAxisData']]
prod.columns = ['Time (Year)','Cumulative production (MWh)']
prod = prod[['Time (Year)','Cumulative production (MWh)']]

df = pd.merge(price, prod, on='Time (Year)').reset_index(drop=True)
df['Production (MWh)'] = df['Cumulative production (MWh)'].diff()
df = df[['Unit cost (2018 USD/kWh)', 'Time (Year)', 
         'Production (MWh)','Cumulative production (MWh)']]

df.columns = ['Unit cost', 'Year', 'Production', 'Cumulative production']

cost = df['Unit cost'].values
prod = df['Cumulative production'].values

# fit piecewise to updated data
nexp = 10 # number of experiments to eqvaluate best models
# empty lists to store results
best = []
breaks = []
lexps = []
intercepts = []
metrics = []

# create figure to visualize fits
plt.figure()
plt.scatter(np.log10(prod), np.log10(cost), marker='o')
res = sm.OLS(np.log10(cost), sm.add_constant(np.log10(prod))).fit()
plt.plot(np.log10(prod), res.params[0] + res.params[1]*np.log10(prod))

# repeat for number of experiments
for i in range(nexp):

    # perform model selection 
    # (i.e. find best continuous piecewise linear)
    # with an increasing number of breakpoints
    # whose locations are to be optimized
    pwfit = pw.ModelSelection(np.log10(prod), np.log10(cost), verbose=False,
                              min_distance_between_breakpoints = \
                                min(0.99, \
                                    np.log10(2)/np.log10(prod[-1]/prod[0])),
                              min_distance_to_edge = \
                                min(0.99, \
                                    np.log10(2)/np.log10(prod[-1]/prod[0])))

    # save bayesian information criterion
    bics = [pwfit.models[i].get_results()['bic'] \
                for i in range(len(pwfit.models))]

    # if no bic is available for a model, set it to infinity
    for i in range(len(bics)):
        if bics[i] == None:
            bics[i] = np.inf
    
    # plot models
    pwfit.models[np.argmin(bics)].plot_fit()
    # select model with lowest bic
    best.append(np.argmin(bics)+1)
    # get breakpoints
    breaks.append([np.log10(prod[0]), 
                * [pwfit.models[np.argmin(bics)].get_results()\
                        ['estimates']['breakpoint'+str(i)]['estimate']
                    for i in range(1,np.argmin(bics)+2)],
                np.log10(prod[-1])])
    # get learning exponents
    l = [pwfit.models[np.argmin(bics)].get_results()\
                                ['estimates']['alpha1']['estimate']]
    [l.append(x) for x in [pwfit.models[np.argmin(bics)].get_results()\
                                ['estimates']['beta'+str(i)]['estimate']
               for i in range(1,np.argmin(bics)+2)]]
    lexps.append(l)
    # get intercept
    intercepts.append(pwfit.models[np.argmin(bics)].get_results()\
                                ['estimates']['const']['estimate'])
    # get bic and rss
    metrics.append([pwfit.models[np.argmin(bics)].get_results()['bic'],
                    pwfit.models[np.argmin(bics)].get_results()['rss']])

# convert lists to dataframes
best = pd.DataFrame(best)
breaks = pd.DataFrame(breaks)
lexps = pd.DataFrame(lexps)
intercepts = pd.DataFrame(intercepts)
metrics = pd.DataFrame(metrics)

# put all dataframes together
best = pd.concat([best, breaks, lexps, intercepts, metrics], axis=1)

best.columns = ['Segments'] + \
                ['Initial production'] + \
                ['Breakpoint '+str(i) for i in range(1,len(breaks.columns)-1)] +\
                ['Final production'] + \
                ['LEXP '+str(i) for i in range(1,len(lexps.columns)+1)] + \
                ['Intercept','BIC', 'RSS']

## sort breakpoints and learning rates
for i in range(best.shape[0]):
    bps = best.loc[i,['Breakpoint '+str(i) for i in range(1, len(breaks.columns)-1)]].values
    lrs = best.loc[i,['LEXP '+str(i) for i in range(1, len(lexps.columns)+1)]].values[1:]
    idx = np.argsort(bps)
    lrs = lrs[idx]
    bps = bps[idx]
    best.loc[i,['Breakpoint '+str(i) for i in range(1, len(breaks.columns)-1)]] = bps
    best.loc[i,['LEXP '+str(i) for i in range(2, len(lexps.columns)+1)]] = lrs
    best.loc[i, ['LEXP '+str(i) for i in range(1, len(lexps.columns)+1)]] = \
        best.loc[i, ['LEXP '+str(i) for i in range(1, len(lexps.columns)+1)]].cumsum()

# figure: what number of breaks is best?
plt.figure()
plt.hist(best['Segments'].values)

# are the breakpoints similar across experiments?
plt.figure()
plt.scatter(prod, cost, marker='o')
for i in range(best.shape[0]):
    seg = best.loc[i,['Breakpoint '+str(i) \
                      for i in range(1,len(breaks.columns)-1)]].values
    for s in seg:
        plt.axvline(10**s)
plt.xscale('log')
plt.yscale('log')

## build projection

## plot data available
fig, ax = plt.subplots(figsize=(7,7))

ax.scatter(prod, cost, marker='o', 
           edgecolors='#EF476F', facecolors='none', 
           label='Observations (1991-2016)')
ax.set_xscale('log')
ax.set_yscale('log')

# select best model
IC = best.loc[best['RSS']==best['RSS'].min(), \
              best.columns[1:]].reset_index(drop=True)

## plot piecewise linear fit
yi = []
seg = 0
for xi in np.arange(np.log10(prod[0]),
                    np.log10(prod[-1]),
                    (np.log10(prod[-1]) -
                        np.log10(prod[0]))/1000):
    if 'Breakpoint '+str(seg+1) not in IC.columns or \
        xi < IC['Breakpoint '+str(seg+1)].values[0]:
        if seg == 0:
            yi.append(IC['Intercept'].values[0] + 
                        IC['LEXP 1'].values[0] * xi)
        else:
            yi.append(costbp + \
                        IC['LEXP '+str(seg+1)].values[0] * \
                            (xi - IC['Breakpoint '+str(seg)].values[0]))
    else:
        if seg == 0:
            costbp = IC['Intercept'].values[0] + \
                        IC['LEXP 1'].values[0] * \
                            IC['Breakpoint 1'].values[0]
        else:
            costbp = costbp + \
                        IC['LEXP '+str(seg+1)].values[0] * \
                            (IC['Breakpoint '+str(seg+1)].values[0] - \
                                IC['Breakpoint '+str(seg)].values[0])
        seg += 1
        yi.append(costbp + \
                    IC['LEXP '+str(seg+1)].values[0] * \
                        (xi - IC['Breakpoint '+str(seg)].values[0]))

yi = 10**np.array(yi)
xi = 10**np.array([xi for xi in np.arange(np.log10(prod[0]),
                    np.log10(prod[-1]),
                    (np.log10(prod[-1]) -
                        np.log10(prod[0]))/1000)])

ax.plot(xi, yi, color='#6a4c93')
ax.set_xlabel('Cumulative production (MWh)')
ax.set_ylabel('Unit cost (2018 USD/kWh)')

# model errors of piecewise linear
resid = []
seg = 0
yi = []
for xi, yobsi in zip(np.log10(prod), np.log10(cost)):
    if 'Breakpoint '+str(seg+1) not in IC.columns or \
        xi < IC['Breakpoint '+str(seg+1)].values[0]:
        if seg == 0:
            yi.append(IC['Intercept'].values[0] + 
                        IC['LEXP 1'].values[0] * xi)
        else:
            yi.append(costbp + \
                        IC['LEXP '+str(seg+1)].values[0] * \
                            (xi - IC['Breakpoint '+str(seg)].values[0]))
            resid.append(yi[-1] - yobsi)
    else:
        if seg == 0:
            costbp = IC['Intercept'].values[0] + \
                        IC['LEXP 1'].values[0] * \
                            IC['Breakpoint 1'].values[0]
        else:
            costbp = costbp + \
                        IC['LEXP '+str(seg+1)].values[0] * \
                            (IC['Breakpoint '+str(seg+1)].values[0] - \
                                IC['Breakpoint '+str(seg)].values[0])
        seg += 1
        yi.append(costbp + \
                    IC['LEXP '+str(seg+1)].values[0] * \
                        (xi - IC['Breakpoint '+str(seg)].values[0]))
        resid.append(yi[-1] - yobsi)

resid = np.array(resid)
modelerr = sm.tsa.SARIMAX(resid, order=(1,0,0)).fit()
print(modelerr.summary())
modelerr = sm.tsa.SARIMAX(resid, order=(0,0,0)).fit()
print(modelerr.summary())
modelerrp = modelerr.params
# no autocorrelation is better (lower BIC)
modelerrp = np.insert(modelerrp, 0, 0)

# set parameters of distribution of distance betweeen breakpoints
params = [0.696151716033671, 0.03809062707909341, 0.640013969109441]
dist_breaks = scipy.stats.lognorm(*params)
ar1 = 0.0
sigma = 0.4370592844181447

## simulate future costs
proj = []
nsim = 10000
# project for 4 order of magnitude increase in cumulative production
horizon = 10**((np.log10(prod[-1]) - np.log10(prod[-2])) * (2051-2017))
fut_prod = np.arange(0, np.log10(horizon), 
                     np.log10(prod[-1]) - np.log10(prod[-2]))

# starting from last observation
starting_point = 1
fut_prod += np.log10(prod[-starting_point])

for s in range(nsim):
    # get last cost at breakpoint
    costbp = IC['Intercept'].values[0] + \
                IC['LEXP 1'].values[0] * \
                    IC['Breakpoint 1'].values[0] + \
                sum(
                    [IC['LEXP '+str(i+1)].values[0] * \
                        (IC['Breakpoint '+str(i+1)].values[0] - \
                            IC['Breakpoint '+str(i)].values[0])
                    for i in range(1,len(lrs)) if IC['Breakpoint '+str(i+1)].values[0]<fut_prod[0]]
                )

    seg = 0
    while 'Breakpoint '+str(seg+1) in IC.columns and IC['Breakpoint '+str(seg+1)].values[0] < fut_prod[0]:
        seg += 1
    # sample next breakpoint
    next_bp = IC['Breakpoint '+str(seg)].values[0] + np.fmax(np.log10(2),dist_breaks.rvs(size=1)[0])
    next_bp = np.fmax(next_bp, np.log10(prod[-starting_point])-np.log10(2))
    last_bp = IC['Breakpoint '+str(seg)].values[0] * 1.0
    lr = IC['LEXP '+str(seg+1)].values[0]
    lr_change = IC['LEXP '+str(seg+1)].values[0] - IC['LEXP '+str(seg)].values[0]
    yi = []
    error = np.random.normal(0,1) * modelerrp[1]
    for xi in fut_prod:
        error = np.random.normal(0,1) * modelerrp[1] + modelerrp[0] * error
        if xi < next_bp:
            yi.append(costbp + lr * (xi - last_bp) + error)
        else:
            costbp = costbp + lr * (next_bp - last_bp)
            last_bp = next_bp * 1.0
            next_bp = next_bp + np.fmax(np.log10(2),dist_breaks.rvs(size=1)[0])
            lr_change = ar1 * lr_change + sigma * np.random.randn()
            lr = lr + lr_change
            lr = np.random.normal(-0.33, 0.36)
            yi.append(costbp + lr * (xi - last_bp) + error)
    proj.append(yi)

proj = np.array(proj)

proj = proj[:,fut_prod>np.log10(prod[-starting_point])-np.log10(2)]
fut_prod = fut_prod[fut_prod>np.log10(prod[-starting_point])-np.log10(2)]

if boxplot:
    bproj = proj[:,[2030-2017,-1]]
    bfut_prod = fut_prod[[2030-2017,-1]]

    for v, pos in zip(bproj.T, bfut_prod-0.125):
        plt.plot([10**(pos-0.1), 10**(pos+0.1)],
                 [10**np.percentile(v, 50, axis=0),
                  10**np.percentile(v, 50, axis=0)],
                    color='#6a4c93')
        plt.plot([10**(pos-0.1),10**(pos-0.1), 
                  10**(pos+0.1),10**(pos+0.1),
                  10**(pos-0.1)],
                 [10**np.percentile(v, 25, axis=0),
                  10**np.percentile(v, 75, axis=0),
                  10**np.percentile(v, 75, axis=0),
                  10**np.percentile(v, 25, axis=0),
                  10**np.percentile(v, 25, axis=0)
                  ],
                    color='#6a4c93')
        plt.fill_between([10**(pos-0.1), 10**(pos+0.1)],
                 10**np.percentile(v, 25, axis=0),
                 10**np.percentile(v, 75, axis=0),
                    color='#6a4c93', alpha=0.1)
        plt.plot([10**pos, 10**pos],
                [10**np.percentile(v, 75, axis=0),
                10**np.percentile(v, 95, axis=0)],
                    color='#6a4c93')
        plt.plot([10**(pos-.1),10**(pos+.1)],
                [10**np.percentile(v, 95, axis=0),
                10**np.percentile(v, 95, axis=0)],
                    color='#6a4c93')
        
        plt.plot([10**(pos), 10**pos],
                [10**np.percentile(v, 25, axis=0),
                10**np.percentile(v, 5, axis=0)],
                    color='#6a4c93')
        plt.plot([10**(pos-.1), 10**(pos+.1)],
                 [10**np.percentile(v, 5, axis=0),
                  10**np.percentile(v, 5, axis=0)],
                    color='#6a4c93')

# ax.fill_between(10**fut_prod, 10**np.percentile(proj, 5, axis=0),
#                 10**np.percentile(proj, 95, axis=0), 
#                 color='#6a4c93', alpha=0.05, zorder=-10)
# ax.fill_between(10**fut_prod, 10**np.percentile(proj, 25, axis=0),
#                 10**np.percentile(proj, 75, axis=0),
#                 color='#6a4c93', alpha=0.1, zorder=-10)
# ax.plot(10**fut_prod, 10**np.percentile(proj, 5, axis=0), 
#         color='#6a4c93', ls=':',
#         zorder=-8,
#         # alpha=.25,
#         lw=1
#         )
# ax.plot(10**fut_prod, 10**np.percentile(proj, 95, axis=0), 
#         color='#6a4c93', ls=':', 
#         zorder=-8,
#         # alpha=.25,
#         lw=1
#         )
# ax.plot(10**fut_prod, 10**np.percentile(proj, 25, axis=0), 
#         color='#6a4c93', ls='--', 
#         zorder=-8,
#         lw=1,
#         # alpha=.5
#         )
# ax.plot(10**fut_prod, 10**np.percentile(proj, 75, axis=0), 
#         color='#6a4c93', ls='--', 
#         zorder=-8, lw=1,
#         # alpha=.5
#         )

ax.plot(10**fut_prod, 10**np.median(proj, axis=0), '#6a4c93', lw=1,
        label='Piecewise linear experience curve',
        zorder=-5
        )

## overlay first difference wright's law projection

cost_d = np.diff(np.log10(cost))
prod_d = np.diff(np.log10(prod))

slope = np.sum(cost_d * prod_d) / np.sum(prod_d**2)
# params from Way et al., 2022
slope = -0.421

residuals = []
for i in range(len(cost_d)):
    residuals.append(cost_d[i] - slope * prod_d[i])

residuals = np.array(residuals)

# build ar1 model of residuals

res = sm.tsa.SARIMAX(residuals, order=(1,0,0)).fit()
print(res.summary())
ar1 = res.params[0]
sigma = res.params[1]
# params from Way et al., 2022
ar1 = .19
sigma = 0.103

## simulate future costs
proj = []
nsim = 10000

for s in range(nsim):
    yi = [np.log10(cost[-starting_point])]
    residual = residuals[-1]

    for xi in np.diff(fut_prod):
        residual = ar1 * residual + sigma * np.random.randn()
        yi.append(yi[-1] + slope * xi + residual)

    proj.append(yi)


proj = np.array(proj)

if boxplot:
    bproj = proj[:,[2030-2017,-1]]
    bfut_prod = fut_prod[[2030-2017,-1]]

    for v, pos in zip(bproj.T, bfut_prod+0.125):
        plt.plot([10**(pos-0.1), 10**(pos+0.1)],
                 [10**np.percentile(v, 50, axis=0),
                  10**np.percentile(v, 50, axis=0)],
                    color='#1982c4')
        plt.plot([10**(pos-0.1),10**(pos-0.1), 
                  10**(pos+0.1),10**(pos+0.1),
                  10**(pos-0.1)],
                 [10**np.percentile(v, 25, axis=0),
                  10**np.percentile(v, 75, axis=0),
                  10**np.percentile(v, 75, axis=0),
                  10**np.percentile(v, 25, axis=0),
                  10**np.percentile(v, 25, axis=0)
                  ],
                    color='#1982c4')
        plt.fill_between([10**(pos-0.1), 10**(pos+0.1)],
                 10**np.percentile(v, 25, axis=0),
                 10**np.percentile(v, 75, axis=0),
                    color='#1982c4', alpha=0.1)
        plt.plot([10**pos, 10**pos],
                [10**np.percentile(v, 75, axis=0),
                10**np.percentile(v, 95, axis=0)],
                    color='#1982c4')
        plt.plot([10**(pos-.1),10**(pos+.1)],
                [10**np.percentile(v, 95, axis=0),
                10**np.percentile(v, 95, axis=0)],
                    color='#1982c4')
        
        plt.plot([10**(pos), 10**pos],
                [10**np.percentile(v, 25, axis=0),
                10**np.percentile(v, 5, axis=0)],
                    color='#1982c4')
        plt.plot([10**(pos-.1), 10**(pos+.1)],
                 [10**np.percentile(v, 5, axis=0),
                  10**np.percentile(v, 5, axis=0)],
                    color='#1982c4')

# ax.fill_between(10**fut_prod, 10**np.percentile(proj, 5, axis=0),
#                 10**np.percentile(proj, 95, axis=0), 
#                 color='#1982c4', alpha=0.05, zorder=-10)
# ax.fill_between(10**fut_prod, 10**np.percentile(proj, 25, axis=0),
#                 10**np.percentile(proj, 75, axis=0), 
#                 color='#1982c4', alpha=0.1, zorder=-10)
# ax.plot(10**fut_prod, 10**np.percentile(proj, 5, axis=0), 
#         color='#1982c4', ls=':', 
#         zorder=-8,
#         # alpha=.25,
#         lw=1
#         )
# ax.plot(10**fut_prod, 10**np.percentile(proj, 95, axis=0), 
#         color='#1982c4', ls=':', 
#         zorder=-8,
#         # alpha=.25,
#         lw=1
#         )
# ax.plot(10**fut_prod, 10**np.percentile(proj, 25, axis=0), 
#         color='#1982c4', ls='--', 
#         zorder=-8,
#         # alpha=.5,
#         lw=1
#         )
# ax.plot(10**fut_prod, 10**np.percentile(proj, 75, axis=0), 
#         color='#1982c4', ls='--', 
#         zorder=-8,
#         # alpha=.5,
#         lw=1
#         )

ax.plot(10**fut_prod, 10**np.median(proj, axis=0), '#1982c4', lw=1,
        label='First difference Wright\'s law',
        zorder=-5
        )

ax.minorticks_off()

plt.legend(loc='lower left')
plt.title('Li-ion Battery')
plt.tight_layout()

axes = fig.add_axes([0.3, 0.4, 0.15, 0.25])

axes.axis('off')
axes.fill_between([0.5,1], [0,0], [1,1], color='#47666F', alpha=.2, lw=0)
axes.fill_between([0.5,1], [0.2,0.2], [0.8,0.8], color='#47666F', alpha=.4, lw=0)
axes.plot([0.5,1], [0.5,0.5], color='#47666F', lw=2)
axes.plot([0.5,1], [0.8,0.8], color='#47666F', ls='--', lw=2)
axes.plot([0.5,1], [0.2,0.2], color='#47666F', ls='--', lw=2)
axes.plot([0.5,1], [1,1], color='#47666F', ls=':', lw=2)
axes.plot([0.5,1], [0,0], color='#47666F', ls=':', lw=2)
axes.annotate('90%', xy=(1.5,1.05), xycoords='data', 
            ha='center', va='center', color='k',
            fontsize=12
            )
axes.annotate('50%', xy=(1.5,0.85), xycoords='data',
            ha='center', va='center', color='k',
            fontsize=12
            )
axes.annotate('Median', xy=(1.5,0.5), xycoords='data',
            ha='center', va='center', color='k',
            fontsize=12
            )

axes.plot([1.1,2,2,1.1], [0.2,0.2,0.8,0.8], color='k', lw=.2)
axes.plot([1.1,2.1,2.1,1.1], [0,0,1,1], color='k', lw=.2)

plt.gcf().savefig('figs' + os.path.sep + 'BatteryProjection.png')

plt.show()