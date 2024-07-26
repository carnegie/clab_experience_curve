import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import statsmodels.api as sm
import utils


def get_stats(df, quantiles=[.05,.25,.50,.75,.95]):
    return df.quantile(quantiles)

sns.set_style("ticks")
sns.set_context("talk")
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300

# load data
df = pd.read_csv('ExpCurves.csv')

# add sectors
df['Sector'] = [utils.sectorsinv[x] for x in df['Tech']]

# set color sectors
sectors_colors = {'Chemicals':'#DE196B',
                    'Consumer goods':'#640FBF',
                    'Energy':'#FF9100',
                    'Food':'#048E2E',
                    'Genomics':'#632E0D',
                    'Hardware':'#1F92F0',
                    }
df['Color'] = df['Sector'].map(sectors_colors)

slopes = []

for tech in df['Tech'].unique():
    df_tech = df.loc[df['Tech'] == tech]

    x = np.log10(df_tech['Cumulative production'].values)
    y = np.log10(df_tech['Unit cost'].values)
    years = df_tech['Year'].values

    x_ = np.diff(x)
    y_ = np.diff(y)
    years_ = np.diff(years)
    res = sm.OLS(y_, x_).fit()

    slope = res.params[0]

    for i in range(3, len(x)-1):
        x_ = np.diff(x)[:i]
        y_ = np.diff(y)[:i]
        years_ = np.diff(years[:i])

        res = sm.OLS(y_, x_).fit()

        slopes.append([tech, res.params[0], 
                       slope, x[0], x[i], x[-1], 
                       years[0], years[i], years[-1]])

slopes = pd.DataFrame(slopes, columns=['Tech', 
                                       'Slope', 
                                       'Final estimate',
                                       'Initial cumulative production', 
                                       'Intermediate cumulative production',
                                        'Final cumulative production',
                                        'Initial year',
                                        'Intermediate year',
                                        'Final year'])

slopes['Learning exponent error'] = slopes['Slope'] - slopes['Final estimate']
slopes['Learning rate error'] = 100 * (1 - 2**slopes['Slope']) - \
                                100 * (1 - 2**slopes['Final estimate'])

slopes['Cumulative production relative to initial'] = \
    (slopes['Intermediate cumulative production'] - \
        slopes['Initial cumulative production']) 

slopes['Years from first observation'] = \
    (slopes['Intermediate year'] - \
        slopes['Initial year'])

windows = [np.log10(1+1/3)*4, 4]
steps = [np.log10(1+1/3), 1]

statsprod = []
countprod = []
xprod = np.arange(steps[0], max(slopes['Cumulative production relative to initial']),
                  steps[0])
statstime = []
counttime = []
xtime = np.arange(4, max(slopes['Years from first observation']),
                    steps[1])

fig, ax = plt.subplots(2, 2, figsize=(13,6), sharey='row', 
                       sharex='col', height_ratios=[0.1,0.9])

for i in range(xprod.shape[0]):
    sel = slopes.loc[
        (slopes['Cumulative production relative to initial'] >= xprod[i] - windows[0]) & \
        (slopes['Cumulative production relative to initial'] < xprod[i])]
    statsprod.append(get_stats(sel['Learning exponent error']).values)
    countprod.append(sel['Tech'].nunique())

for i in range(xtime.shape[0]):
    sel = slopes.loc[
        (slopes['Years from first observation'] >= xtime[i] - windows[1]) & \
        (slopes['Years from first observation'] < xtime[i])]
    statstime.append(get_stats(sel['Learning exponent error']).values)
    counttime.append(sel['Tech'].nunique())

ax[0][0].fill_between(10**xprod, 0, countprod, color='#57B8FF', alpha=0.3)
ax[0][1].fill_between(xtime, 0, counttime, color='#57B8FF', alpha=0.3)

ax[0][0].axis('off')
ax[0][1].axis('off')

ax[1][0].fill_between(10**xprod, np.array(statsprod)[:,0], np.array(statsprod)[:,4],
                   color='#57B8FF', alpha=0.1)
ax[1][0].fill_between(10**xprod, np.array(statsprod)[:,1], np.array(statsprod)[:,3],
                     color='#57B8FF', alpha=0.3)
ax[1][0].plot(10**xprod, np.array(statsprod)[:,2], color='#57B8FF')
ax[1][0].set_xscale('log')
ax[1][0].set_xlabel('Cumulative production relative to initial')
ax[1][0].set_ylabel('Learning exponent error')

ax[1][1].fill_between(xtime, np.array(statstime)[:,0], np.array(statstime)[:,4],
                     color='#57B8FF', alpha=0.1)
ax[1][1].fill_between(xtime, np.array(statstime)[:,1], np.array(statstime)[:,3],
                        color='#57B8FF', alpha=0.3)
ax[1][1].plot(xtime, np.array(statstime)[:,2], color='#57B8FF')

ax[1][1].set_xlabel('Years from first observation')
ax[1][1].set_ylabel('Learning exponent error')


plt.savefig('figs' + utils.os.path.sep + 'SupplementaryFigures' + \
                utils.os.path.sep + 'LearningExponentError.png')

fig, ax = plt.subplots(2, 2, figsize=(13,8), sharey='row', sharex='col',
                       height_ratios=[0.2,0.9])
statsprod, statstime = [], []
countprod, counttime = [], []
for i in range(xprod.shape[0]):
    sel = slopes.loc[
        (slopes['Cumulative production relative to initial'] >= xprod[i] - windows[0]) & \
        (slopes['Cumulative production relative to initial'] < xprod[i] )]
    for t in sel['Tech'].unique():
        sel.drop(sel.loc[sel['Tech'] == t].index[:-1], inplace=True)
    statsprod.append(get_stats(sel['Learning rate error']).values)
    countprod.append(sel['Tech'].nunique())

for i in range(xtime.shape[0]):
    sel = slopes.loc[
        (slopes['Years from first observation'] >= xtime[i] - windows[1]) & \
        (slopes['Years from first observation'] < xtime[i] )]
    for t in sel['Tech'].unique():
        sel.drop(sel.loc[sel['Tech'] == t].index[:-1], inplace=True)
    statstime.append(get_stats(sel['Learning rate error']).values)
    counttime.append(sel['Tech'].nunique())

ax[0][0].fill_between(10**xprod, 0, countprod, color='#57B8FF', alpha=0.3)
ax[0][1].fill_between(xtime, 0, counttime, color='#57B8FF', alpha=0.3)
# ax[0][0].axis('off')
# ax[0][1].axis('off')

ax[1][0].fill_between(10**xprod, np.array(statsprod)[:,0], np.array(statsprod)[:,4],
                   color='#57B8FF', alpha=0.1)
ax[1][0].fill_between(10**xprod, np.array(statsprod)[:,1], np.array(statsprod)[:,3],
                     color='#57B8FF', alpha=0.3)
ax[1][0].plot(10**xprod, np.array(statsprod)[:,2], color='#57B8FF')
ax[1][0].set_xscale('log')
ax[1][0].set_xlabel('Cumulative production relative to initial')
ax[1][0].set_ylabel('Learning rate error (%)')

ax[1][1].fill_between(xtime, np.array(statstime)[:,0], np.array(statstime)[:,4],
                     color='#57B8FF', alpha=0.1)
ax[1][1].fill_between(xtime, np.array(statstime)[:,1], np.array(statstime)[:,3],
                        color='#57B8FF', alpha=0.3)
ax[1][1].plot(xtime, np.array(statstime)[:,2], color='#57B8FF')

ax[1][1].set_xlabel('Years from first observation')
ax[1][1].set_ylabel('Learning rate error (%)')

ax[0][0].set_ylabel('Data series')
ax[0][0].set_ylim(0,90)

for axx in [ax[0][1], ax[1][1]]:
    axx.fill_between([100,110], [-25,-25], [-10,-10], 
                     color='#57B8FF', alpha=.2)
    axx.fill_between([100,110], [-22,-22], [-13,-13],
                     color='#57B8FF', alpha=.4)
    axx.plot([100,110], [-17.5,-17.5], color='#57B8FF', lw=2)
    axx.annotate('90%', xy=(120,-8.5), xycoords='data', 
                ha='center', va='center', color='k',
                fontsize=12
                )
    axx.annotate('50%', xy=(120,-12), xycoords='data', 
                ha='center', va='center', color='k',
                fontsize=12
                )
    axx.annotate('Median', xy=(120,-17.5), xycoords='data', 
                ha='center', va='center', color='k',
                fontsize=12
                )
    axx.plot([112,130,130,112], [-22,-22,-13,-13], color='k', lw=.2)
    axx.plot([112,135,135,112], [-25,-25,-10,-10], color='k', lw=.2)

plt.subplots_adjust(wspace=0.2, hspace=0.1, bottom=0.1, top=0.95)

plt.savefig('figs' + utils.os.path.sep + 'SupplementaryFigures' + \
                utils.os.path.sep + 'LearningRateError.png')

plt.show()




