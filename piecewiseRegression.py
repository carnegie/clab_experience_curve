import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import piecewise_regression as pw
import statsmodels.api as sm
import os, utils

# set figures' parameters
sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300
sns.color_palette('colorblind')

# set to true to plot a figure for each technology
plot_fig_tech = False

# set to True if the regression dataset needs to be built
build_pwreg_dataset = True

# first differencing
first_diff = False

# set the maximum number of breakpoints
max_breakpoints = 6

# set min distance between breakpoints
min_dist = np.log10(2)

# set random seed
np.random.seed(0)

try:
    if not(first_diff):
        IC = pd.read_csv('IC.csv')
    else:
        IC = pd.read_csv('IC_first_diff.csv')
except FileNotFoundError:
    build_pwreg_dataset = True

if build_pwreg_dataset:
    print('Performing piecewise linear regression',
            'on the experience curve dataset')

    # Load the data
    df = pd.read_csv('ExpCurves.csv')

    # perform continuous piecewise regression 
    # on each technology in the dataset
    # and return a dataframe containing
    # the parameters of each fit
    # and the information criteria
    IC = utils.build_piecewise_regression_dataset(df,
                                                    max_breakpoints,
                                                    min_dist,
                                                    first_diff,
                                                    plot_fig_tech)

# get the number of technologies 
AIC = IC.loc[IC.groupby('Tech')['AIC'].idxmin()]\
            .groupby('n_breaks').count()['Tech'].reset_index()
BIC = IC.loc[IC.groupby('Tech')['BIC'].idxmin()]\
            .groupby('n_breaks').count()['Tech'].reset_index()

# rename metrics
AIC['metric'] = 'Akaike'
BIC['metric'] = 'Bayesian'

# create new dataframe with tech counts and metrics
metrics = pd.concat([AIC, BIC]).reset_index(drop=True)

# rename columns and add number of segments
metrics.columns = ['n_breaks', 'Count', 'Metric']
metrics['Number of segments'] = metrics['n_breaks'] + 1

# plot the distribution of technologies over number of segments - bar plot
fig, ax = plt.subplots(1,1, figsize=(9,6), sharex=True, sharey=True)
sns.barplot(data=metrics, 
        x = 'Number of segments',
        y = 'Count',
        hue='Metric',
        ax = ax,
        legend=False)
ax.annotate('Akaike\nInformation\nCriterion', (-.25, 37.5),
            xycoords='data', color=sns.color_palette()[0],
            ha='center', va='center')
ax.annotate('Bayesian\nInformation\nCriterion', (2.25, 37.5),
            xycoords='data', color=sns.color_palette()[1],
            ha='center', va='center')

ax.set_xlim(-1,7)
ax.set_xlabel('Optimal number of segments')
ax.set_ylabel('Number of data series')

fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95)

if not os.path.exists('figs' + os.path.sep + 'SupplementaryFigures'):
    os.makedirs('figs' + os.path.sep + 'SupplementaryFigures')
fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                os.path.sep +
            'TechCountVsOptimalSegments' + \
                first_diff * '_first_diff' + '.png')
fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                os.path.sep +
            'TechCountVsOptimalSegments' + \
                first_diff * '_first_diff' + '.eps')

# plot the distribution of technologies over number of segments - pie chart
fig, ax = plt.subplots(1,2, figsize=(10,6))

metrics.loc[metrics['Metric']=='Akaike'].set_index('Number of segments')\
    .plot.pie(y='Count', autopct='%1.1f%%', 
                pctdistance = 1.25, labeldistance=175,
                ax=ax[0], legend=False, 
                label='',
                colors=sns.color_palette('Set2'))

metrics.loc[metrics['Metric']=='Bayesian'].set_index('Number of segments')\
    .plot.pie(y='Count', autopct='%1.1f%%', 
                pctdistance = 1.25, labeldistance=175,
                ax=ax[1], legend=False, 
                label='', colors=sns.color_palette('Set2'))

fig.legend(metrics['Number of segments'].unique(),
            title='Optimal number of segments',
            loc='lower center', ncol=6)

ax[0].set_title('Akaike Information Criterion')
ax[1].set_title('Bayesian Information Criterion')
fig.subplots_adjust(bottom=0.125, left=0.05, right=0.95)
plt.tight_layout()

fig.savefig('figs' + os.path.sep + \
            'PieSegments' + \
                first_diff * '_first_diff' + '.png')
fig.savefig('figs' + os.path.sep + \
            'PieSegments' + \
                first_diff * '_first_diff' + '.eps')

# plot the distribution of technologies over number of segments and sectors - pie chart
fig, ax = plt.subplots(1,2, figsize=(12.5,7))

size = .3

metrics.loc[metrics['Metric']=='Akaike'].set_index('Number of segments')\
    .plot.pie(y='Count', autopct='%1.1f%%', 
                pctdistance = 1.25, labeldistance=175,
                ax=ax[0], radius = 1,
                wedgeprops=dict(width=size, edgecolor='w'),
                legend=False, label='',
                colors=sns.color_palette('Set2'))

metrics.loc[metrics['Metric']=='Bayesian'].set_index('Number of segments')\
    .plot.pie(y='Count', autopct='%1.1f%%', 
                pctdistance = 1.25, labeldistance=175,
                ax=ax[1], radius = 1,
                wedgeprops=dict(width=size, edgecolor='w'),
                legend=False, 
                label='', colors=sns.color_palette('Set2'))

fig.legend(metrics['Number of segments'].unique(),
            title='Optimal number of segments',
            loc='lower center', ncol=6)

IC['Sector'] = IC['Tech'].apply(lambda x: utils.sectorsinv[x])

AIC = IC.loc[IC.groupby('Tech')['AIC'].idxmin()]\
            .groupby(['n_breaks', 'Sector']).count()['Tech'].reset_index()
BIC = IC.loc[IC.groupby('Tech')['BIC'].idxmin()]\
            .groupby(['n_breaks', 'Sector']).count()['Tech'].reset_index()

AIC['metric'] = 'Akaike'
BIC['metric'] = 'Bayesian'

metrics = pd.concat([AIC, BIC]).reset_index(drop=True)

metrics.columns = ['n_breaks', 'Sector', 'Count', 'Metric']
metrics['Number of segments'] = metrics['n_breaks'] + 1


metrics.sort_values(by=['n_breaks','Sector','Metric'], inplace=True)
metrics = metrics.reset_index(drop=True)
print(metrics.tail(39))


for n in range(0, max_breakpoints):
    for s in metrics['Sector'].unique():
        for m in metrics['Metric'].unique():
            if metrics.loc[metrics['Sector'] == s]\
                .loc[metrics['n_breaks'] == n]\
                .loc[metrics['Metric']==m].shape[0] == 0:
                metrics.loc[metrics.shape[0]] = [n,s,0.,m,n]

metrics.sort_values(by=['n_breaks','Sector','Metric'], inplace=True)
metrics = metrics.reset_index(drop=True)

print(metrics.loc[metrics['Count']>0].tail(39))

metrics.loc[metrics['Metric']=='Akaike'].set_index('Number of segments')\
    .plot.pie(y='Count', 
                labeldistance=175,
                ax=ax[0], radius = 1 - size - 0.05,
                wedgeprops=dict(width=size, edgecolor='w'),
                legend=False, label='',
                colors=utils.sectors_colors.values())

metrics.loc[metrics['Metric']=='Bayesian'].set_index('Number of segments')\
    .plot.pie(y='Count', 
                labeldistance=175,
                ax=ax[1], radius = 1 - size - 0.05,
                wedgeprops=dict(width=size, edgecolor='w'),
                legend=False, 
                label='', 
                colors=utils.sectors_colors.values())


fig.legend(handles=ax[0].patches[-6:],
           labels=utils.sectors_colors.keys(),
            title='Sector',
            bbox_to_anchor=[.5,.225], 
            loc='center',
            ncol=3)

ax[0].set_title('Akaike Information Criterion')
ax[1].set_title('Bayesian Information Criterion')
fig.subplots_adjust(bottom=0.3, left=0.05, right=0.95)
plt.tight_layout()
plt.show()
if not os.path.exists('figs' + os.path.sep + 'SupplementaryFigures'):
    os.makedirs('figs' + os.path.sep + 'SupplementaryFigures')
fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                os.path.sep +
            'PieSegmentsSectors' + \
                first_diff * '_first_diff' + '.png')


# get the number of technologies 
AIC = IC.loc[IC.groupby('Tech')['AIC'].idxmin()].reset_index()
BIC = IC.loc[IC.groupby('Tech')['BIC'].idxmin()].reset_index()    

# rename metrics
AIC['metric'] = 'Akaike'
BIC['metric'] = 'Bayesian'

# create new dataframe with tech counts and metrics
metrics = pd.concat([AIC, BIC]).reset_index(drop=True)
metrics = metrics[['n_breaks','metric', 'Number of observations']]

metrics['Number of segments'] = metrics['n_breaks'] + 1

# plot number of segments vs number of observations
fig, ax = plt.subplots(1,1, figsize=(9,5), sharex=True, sharey=True)
sns.scatterplot(data=metrics, 
        x = 'Number of observations',
        y = 'Number of segments',
        hue='Metric',
        ax = ax,
        alpha=0.5,
        legend=False)

ax.set_xlabel('Number of observations')
ax.set_ylabel('Number of segments')
ax.set_xscale('log', base=10)

fig.subplots_adjust(bottom=0.15)

# plot learning rates and correlation coefficients
fig, ax = plt.subplots(max_breakpoints, 2, figsize=(15,10), 
                        sharex=True, sharey='col')


for n_break in range(1, max_breakpoints + 1):
    sel = IC.loc[(IC['n_breaks'] == n_break) & \
            ((IC.index.isin(IC.groupby('Tech')\
                            ['AIC'].idxmin().values)) | \
            (IC.index.isin(IC.groupby('Tech')\
                            ['BIC'].idxmin().values)))]

    for i in range(n_break +  1):
        ax[n_break-1][0].plot([i-0.2, i+0.2],
                            [sel['LR '+str(i+1)].median(),
                            sel['LR '+str(i+1)].median()],
                            ls='--', color='w', lw=2)
        ax[n_break-1][0].fill_between(
            [i-0.2, i+0.2],
            sel['LR '+str(i+1)].quantile(0.25),
            sel['LR '+str(i+1)].quantile(0.75),
            alpha=0.6,
            lw=0,
            color=sns.color_palette()[0]
            )
        ax[n_break-1][0].fill_between(
            [i-0.2, i+0.2],
            sel['LR '+str(i+1)].quantile(0.05),
            sel['LR '+str(i+1)].quantile(0.95),
            alpha=0.3,
            lw=0,
            color=sns.color_palette()[0],
            )
        ax[n_break-1][0].scatter(i * np.ones(sel.shape[0]),
                                sel['LR '+str(i+1)], 
                                color=sns.color_palette()[0],
                                alpha=0.5,
                                s=25)
        if i < n_break:
            ax[n_break-1][1].bar(i+0.5, np.corrcoef(sel['LR '+str(i+1)],
                                            sel['LR '+str(i+2)])[0,1],
                                            color=sns.color_palette()[0])
            
        
    for t in sel['Tech'].unique():
        ax[n_break-1][0].plot([x for x in range(n_break +1)], 
                                sel[sel['Tech'] == t][['LR '+str(i+1) 
                                for i in range(n_break + 1)]].values[0],
                                color=sns.color_palette()[0],
                                lw=.5,)
        
    ax[n_break-1][0].axhline(0, color='k', 
                                ls='--', lw=.5, alpha=.5, zorder=-5)
    ax[n_break-1][1].axhline(0, color='k', 
                                ls='--', lw=.5, alpha=.5, zorder=-5)

ax[0][0].set_ylim(-120, 120)
ax[0][1].set_ylim(-1.2, 1.2)
ax[0][-1].set_xticks([])

ax[0][0].annotate('Learning rate [%]',
                    xy=(0.025, 0.5),
                    xycoords='figure fraction',
                    ha='center', va='center',
                    rotation=90)
ax[0][0].annotate('Correlation coefficient',
                    xy=(0.525, 0.5),
                    xycoords='figure fraction',
                    ha='center', va='center',
                    rotation=90)

plt.subplots_adjust(hspace=0.3, bottom=0.025, 
                    left=0.1, right=0.95, top=0.95)

plt.show()

