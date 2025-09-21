import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, utils

# set figures' parameters
sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300
sns.color_palette('colorblind')
palette = 'crest'

# set to true to plot a figure for each technology
plot_fig_tech = False

# set to True if the regression dataset needs to be built
build_pwreg_dataset = False

# use half of the data for each technology
half_data = False

# use half of the technologies in the dataset
half_techs = False

# set the maximum number of breakpoints
max_breakpoints = 6

# set min distance between breakpoints
min_dist = np.log10(2)
# min_dist = np.log10(1.001)

# include Moore's model
include_moore = False

# set random seed
np.random.seed(0)

try:
    if min_dist == np.log10(2):
        IC = pd.read_csv('IC.csv')
    else:
        IC = pd.read_csv('IC_' + str(10**min_dist) + '.csv')
    if include_moore:
        df = pd.read_csv('ExpCurves.csv')
        IC = utils.add_moore_model_regression_dataset(
            df, IC
        )
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
                                                    plot_fig_tech,
                                                    half_data,
                                                    half_techs)
    if include_moore:
        IC = utils.add_moore_model_regression_dataset(
            df, IC
        )

# get the number of technologies belonging to each number of segments
AIC = (IC.loc[IC.groupby('Tech')['AIC']
              .idxmin()]
              .groupby('n_breaks')
              .count()['Tech']
              .reset_index())
BIC = (IC.loc[IC.groupby('Tech')['BIC']
              .idxmin()]
              .groupby('n_breaks')
              .count()['Tech'].reset_index())

# rename metrics
AIC['metric'] = 'Akaike'
BIC['metric'] = 'Bayesian'

# create new dataframe with tech counts and metrics
metrics = pd.concat([AIC, BIC]).reset_index(drop=True)

# rename columns and add number of segments
metrics.columns = ['n_breaks', 'Count', 'Metric']
metrics['Number of segments'] = metrics['n_breaks'] + 1

# group technologies with more than 3 segments
metrics.loc[metrics['Number of segments']>3, 'Number of segments'] = '>3'
metrics.loc[metrics['Number of segments'] == '>3', 'n_breaks'] = 3
# sum metrics for technologies with more than 3 segments
for m in metrics['Metric'].unique():
    s = metrics.loc[(metrics['Number of segments']=='>3') & 
                    (metrics['Metric']==m),'Count'].sum()
    metrics.loc[(metrics['Number of segments']=='>3') & 
                (metrics['Metric']==m),'Count'] = metrics.loc[
                    (metrics['Number of segments']=='>3') & 
                        (metrics['Metric']==m),'Count'].sum()
# rename number of segments
metrics.loc[metrics['Number of segments']=='>3', 
            'Number of segments'] = '$\geq$4'
metrics = metrics.drop_duplicates()

# plot the distribution of technologies over number of segments - pie chart
fig, ax = plt.subplots(1,2, figsize=(12.5,8))

pie = metrics.loc[metrics['Metric']=='Akaike']\
    .set_index('Number of segments')\
    .plot.pie(y='Count', autopct='%1.1f%%', 
                counterclock=False,
                pctdistance = 1.25, labeldistance=.6,
                ax=ax[0], legend=False, 
                label='',
                colors=sns.color_palette(palette))

pie.patches[0].set_facecolor([.4*x for x in pie.patches[0].get_facecolor()])

pie = metrics.loc[metrics['Metric']=='Bayesian']\
    .set_index('Number of segments')\
    .plot.pie(y='Count', autopct='%1.1f%%', 
                counterclock=False,
                pctdistance = 1.25, labeldistance=.6,
                ax=ax[1], legend=False, 
                label='', colors=sns.color_palette(palette))

pie.patches[0].set_facecolor([.4*x for x in pie.patches[0].get_facecolor()])

fig.legend(metrics['Number of segments'].unique(),
            title='Optimal number of segments',
            loc='lower center', ncol=6)

ax[0].set_title('Akaike Information Criterion')
ax[1].set_title('Bayesian Information Criterion')
fig.subplots_adjust(bottom=0.125, left=0.05, right=0.95)
plt.tight_layout()

if not os.path.exists('figs'):
    os.makedirs('figs')

fig.savefig('figs' + os.path.sep 
            + 'PieSegments'
            + '_' + str(10**min_dist) 
            + ('_moore' if include_moore else '') 
            + '.png')
fig.savefig('figs' + os.path.sep
            + 'PieSegments' 
            + '_' + str(10**min_dist) + 
            ('_moore' if include_moore else '') +
            '.pdf')
fig.savefig('figs' + os.path.sep 
            + 'PieSegments' 
            + '_' + str(10**min_dist) 
            + ('_moore' if include_moore else '') 
            + '.eps')

# repeat the same analysis with sectors, 
# without aggregating the number of segments above 3
# and separating one segments vs all other number of segments

# get the number of technologies 
AIC = (IC.loc[IC.groupby('Tech')['AIC']
              .idxmin()]
              .groupby('n_breaks')
              .count()['Tech']
              .reset_index())
BIC = (IC.loc[IC.groupby('Tech')['BIC']
              .idxmin()]
              .groupby('n_breaks')
              .count()['Tech']
              .reset_index())

# rename metrics
AIC['metric'] = 'Akaike'
BIC['metric'] = 'Bayesian'

# create new dataframe with tech counts and metrics
metrics = pd.concat([AIC, BIC]).reset_index(drop=True)

# rename columns and add number of segments
metrics.columns = ['n_breaks', 'Count', 'Metric']
metrics['Number of segments'] = metrics['n_breaks'] + 1

# plot the distribution of technologies 
# over number of segments and sectors - pie chart
fig, ax = plt.subplots(1,2, figsize=(12.5,8))
# at the same time store the right panel in a separate figure
figbic, axbic = plt.subplots(1,1,figsize=(12.5,8))

size = .3

pie = metrics.loc[metrics['Metric']=='Akaike']\
    .set_index('Number of segments')\
    .plot.pie(y='Count', autopct='%1.1f%%', 
                counterclock=False,
                pctdistance = 1.25, labeldistance=175,
                ax=ax[0], radius = 1,
                wedgeprops=dict(width=size, edgecolor='w'),
                legend=False, label='',
                colors=sns.color_palette(palette, 
                                         n_colors=metrics['Number of segments'].nunique())
                )

pie.patches[0].set_facecolor([.4*x for x in pie.patches[0].get_facecolor()])

pie = metrics.loc[metrics['Metric']=='Bayesian']\
    .set_index('Number of segments')\
    .plot.pie(y='Count', autopct='%1.1f%%', 
                counterclock=False,
                pctdistance = 1.25, labeldistance=175,
                ax=ax[1], radius = 1,
                wedgeprops=dict(width=size, edgecolor='w'),
                legend=False, 
                label='', colors=sns.color_palette(palette,
                                                   n_colors=metrics['Number of segments'].nunique()))

pie.patches[0].set_facecolor([.4*x for x in pie.patches[0].get_facecolor()])

# use for separate segments plot
pie = metrics.loc[metrics['Metric']=='Bayesian']\
    .set_index('Number of segments')\
    .plot.pie(y='Count', autopct='%1.1f%%', 
                counterclock=False,
                pctdistance = 1.25, labeldistance=None,
                ax=axbic, radius = 1,
                wedgeprops=dict(width=size, edgecolor='w'),
                legend=False, 
                label='', colors=sns.color_palette(palette,
                                                   n_colors=metrics['Number of segments'].nunique()))

pie.patches[0].set_facecolor([.4*x for x in pie.patches[0].get_facecolor()])


fig.legend(metrics['Number of segments'].unique(),
            title='Optimal number of segments',
            loc='lower center', ncol=metrics['Number of segments'].nunique())

figbic.legend(handles=axbic.patches[:metrics['Number of segments'].nunique()],
            labels=metrics['Number of segments'].unique().tolist(),
            title='Optimal number of segments',
            loc='upper right', ncol=1)

## get the number of sectors belonging to each number of segments
IC['Sector'] = IC['Tech'].apply(lambda x: utils.sectorsinv[x])

AIC = (IC.loc[IC.groupby('Tech')['AIC']
              .idxmin()]
              .groupby(['n_breaks', 'Sector'])
              .count()['Tech'].reset_index())
BIC = (IC.loc[IC.groupby('Tech')['BIC']
              .idxmin()]
              .groupby(['n_breaks', 'Sector'])
              .count()['Tech'].reset_index())

AIC['metric'] = 'Akaike'
BIC['metric'] = 'Bayesian'

metrics = pd.concat([AIC, BIC]).reset_index(drop=True)

metrics.columns = ['n_breaks', 'Sector', 'Count', 'Metric']
metrics['Number of segments'] = metrics['n_breaks'] + 1

# prepare inner circle data (sectors belonging to one segment)
metrics.sort_values(by=['n_breaks','Sector','Metric'], inplace=True)
metrics = metrics.reset_index(drop=True)

for n in range(0, max_breakpoints):
    for s in metrics['Sector'].unique():
        for m in metrics['Metric'].unique():
            if (metrics.loc[metrics['Sector'] == s]
                .loc[metrics['n_breaks'] == n]
                .loc[metrics['Metric']==m].shape[0] == 0):
                metrics.loc[metrics.shape[0]] = [n,s,0.,m,n]

metrics.sort_values(by=['n_breaks','Sector','Metric'], inplace=True)
metrics = metrics.reset_index(drop=True)

# plot inner circles

metrics.loc[metrics['Metric']=='Akaike']\
    .set_index('Number of segments')\
    .plot.pie(y='Count', 
                labeldistance=175,
                counterclock=False,
                ax=ax[0], radius = 1 - size - 0.05,
                wedgeprops=dict(width=size, edgecolor='w'),
                legend=False, label='',
                colors=utils.sectors_colors.values())

metrics.loc[metrics['Metric']=='Bayesian']\
    .set_index('Number of segments')\
    .plot.pie(y='Count', 
                labeldistance=175,
                counterclock=False,
                ax=ax[1], radius = 1 - size - 0.05,
                wedgeprops=dict(width=size, edgecolor='w'),
                legend=False, 
                label='', 
                colors=utils.sectors_colors.values())

# use for separate segments plot
pie = metrics.loc[metrics['Metric']=='Bayesian']\
    .set_index('Number of segments')\
    .plot.pie(y='Count', 
                labeldistance=None,
                counterclock=False,
                ax=axbic, radius = 1 - size - 0.05,
                wedgeprops=dict(width=size, edgecolor='w'),
                legend=False, 
                label='', 
                colors=utils.sectors_colors.values())

# separate segments plot
for i,w in enumerate(pie.patches):
    if min_dist==np.log10(2) and (i == 0 or ( i > 5 and i < 12) ):
        w.set_center((.2,-.2))

pie.texts[0].set_position([x*1.2 for x in pie.texts[0].get_position()])

fig.legend(handles=ax[0].patches[-metrics['Number of segments'].nunique():],
           labels=utils.sectors_colors.keys(),
            title='Sector',
            bbox_to_anchor=[.5,.225], 
            loc='center',
            ncol=3)



legend = figbic.legend(handles=axbic.patches[-metrics['Number of segments'].nunique():],
            labels=utils.sectors_colors.keys(),
            title='Sector',
            loc='lower right',
            ncol=1)

ax[0].set_title('Akaike Information Criterion')
ax[1].set_title('Bayesian Information Criterion')
fig.subplots_adjust(bottom=0.3, left=0.05, right=0.95, top=0.95)
figbic.subplots_adjust(bottom=0.1, left=0.05, right=0.6, top=0.95)
figbic.savefig('figs'+os.path.sep+'BIC' 
                + '_' + str(10**min_dist) 
                + ('_moore' if include_moore else '') 
                + '.pdf')

if not os.path.exists('figs' + os.path.sep + 'SupplementaryFigures'):
    os.makedirs('figs' + os.path.sep + 'SupplementaryFigures')
fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' 
            + os.path.sep + 'PieSegmentsSectors' 
            + '_' + str(10**min_dist) 
            + ('_moore' if include_moore else '') 
            + '.png')
fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' 
            + os.path.sep + 'PieSegmentsSectors' 
            + '_' + str(10**min_dist) 
            + ('_moore' if include_moore else '') 
            + '.pdf')

### plot bar by sector
metrics_by_sector = []
for sector in metrics["Sector"].unique():
    for metric in ["Akaike", "Bayesian"]:
        sel = metrics.loc[(metrics["Sector"]==sector) & (metrics["Metric"]==metric)]
        metrics_by_sector.append([sector, metric, sel.loc[sel["Number of segments"]>1, "Count"].sum(), sel["Count"].sum()])
metrics_by_sector = pd.DataFrame(metrics_by_sector,
                                 columns = ["Sector", "Metric", "Technologies with breakpoints", "Total technologies"])
metrics_by_sector["Technologies with breakpoints (%)"] = (
    metrics_by_sector["Technologies with breakpoints"]
    / metrics_by_sector["Total technologies"] * 100
)
print("Sector-weighted percentage of technologies with breakpoints:")
print(f"AIC: {metrics_by_sector.loc[metrics_by_sector['Metric']=='Akaike', 'Technologies with breakpoints (%)'].mean()} %")
print(f"BIC: {metrics_by_sector.loc[metrics_by_sector['Metric']=='Bayesian', 'Technologies with breakpoints (%)'].mean()} %")
sns.catplot(data=metrics_by_sector, 
            col="Metric",
            hue="Sector",
            y="Technologies with breakpoints (%)",
            kind="bar",
            palette=utils.sectors_colors.values(),
            height=7,
            aspect=0.9)
plt.ylim(0, 100)
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' 
            + os.path.sep + 'BreakpointsAllSectorWeighted' + '.png')
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' 
            + os.path.sep + 'BreakpointsAllSectorWeighted' + '.pdf')

metrics_by_sector = metrics_by_sector.loc[metrics_by_sector["Metric"]=="Bayesian"]
sns.catplot(data=metrics_by_sector, 
            hue="Sector",
            y="Technologies with breakpoints (%)",
            kind="bar",
            palette=utils.sectors_colors.values(),
            height=7,
            aspect=1.2)
plt.ylim(0, 100)
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' 
            + os.path.sep + 'BreakpointsSectorWeighted' + '.png')
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' 
            + os.path.sep + 'BreakpointsSectorWeighted' + '.pdf')

# examine learning rates from piecewise regression

# get the number of technologies 
AIC = IC.loc[IC.groupby('Tech')['AIC'].idxmin()].reset_index()
BIC = IC.loc[IC.groupby('Tech')['BIC'].idxmin()].reset_index()    

# rename metrics
AIC['Metric'] = 'Akaike'
BIC['Metric'] = 'Bayesian'

# create new dataframe with tech counts and metrics
metrics = pd.concat([AIC, BIC]).reset_index(drop=True)
metrics = metrics[['n_breaks','Metric', 'Number of observations']]
metrics['Number of segments'] = metrics['n_breaks'] + 1

# plot learning rates and correlation coefficients
fig, ax = plt.subplots(max_breakpoints, 2, figsize=(15,10), 
                        sharex=True, sharey='col')

for n_break in range(1, max_breakpoints + 1):
    sel = IC.loc[(IC['n_breaks'] == n_break) & 
            ((IC.index.isin(IC.groupby('Tech')
                            ['AIC'].idxmin().values)) | 
            (IC.index.isin(IC.groupby('Tech')
                            ['BIC'].idxmin().values)))]

    for i in range(n_break +  1):
        ax[n_break-1][0].plot([i-0.2, i+0.2],
                            [sel['LR '+str(i+1)].median(),
                            sel['LR '+str(i+1)].median()],
                            ls='--', color='w', lw=2)
        if sel.shape[0] > 1:
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
            if sel.shape[0] > 1:
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

fig.subplots_adjust(hspace=0.3, bottom=0.025, 
                    left=0.1, right=0.95, top=0.95)

plt.show()

