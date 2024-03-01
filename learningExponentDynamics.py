import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib, cmcrameri, os
import statsmodels.api as sm

# set figure parameters
matplotlib.rc('savefig', dpi=300)
sns.set_palette([cmcrameri.cm.batlowS(x) for x in range(10)])
sns.set_palette('colorblind')
sns.set_context('talk')
sns.set_style('ticks')
matplotlib.rc('font', family='Helvetica')


# set colormap
cmap = cmcrameri.cm.hawaii

# select techs to be plotted
selTechs = ['Photovoltaics_2']

# set column names 
cols = ['Unit cost', 'Year',
        'Production','Cumulative production']

fig = plt.figure(figsize=(12,12))

gs = matplotlib.gridspec.GridSpec(2,2)

ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])]

# iterate over selected techs
for tech in selTechs:

    # read data and drop nan values
    df = pd.read_csv('expCurveData/' + tech + '.csv').dropna()

    # rename columns
    df.columns = [ a + ' ('+b.split('(')[1]
                   for a,b in zip(cols,df.columns)]

    # convert data to float
    for col in df.columns:
        df[col] = [float(x) for x in df[col].values]
        
    # some techs have multiple data per year
    # and the year is not an integer:
    # assume all data collected in the same year 
    # are avaialable for prediction at the end of the year
    for col in [df.columns[1]]:
        df[col] = [int(x) for x in df[col].values]


    # set norm for colormap
    norm = matplotlib.colors.Normalize(
            vmin=df[df.columns[1]].unique()[0],
            vmax=df[df.columns[1]].unique()[-1])   
    
    # create figure
    sns.scatterplot(data=df, x=df.columns[3], y=df.columns[0],
                    hue=df.columns[1], ax=ax[0], palette=cmap,
                    legend=False, edgecolor='k', s=100)
    # set log-log scale and label axes
    ax[0].set_xscale('log', base=10)
    ax[0].set_yscale('log', base=10)
    ax[0].set_xlabel(df.columns[3])
    ax[0].set_ylabel(df.columns[0])

    # add star representing technology-specific learning exponent
    model = sm.OLS(np.log10(df[df.columns[0]].values),
                     sm.add_constant(np.log10(df[df.columns[3]].values)))
    res = model.fit()
    ax[1].set_xlabel('Past learning rate [%]')
    ax[1].set_ylabel('Future learning rate [%]')


    # iterate over all years from 2nd to second to last
    for i in range(df[df.columns[1]].unique()[1],
                   df[df.columns[1]].unique()[-1]):
        
        # split data into calibration and validation sets
        cal = df[df[df.columns[1]]<=i]
        val = df[df[df.columns[1]]>=i]

        # fit regression model to calibration  and validation data
        modelcal = sm.OLS(np.log10(cal[cal.columns[0]].values),
                            sm.add_constant(
                                np.log10(cal[cal.columns[3]].values)))
        modelval = sm.OLS(np.log10(val[val.columns[0]].values),
                            sm.add_constant(
                                np.log10(val[val.columns[3]].values)))
        rescal = modelcal.fit()
        resval = modelval.fit()

        # add lines to scatter plot
        if i == 1983 or i == 2003:
            ax[0].plot(cal[cal.columns[3]].values,
                        10**rescal.predict(
                            sm.add_constant(
                                np.log10(cal[cal.columns[3]].values))),
                        # color=cmap(norm(i)),
                        color='k',
                        zorder=1, lw=2)
            ax[0].plot(val[val.columns[3]].values,
                            10**rescal.predict(
                                sm.add_constant(
                                    np.log10(val[val.columns[3]].values))),
                            # color=cmap(norm(i)),
                            color='k',
                           zorder=1, lw=2, 
                            linestyle='--')
        
        # add points to learning exponent dynamics
        sns.scatterplot(x=[100*(1 - 2**rescal.params[1])], 
                        y=[100*(1-2**resval.params[1])],
                        color=cmap(norm(i)), edgecolor='k', s=100,
                        ax=ax[1], zorder=1, legend=False)
        # ax[1].scatter(lexpcal, lexpval,
        #                 color=cmap(norm(i)), zorder=1, marker='x')
        
    
    # add colorbar
    smap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    smap.set_array([])
    fig.subplots_adjust(right=0.7, top=0.95, bottom=0.075, left=0.15,
                        hspace=0.25)
    # cbar_ax = fig.add_axes([0.75, 0.25, 0.05, 0.5])
    cbar_ax = fig.add_axes([0.1, 0.525, 0.8, 0.02])
    cbar = fig.colorbar(smap, cax=cbar_ax, label='Year', 
                        orientation='horizontal')
    cbar.set_ticks([x for x in [1977, 1980, 1985, 1990, 
                                1995, 2000, 2005, 2009]])
    cbar.set_ticklabels([str(int(x)) for x in
                             [1977, 1980, 1985, 1990, 
                                1995, 2000, 2005, 2009]])
    # fig.colorbar(smap, ax=ax[1], label='Year')

    # add identity line to learning exponent dynamics
    lims = [ax[1].get_xlim(), ax[1].get_ylim()]
    lims = [min(lims[0][0],lims[1][0]), max(lims[0][1],lims[1][1])]
    ax[1].plot([-20,40],[-20,40], color='k', ls='--', lw=1, zorder=-10)
    ax[1].set_xlim(lims)
    ax[1].set_ylim(lims)
    ax[1].set_aspect('equal')

    ## annotate panels
    ax[0].annotate('a', xy=(0.05, 0.15),
                    xycoords='axes fraction',
                    ha='center', va='center')
    ax[1].annotate('b', xy=(0.05, 0.15),
                    xycoords='axes fraction',
                    ha='center', va='center')
    
    ax[0].annotate('Solar photovoltaics', xy=(0.5, 0.96),
                    xycoords='figure fraction',
                    ha='center', va='center')
    ax[0].annotate('All technologies', xy=(0.5, 0.4),
                    xycoords='figure fraction',
                    ha='center', va='center')
    
    ax[0].annotate('Underestimates\nfuture learning',
                   xy=(0.75, 0.7),
                    xycoords='axes fraction',
                    ha='center', va='center',
                    fontsize=12)
    ax[0].annotate('Overestimates\nfuture learning',
                   xy=(0.7, 0.05),
                    xycoords='axes fraction',
                    ha='center', va='center',
                    fontsize=12)
    
    ax[1].annotate('', xy=(12,8), xytext=(8,12),
                   xycoords='data', textcoords='data',
                     arrowprops=dict(arrowstyle='<->',
                                      color='k', lw=2))
    ax[1].annotate('Overestimates\nfuture learning',
                     xy=(15,5), xycoords='data',
                     ha='center', va='center',
                     fontsize=12)
    ax[1].annotate('Underestimates\nfuture learning',
                        xy=(5,15), xycoords='data',
                        ha='center', va='center',
                        fontsize=12)
    ax[1].annotate('1:1',
                        xy=(4.75,6.25), xycoords='data',
                        ha='center', va='center',
                        fontsize=12,
                        rotation=45)
    
    ax[1].annotate('1978',
                   xy=(22.5,28),
                     xycoords='data',
                        ha='center', va='center',
                        fontsize=12)
    ax[1].annotate('2008',
                   xy=(29,18),
                     xycoords='data',
                        ha='center', va='center',
                        fontsize=12)
                   


## add piecewise regression results
    
IC = pd.read_csv('IC.csv')

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

# plot the distribution of technologies over number of segments
ax = fig.add_subplot(gs[1,:])
sns.barplot(data=metrics, 
        x = 'Number of segments',
        y = 'Count',
        hue='Metric',
        ax = ax,
        legend=False)

# annotate figure
ax.annotate('Akaike\nInformation\nCriterion', (-.25, 25),
            xycoords='data', color=sns.color_palette()[0],
            ha='center', va='center', fontsize=14)
ax.annotate('Bayesian\nInformation\nCriterion', (2.25, 25),
            xycoords='data', color=sns.color_palette()[1],
            ha='center', va='center', fontsize=14)
style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color=sns.color_palette('colorblind')[0])
a = matplotlib.patches.FancyArrowPatch((-.25, 21), (.5, 18), 
                                       connectionstyle="arc3,rad=.1", **kw)
ax.add_patch(a)
kw = dict(arrowstyle=style, color=sns.color_palette('colorblind')[1])
a = matplotlib.patches.FancyArrowPatch((2.25, 21), (1.5, 18), 
                                       connectionstyle="arc3,rad=-.1", **kw)
ax.add_patch(a)
ax.annotate('c', xy=(0.025, 0.1),
                    xycoords='axes fraction',
                    ha='center', va='center')

# ax.yaxis.grid(lw=.5)

ax.set_xlim(-1,7)
ax.set_xlabel('Optimal number of segments')
ax.set_ylabel('Number of technologies')

plt.subplots_adjust(left=0.1, bottom=0.075, 
                    top=0.925, right=0.95,
                    hspace=0.8, wspace=0.5)

# save figure
if not(os.path.exists('figs' + 
                        os.path.sep + 
                        'learningExponentDynamics')):
    os.makedirs('figs' + 
                os.path.sep +
                    'learningExponentDynamics')

fig.savefig('figs' + os.path.sep + 
            'learningExponentDynamics' +
                os.path.sep + tech + '_InformationCriteria.png')
fig.savefig('figs' + os.path.sep + 
            'learningExponentDynamics' +
                os.path.sep + tech + '_InformationCriteria.eps')


plt.show()
        

