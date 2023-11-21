import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import analysisFunctions

def plotObsPredErr(dfObsErr):

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

    # scatter errors
    ax[1][1].scatter(10**dfObsErr['Forecast horizon'].values,
                    10**dfObsErr['Error (Tech)'].values,
                    marker = '.', s=2, alpha=.1,
                    color=sns.color_palette('colorblind')[0])
    ax[1][2].scatter(10**dfObsErr['Forecast horizon'].values,
                    10**dfObsErr['Error (Avg)'].values,
                    marker = '.', s=2, alpha=.1,
                    color=sns.color_palette('colorblind')[2])

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


    # remove empty panels
    ax[0][3].axis('off')
    ax[1][0].axis('off')

    # set axes labels
    ax[0][0].set_ylabel('Unit cost')
    ax[1][1].set_ylabel('Observed / forecasted cost')
    ax[0][0].annotate('Cumulative production'
                    ' / current cumulative production', 
                    xy=(0.5, 0.05), 
                xycoords='figure fraction',
                ha='center', va='center',)

    # adjust axes limits
    ax[0][0].set_xlim(1,12)
    ax[0][0].set_xticklabels([])
    ax[0][1].set_xlim(1,12)
    ax[0][1].set_xticklabels([])
    ax[0][2].set_xlim(1,12)
    ax[0][2].set_xticklabels([])
    ax[0][0].set_ylim(0,1.75)
    ax[0][1].set_ylim(0,1.75)
    ax[0][2].set_ylim(0,1.75)
    ax[1][1].set_ylim(0.1, 10)
    ax[1][1].set_xlim(1, 12)
    ax[1][2].set_ylim(0.1, 10)
    ax[1][2].set_xlim(1,12)
    ax[1][3].set_ylim(0.1,10)

    # set log scale for bottom panels
    ax[1][1].set_yscale('log', base=10)
    ax[1][2].set_yscale('log', base=10)
    ax[1][3].set_yscale('log', base=10)

    # set yticklabels and xticks for boxplot
    ax[0][1].set_yticklabels([])
    ax[0][2].set_yticklabels([])
    ax[1][2].set_yticklabels([])
    ax[1][3].set_yticklabels([])
    ax[1][3].set_xticks([0,1], 
                        labels=['Technology-specific\nslope', 
                                'Average\nslope'])

    # set minor grid log scale plots
    ax[1][1].yaxis.grid(which='minor', linewidth=0.5)
    ax[1][2].yaxis.grid(which='minor', linewidth=0.5)
    ax[1][3].yaxis.grid(which='minor', linewidth=0.5)

    fig.subplots_adjust(bottom=0.15, top=0.95,
                        left=.06, right=.98,)
    
    return fig, ax

def plotErrorTech(df):

    df = df.loc[\
        (df['Forecast horizon']>=np.log10(10**0.5)) &\
        (df['Forecast horizon']<np.log10(10**1.5))].copy()

    df['Sector'] = [analysisFunctions\
                    .sectorsinv[x] for x in df['Tech'].values]

    df['Median error'] = 1
    for t in df['Tech'].unique():
        df.loc[df['Tech']==t, 'Median error'] = \
            df.loc[(df['Tech']==t), 
                   'Error (Avg)'].median()
    
    df = df.sort_values(by=['Sector', 'Median error'])

    # create figure
    fig, ax = plt.subplots(1, df['Tech'].nunique()+1,
                        figsize = (15,9),
                        sharey=True)

    sel = df.copy()

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
        ax[0].plot(\
            [1*(var=='Error (Avg)')-0.25, 1*(var=='Error (Avg)')+0.25], 
            10**np.array([q50, q50]), 
            color=color)
        ax[0].plot(\
            [1*(var=='Error (Avg)')-0.25, 1*(var=='Error (Avg)')+0.25, 
                1*(var=='Error (Avg)')+0.25, 1*(var=='Error (Avg)')-0.25, 1*(var=='Error (Avg)')-0.25], 
            10**np.array([q25, q25, q75, q75, q25]), 
            color=color)

        # plot shaded area and median line
        ax[0].fill_between(\
            [1*(var=='Error (Avg)')-.25, 1*(var=='Error (Avg)')+.25], 
            10**np.array([q5,q5]), 10**np.array([q95,q95]), 
            color=color, alpha=.2)    
        ax[0].fill_between(\
            [1*(var=='Error (Avg)')-.25, 1*(var=='Error (Avg)')+.25], 
            10**np.array([q25,q25]), 10**np.array([q75,q75]), 
            color=color, alpha=.2)  

    ax[0].set_ylabel('Observed / forecasted cost')
    ax[0].set_xticks([0.5], ['All technologies'],
                     rotation=90)
    
    for t in enumerate(df['Tech'].unique()):
        sel = df.loc[\
            (df['Forecast horizon']>=np.log10(10**0.5)) &\
            (df['Forecast horizon']<=np.log10(10**1.5)) &\
            (df['Tech']==t[1])].copy()

        for var in ['Error (Tech)', 'Error (Avg)']:
            # compute quantiles
            q5, q25, q50, q75, q95 = \
                sel[var].quantile([0.05, 0.25, 0.5, 0.75, 0.95])

            if var=='Error (Tech)':
                color = sns.color_palette('colorblind')[0]
            else:
                color = sns.color_palette('colorblind')[2]

            # plot boxplots
            ax[t[0]+1].plot(\
                [1*(var=='Error (Avg)')-0.25, 1*(var=='Error (Avg)')+0.25], 
                10**np.array([q50, q50]), 
                color=color)
            ax[t[0]+1].plot(\
                [1*(var=='Error (Avg)')-0.25, 1*(var=='Error (Avg)')+0.25, 
                    1*(var=='Error (Avg)')+0.25, 1*(var=='Error (Avg)')-0.25, 1*(var=='Error (Avg)')-0.25],
                10**np.array([q25, q25, q75, q75, q25]),
                color=color)
            
            if var=='Error (Tech)':
                label = 'Technology-specific slope'
            else:
                label = 'Average slope'

            # plot shaded area and median line
            ax[t[0]+1].fill_between(\
                [1*(var=='Error (Avg)')-.25, 1*(var=='Error (Avg)')+.25], 
                10**np.array([q5,q5]), 10**np.array([q95,q95]), 
                color=color, alpha=.2)
            ax[t[0]+1].fill_between(\
                [1*(var=='Error (Avg)')-.25, 1*(var=='Error (Avg)')+.25],
                10**np.array([q25,q25]), 10**np.array([q75,q75]),
                color=color, alpha=.2,
                label=label)
            ax[t[0]+1].set_xticks([0.5], [sel['Sector'].values[0] + 
                                          ' - ' + t[1].replace('_',' ')],
                                rotation=90)


    for axx in ax:
        axx.set_ylim(.09,10)
        axx.set_yscale('log', base=10)
        axx.yaxis.grid(which='minor', linewidth=0.5)
        axx.xaxis.grid(visible=False)

    axes = fig.add_axes([0.825, 0.6, 0.2, 0.375])
    axes.grid(False)
    axes.set_axis_off()

    axes.plot([0,.5], [1,1], color='black')
    axes.plot([0,.5,.5,0,0], [0.5,0.5,1.5,1.5,0.5], color='black')
    axes.fill_between([0,.5], [0,0], [2,2], color='black', alpha=.2)
    axes.fill_between([0,.5], [.5,.5], [1.5,1.5], color='black', alpha=.2)
    axes.set_ylim(-1,3)
    axes.set_xlim(-1.8,3)
    axes.annotate('50%', xy=(-.5,1),
                        ha='center', va='center',
                        xycoords='data')
    axes.annotate('90%', xy=(-1.5,1),
                        ha='center', va='center',
                        xycoords='data')
    axes.annotate('Median', xy=(.6,1),
                    ha='left', va='center',
                        xycoords='data')
    axes.plot([-.1,-.5,-.5], [1.5,1.5,1.25], color='black')
    axes.plot([-.1,-.5,-.5], [.5,.5,.75], color='black')
    axes.plot([-.1,-1.5,-1.5], [2,2,1.25], color='silver')
    axes.plot([-.1,-1.5,-1.5], [0,0,.75], color='silver')

    legend = fig.legend(handles = ax[-1].get_legend_handles_labels()[0],
                labels = ax[-1].get_legend_handles_labels()[1],
                loc='lower center',
                ncol=2)

    for x in legend.legendHandles:
        x.set_alpha(.4)
        x.set_linewidth(4)

    fig.subplots_adjust(bottom=0.6, top=0.975,
                        left=0.075, right=0.8,
                        wspace=0)

    return fig, ax
    