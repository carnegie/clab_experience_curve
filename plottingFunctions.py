import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import analysisFunctions

def plotErr(dfObsErr):

    # create figure
    fig, ax = plt.subplots(1, 3, 
                        figsize = (15,8),
                        sharex=True,
                        sharey=True,
                        )

    # plot boxplots

    # select upper and lower bound for bins
    lower = [10**0, 10**(1/3), 10**(2/3), 10**(1)]
    medium = [10**(1/6), 10**(3/6), 10**(5/6), 10**(7/6)]
    upper = [10**(1/3), 10**(2/3), 10**(1), 10**(4/3)]

    # iterate over variables for which to plot boxplots
    for var in [['Error (Tech)', 0, sns.color_palette('colorblind')[0]],
                ['Error (Avg)', 1, sns.color_palette('colorblind')[2]],
                ['Observation', 2, sns.color_palette('colorblind')[3]]]:

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
            
            # append data to lists
            upp.append(q95)
            low.append(q5)
            p75.append(q75)
            p25.append(q25)
            med.append(q50)
        
        # plot shaded area and median line
        ax[var[1]].fill_between(\
            [1,*medium], 10**np.array(low), 10**np.array(upp), 
            color=var[2], alpha=.3, lw=0)    
        ax[var[1]].fill_between(\
            [1,*medium], 10**np.array(p25), 10**np.array(p75), 
            color=var[2], alpha=.6, lw=0)    
        ax[var[1]].plot(\
            [1,*medium], 10**np.array(med), color='w', lw=2,
            linestyle='--')    
        ax[var[1]].plot(\
            [1,*medium], [1 for x in range(len(medium)+1)], 
            color='k', lw=1,
            linestyle='--')


    # annotate figure

    ax[0].set_title('Technology-specific')
    
    ax[1].set_title('Technology-mean')

    ax[2].set_title('Constant cost')

    ax[0].axhline(1, 1, 1.5, color='k', linestyle='--', 
                  lw=1, clip_on=False)
    ax[1].axhline(1, 1, 1.5, color='k', linestyle='--', 
                  lw=1, clip_on=False)
    ax[2].axhline(1, 1, 1.5, color='k', linestyle='--', 
                  lw=1, clip_on=False)
    ax[2].annotate('Underestimates\nfuture learning',
                    xy=(15,.5),
                    ha='center', va='center',
                    xycoords='data',
                    color='k',
                    fontsize=16, annotation_clip=False)
    ax[2].annotate('Overestimates\nfuture learning',
                    xy=(15,2),
                    ha='center', va='center',
                    xycoords='data',
                    color='k',
                    fontsize=16, annotation_clip=False)
    ax[2].annotate('',
                    xy=(15,1.5),
                    xytext=(15,0.66),
                    xycoords='data',
                    color='k',
                    arrowprops=dict(facecolor='k',
                                    color='k', arrowstyle='<->'),
                                    annotation_clip=False)
    
    for axc in [[0,0],[1,2],[2,3]]:
        ax[axc[0]].plot([4.5,5.5], [10**np.log10(4),10**np.log10(4)], 
                color='w', lw=2,
                linestyle='--'
                )
        ax[axc[0]].fill_between([4.5,5.5], 
                        [10**(np.log10(4)-.15),
                            10**(np.log10(4)-.15)], 
                        [10**(np.log10(4)+.15),
                            10**(np.log10(4)+.15)], 
                        color=sns.color_palette('colorblind')[axc[1]], 
                        lw=0, alpha=0.3)
        ax[axc[0]].fill_between([4.5,5.5], 
                        [10**(np.log10(4)-.075),
                            10**(np.log10(4)-.075)], 
                        [10**(np.log10(4)+.075),
                            10**(np.log10(4)+.075)], 
                        color=sns.color_palette('colorblind')[axc[1]], 
                        lw=0, alpha=0.6)
        ax[axc[0]].annotate('Median', xy=(6.75,10**np.log10(4)),
                        ha='center', va='center',
                        xycoords='data',
                        color='k',
                        fontsize=12, annotation_clip=False)
        ax[axc[0]].plot([4.25,3,3,4.25], 
                        [10**(np.log10(4)-.075),
                            10**(np.log10(4)-.075),
                            10**(np.log10(4)+.075),
                            10**(np.log10(4)+.075)],
                        color=sns.color_palette('colorblind')[axc[1]], 
                        lw=1,
                        )
        ax[axc[0]].plot([4.25,2,2,4.25], 
                        [10**(np.log10(4)-.15),
                            10**(np.log10(4)-.15),
                            10**(np.log10(4)+.15),
                            10**(np.log10(4)+.15)],
                        color=sns.color_palette('colorblind')[axc[1]], 
                        lw=1,
                        )
                        
        ax[axc[0]].annotate('50%', xy=(3.75,10**np.log10(4)),
                        ha='center', va='center',
                        xycoords='data',
                        color='k',
                        fontsize=12, annotation_clip=False)
        ax[axc[0]].annotate('90%', xy=(3.75,10**(np.log10(4)+.11)),
                        ha='center', va='center',
                        xycoords='data',
                        color='k',
                        fontsize=12, annotation_clip=False)
                            

    # set axes labels
    ax[0].set_ylabel('Prediction error')
    ax[1].set_xlabel('Cumulative production'
                    ' relative to reference')

    # adjust axes limits
    ax[0].set_xticks([1,5,10],[1,5,10])

    ax[0].set_ylim(0.1, 10)
    ax[0].set_xlim(1, 12)

    # set log scale for bottom panels
    ax[0].set_yscale('log', base=10)

    ## annotate panels
    ax[0].annotate('a', xy=(0.05,0.05),
                    xycoords='axes fraction',
                    ha='center', va='center')
    ax[1].annotate('b', xy=(0.05,0.05),
                    xycoords='axes fraction',
                    ha='center', va='center')
    ax[2].annotate('c', xy=(0.05,0.05),
                    xycoords='axes fraction',
                    ha='center', va='center')

    fig.subplots_adjust(bottom=0.1, top=0.925,
                        left=.1, right=.8,
                        hspace=0.25)
    
    return fig, ax


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
    ax[0][1].annotate('Technology-specific', 
                    xy=(6.5,1.6),
                    ha='center', va='center',
                    xycoords='data',
                    color=sns.color_palette('colorblind')[0])
    ax[0][2].annotate('Technology-mean',
                        xy=(6.5,1.6),
                        ha='center', va='center',
                        xycoords='data',
                        color=sns.color_palette('colorblind')[2])

    ax[1][1].annotate('Technology-specific', 
                    xy=(6.5,6),
                    ha='center', va='center',
                    xycoords='data',
                    color=sns.color_palette('colorblind')[0])
    ax[1][2].annotate('Technology-mean',
                        xy=(6.5,6),
                        ha='center', va='center',
                        xycoords='data',
                        color=sns.color_palette('colorblind')[2])

    # add boxplot legend
    ax[1][0].plot([0,.5], [1,1], color='black')
    ax[1][0].plot([0,.5,.5,0,0], [0.5,0.5,1.5,1.5,0.5], 
                  color='black')
    ax[1][0].fill_between([0,.5], [0,0], [2,2], 
                          color='black', alpha=.2)
    ax[1][0].fill_between([0,.5], [.5,.5], [1.5,1.5], 
                          color='black', alpha=.2)
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
            [1*(var=='Error (Avg)')-0.25, 
                1*(var=='Error (Avg)')+0.25], 
            10**np.array([q50, q50]), 
            color=color)
        ax[1][3].plot(\
            [1*(var=='Error (Avg)')-0.25, 
                1*(var=='Error (Avg)')+0.25, 
                1*(var=='Error (Avg)')+0.25, 
                1*(var=='Error (Avg)')-0.25, 
                1*(var=='Error (Avg)')-0.25], 
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
    ax[0][0].set_ylabel('Unit cost relative to reference')
    ax[1][1].set_ylabel('Prediction error')
    ax[1][1].set_xlabel('Cumulative production'
                    ' relative to reference')

    # adjust axes limits
    ax[0][0].set_xlim(1,12)
    ax[0][1].set_xlim(1,12)
    ax[0][2].set_xlim(1,12)
    [ax[a][b].set_xticks([1,5,10], ['1','5','10']) \
                    for a,b in zip([0,0,0,1,1],[0,1,2,1,2])]

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
                        labels=['Technology\nspecific', 
                                'Technology\nmean'])

    # set minor grid log scale plots
    ax[1][1].yaxis.grid(which='minor', linewidth=0.5)
    ax[1][2].yaxis.grid(which='minor', linewidth=0.5)
    ax[1][3].yaxis.grid(which='minor', linewidth=0.5)

    ## annotate panels
    ax[0][0].annotate('a', xy=(0.05,1.05),
                    xycoords='axes fraction',
                    ha='center', va='center')
    ax[0][1].annotate('b', xy=(0.05,1.05),
                    xycoords='axes fraction',
                    ha='center', va='center')
    ax[0][2].annotate('c', xy=(0.05,1.05),
                    xycoords='axes fraction',
                    ha='center', va='center')
    ax[1][1].annotate('d', xy=(0.05,1.05),
                    xycoords='axes fraction',
                    ha='center', va='center')
    ax[1][2].annotate('e', xy=(0.05,1.05),
                    xycoords='axes fraction',
                    ha='center', va='center')
    ax[1][3].annotate('f', xy=(0.05,1.05),
                    xycoords='axes fraction',
                    ha='center', va='center')

    fig.subplots_adjust(bottom=0.1, top=0.95,
                        left=.06, right=.98,
                        hspace=0.25)
    
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
            [1*(var=='Error (Avg)')-0.25, 
                1*(var=='Error (Avg)')+0.25], 
            10**np.array([q50, q50]), 
            color=color)
        ax[0].plot(\
            [1*(var=='Error (Avg)')-0.25, 
                1*(var=='Error (Avg)')+0.25, 
                1*(var=='Error (Avg)')+0.25, 
                1*(var=='Error (Avg)')-0.25, 
                1*(var=='Error (Avg)')-0.25], 
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

    ax[0].set_ylabel('Prediction error')
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
                [1*(var=='Error (Avg)')-0.25, 
                    1*(var=='Error (Avg)')+0.25], 
                10**np.array([q50, q50]), 
                color=color)
            ax[t[0]+1].plot(\
                [1*(var=='Error (Avg)')-0.25, 
                 1*(var=='Error (Avg)')+0.25, 
                    1*(var=='Error (Avg)')+0.25, 
                    1*(var=='Error (Avg)')-0.25, 
                    1*(var=='Error (Avg)')-0.25],
                10**np.array([q25, q25, q75, q75, q25]),
                color=color)
            
            if var=='Error (Tech)':
                label = 'Technology-specific'
            else:
                label = 'Technology-mean'

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
    
def plotErrTrFor(df):

    # define ranges to be plotted
    rangeslo = [0.0,0.5,1.0]
    rangesmed = [0.5,1.0,2.0]
    rangeshi = [1.0,1.5,3.0]

    # create figure
    fig, ax = plt.subplots(3,3, 
                        figsize = (10,9),
                        subplot_kw=dict(box_aspect=1),
                        sharex=True, sharey=True)

    # iterate over ranges for training
    for tr in enumerate(rangesmed):
        # iterate over ranges for forecast
        for fr in enumerate(rangesmed):

            sel = df.loc[df['Training horizon']>=rangesmed[tr[0]]]\
                    .loc[df['Max forecast horizon']>=rangesmed[fr[0]]]\
                    .loc[df['Forecast horizon']>=rangeslo[fr[0]]]\
                    .loc[df['Forecast horizon']<rangeshi[fr[0]]].copy()
            
            sel['weights'] = 1.0
            for t in sel['Tech'].unique():
                sel.loc[sel['Tech']==t, 'weights'] = 1.0 / \
                    sel.loc[sel['Tech']==t].shape[0] / \
                    sel['Tech'].nunique()
            
            ax[tr[0]][fr[0]].set_title(\
                '(' + str(int(10**rangesmed[tr[0]])) +
                ', ' + str(int(10**rangesmed[fr[0]])) + ') ' + 
                str(sel['Tech'].nunique()) + ' Techs',)

            for var in ['Error (Tech)', 'Error (Avg)']:
                q5, q25, q50, q75, q95 = \
                    sm.stats.DescrStatsW(\
                        sel[var], weights=sel['weights'])\
                            .quantile([0.05, 0.25, 0.5, 0.75, 0.95]) 
                
                if var=='Error (Tech)':
                    color = sns.color_palette('colorblind')[0]
                else:
                    color = sns.color_palette('colorblind')[2]

                ax[tr[0]][fr[0]].plot(\
                        [1*(var=='Error (Avg)')-0.25, 
                            1*(var=='Error (Avg)')+0.25], 
                        10**np.array([q50, q50]), 
                        color=color)
                ax[tr[0]][fr[0]].plot(\
                        [1*(var=='Error (Avg)')-0.25, 
                            1*(var=='Error (Avg)')+0.25, 
                            1*(var=='Error (Avg)')+0.25, 
                            1*(var=='Error (Avg)')-0.25, 
                            1*(var=='Error (Avg)')-0.25],
                        10**np.array([q25, q25, q75, q75, q25]),
                        color=color)
                
                if var=='Error (Tech)':
                    label = 'Technology-specific'
                else:
                    label = 'Technology-mean'

                ax[tr[0]][fr[0]].fill_between(\
                    [1*(var=='Error (Avg)')-.25, 
                        1*(var=='Error (Avg)')+.25], 
                    10**np.array([q5,q5]), 
                    10**np.array([q95,q95]), 
                    color=color, alpha=.2)
                ax[tr[0]][fr[0]].fill_between(\
                    [1*(var=='Error (Avg)')-.25, 
                        1*(var=='Error (Avg)')+.25],
                    10**np.array([q25,q25]), 
                        10**np.array([q75,q75]),
                    color=color, alpha=.2,
                    label=label)

    for a in ax:
        for axx in a:
            axx.set_ylim(.03, 10)
            axx.set_yscale('log', base=10)
            axx.yaxis.grid(which='minor', linewidth=0.5)
            axx.xaxis.grid(visible=False)  

    for a in ax[-1]:
        a.set_xticks([])

    ax[1][0].set_ylabel('Prediction error')
    
    legend = fig.legend(
                handles = ax[-1][-1].get_legend_handles_labels()[0],
                labels = ax[-1][-1].get_legend_handles_labels()[1],
                loc='lower center',
                ncol=2)

    for l in legend.legendHandles:
        l.set_alpha(.4)
        l.set_linewidth(4)

    axes = fig.add_axes([0.775, 0.375, 0.275, 0.3])

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

    fig.subplots_adjust(hspace=0.3, wspace=0,
                        bottom=.1, top=0.95,
                        right=.75, left=.1,
                        )

    return fig, ax
                

def plotStatisticalTestTech(df):

    # define sector colors 
    cmap = sns.color_palette('colorblind')
    sectorsColor = {'Energy': cmap[0], 'Chemicals': cmap[1],
               'Hardware': cmap[2], 'Consumer goods': cmap[3],
               'Food': cmap[4], 'Genomics': cmap[8]}

    # add sector column
    df['Sector'] = [analysisFunctions.sectorsinv[t] 
                        for t in df['Tech'].values]
    
    df['TechIsBetter'] = [1*(x[0] < x[1]) for x in \
        df[['Mean Error (Tech)', 'Mean Error (Avg)']].values]

    # create figure
    fig, ax = plt.subplots(df['Training horizon'].nunique(),
                           df['Forecast horizon'].nunique(),
                           figsize = (15,9),
                           sharey=True,
                           sharex=True)

    # iterate over training and forecast horizons
    for tr in enumerate(df['Training horizon'].unique()):
        for fr in enumerate(df['Forecast horizon'].unique()):
            
            # extract relevant data
            s = df.loc[df['Training horizon']==tr[1]]\
                    .loc[df['Forecast horizon']==fr[1]].copy()
            
            # create empty list to store data
            bars = {}

            # iterate over sectors
            for sec in analysisFunctions.sectors.keys():

                # find how many times technology specific 
                # has lower error and the different is significant
                techbetter = s.loc[s['Sector']==sec]\
                        .loc[s['Paired t-test'] < 0.05]\
                        .loc[s['Wilcoxon signed-ranks test'] < 0.05]\
                        .loc[s['TechIsBetter']==1].shape[0]
                                
                # find how many times the average slope method
                # has lower error and the difference is significant
                averagebetter = s.loc[s['Sector']==sec]\
                        .loc[s['Paired t-test'] < 0.05]\
                        .loc[s['Wilcoxon signed-ranks test'] < 0.05]\
                        .loc[s['TechIsBetter']==0].shape[0]

                # append data to list
                bars[sec] = [techbetter, 
                            s.loc[s['Sector']==sec].shape[0] - \
                                techbetter - averagebetter,
                            averagebetter]

            # convert list to dataframe
            bars = pd.DataFrame(bars, 
                                index=['Technology-specific',
                                        'No significant\ndifference',
                                        'Technology-mean'])

            # plot stacked barplot
            bars.plot.bar(stacked=True, 
                        color=[sectorsColor[x] for x in bars.columns],
                        ax=ax[tr[0]][fr[0]],
                        legend=False)

            # remove xaxis grid            
            ax[tr[0]][fr[0]].xaxis.grid(visible=False)

            # set descriptive title for panel
            ax[tr[0]][fr[0]].set_title(\
                '(' + str(int(10**tr[1])) + ', ' + 
                str(int(10**fr[1])) + ') ' + 
                str(s['Tech'].nunique()) + ' Techs',)

    # set yaxis label
    ax[1][0].set_ylabel('Number of technologies with lowest error')
      
    # add legend
    fig.legend(handles=ax[0][0].get_legend_handles_labels()[0],
                labels=ax[0][0].get_legend_handles_labels()[1],
                loc='lower center',
                ncol=6)

    # adjust figure    
    fig.subplots_adjust(bottom=0.35, top=0.95,
                        right=0.95, left=0.075,
                        wspace=0.05, hspace=0.3)
    
    return fig, ax




            