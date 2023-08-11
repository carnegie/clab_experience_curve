import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib
import analysisFunctions

# set color to be used to represent tech-specific slope
cmapp = matplotlib.colormaps['Purples']
# set color to be used to represent average slope
cmapg = matplotlib.colormaps['Greens']


# produce scatter figure with R2 values for 
# early slope and late slope
def scatterFigure(LR_cal, LR_val, sectorsList):

    # create figure and plot iteratively
    figscatter , axscatter = plt.subplots(figsize=(8,6))
    for item in zip(LR_cal, LR_val, sectorsList):
        axscatter.scatter(item[0], item[1], 
                    color = analysisFunctions.sectorsColor[item[2]],
                    alpha = 0.4)
    
    # compute R2
    if len(LR_cal) > 2:
        model = sm.OLS(LR_val, sm.add_constant(LR_cal))
        result = model.fit()
        axscatter.annotate('R2 = ' + str(round(result.rsquared,2)) + \
                '\n N = ' + str(len(LR_cal)),
                (-1.5,1))
    
    # plotting graphics
    axscatter.plot([0,0],[-3,3], color='k', alpha=0.8, lw=0.2, zorder=-30)
    axscatter.plot([-3,3],[0,0], color='k', alpha=0.8, lw=0.2, zorder=-30)
    axscatter.set_xlim((-2,2))
    axscatter.set_ylim((-2,2))
    axscatter.set_xlabel('First half slope')
    axscatter.set_ylabel('Second half slope')
    axscatter.set_xlim((-2,2))
    axscatter.set_ylim((-2,2))
    figscatter.subplots_adjust(bottom=0.1, top=0.9, left=0.0)
    axscatter.set_aspect('equal', 'box')
    if not(all([x==sectorsList[0] for x in sectorsList])):
        legend_elements = [
            matplotlib.lines.Line2D(\
                [0],[0], lw=0, color='royalblue',
                marker='o', label='Energy'),
            matplotlib.lines.Line2D(\
                [0],[0], lw=0, color='black', 
                marker='o', label='Chemicals'),
            matplotlib.lines.Line2D(\
                [0],[0], lw=0, color='red', 
                marker='o', label='Hardware'),  
            matplotlib.lines.Line2D(\
                [0],[0], lw=0, color='forestgreen', 
                marker='o', label='Consumer goods'),
            matplotlib.lines.Line2D(\
                [0],[0], lw=0, color='cyan', 
                marker='o', label='Food'),
            matplotlib.lines.Line2D(\
                [0],[0], lw=0, color='darkmagenta', 
                marker='o', label='Genomics')
        ]
        figscatter.legend(handles=legend_elements, 
                          loc='center right', title='Sector')
    else:
        axscatter.set_title(sectorsList[0])
    return figscatter, axscatter
    
# produce large plots:
# 1) early and late slope over observations
# 2) predictions over observations
# 3) error distributions
def gridPlots(uc, cpCal, cpVal, ucpred, errpred, ucpred2, errpred2):

    count = 0 # counter for subplot position
    countt = 0 # counter for tech position vs subplot position
    better = 0 # counter for techs with lower MSE using average slope

    # create figures
    fig2 , ax2 = plt.subplots(10,9, figsize=(13,7))
    fig3 , ax3 = plt.subplots(10,9, figsize=(13,7))
    fig4, ax4 = plt.subplots(10,9, figsize=(13,7))

    while count - countt < len(uc):

        # extract relevant data
        uc_ = uc[count - countt]
        cp_ = np.concatenate(\
            [cpCal[count - countt], cpVal[count - countt]])
        cpcal_ = cpCal[count - countt]
        cpval_ = cpVal[count - countt]
        ucpred_ = ucpred[count - countt]
        errpred_ = errpred[count - countt]
        ucpred2_ = ucpred2[count - countt]
        errpred2_ = errpred2[count - countt]

        # skip some subplots for aestethics
        if count in [81]:
            for x in range(int((90-len(uc))/2)):
                ax2[int(count/9)][count%9].axis('off')
                ax3[int(count/9)][count%9].axis('off')
                ax4[int(count/9)][count%9].axis('off')
                count += 1
                countt += 1

        # set axis scales
        ax2[int(count/9)][count%9].set_yscale('log', base=10)
        ax2[int(count/9)][count%9].set_xscale('log', base=10)
        ax4[int(count/9)][count%9].set_yscale('log', base=10)
        ax4[int(count/9)][count%9].set_xscale('log', base=10)

        # plot observations, predictions, and regression lines
        ax2[int(count/9)][count%9].scatter(
            cp_, uc_, 
            marker='o', color='firebrick',
            lw=0.5, facecolor='None', s=2)
        ax2[int(count/9)][count%9].plot(
            cp_, ucpred_, 
            color='k', alpha = 0.6)
        ax2[int(count/9)][count%9].plot(
            cp_[len(cpcal_)-1:], ucpred2_, 
            color='g', alpha = 0.6)
        xlim = ax2[int(count/9)][count%9].get_xlim()
        ylim = ax2[int(count/9)][count%9].get_ylim()
        ax2[int(count/9)][count%9].plot(
            [cp_[len(cpcal_)-1],cp_[len(cpcal_)-1]],
            [0,max(uc_)*1.2],
            'k', alpha=0.2, zorder=-30)

        # plot error distributions
        sns.kdeplot(errpred_, color='k', 
                    ax=ax3[int(count/9)][count%9])
        sns.kdeplot(errpred2_, color='g', 
                    ax=ax3[int(count/9)][count%9])

        # plot early and late slope over observations
        ax4[int(count/9)][count%9].scatter(
            cp_, uc_, 
            marker='o', color='firebrick',
            lw=0.5, facecolor='None', s=2)
        ax4[int(count/9)][count%9].plot(
            cpcal_, ucpred_[:len(cpcal_)], 
            color='k', alpha = 0.6)
        ax4[int(count/9)][count%9].plot(
            cpval_, ucpred_[len(cpcal_):], 
            color='b', alpha = 0.6)
        ax4[int(count/9)][count%9].plot(
            [cp_[len(cpcal_)-1],cp_[len(cpcal_)-1]],
            [0,max(uc_)*1.2],'k', alpha=0.2, zorder=-30)

        ax2[int(count/9)][count%9].set_xlim(xlim)
        ax2[int(count/9)][count%9].set_ylim(ylim)
        ax2[int(count/9)][count%9].set_xticks([])
        ax2[int(count/9)][count%9].set_yticks([])
        ax2[int(count/9)][count%9].minorticks_off()
        ax4[int(count/9)][count%9].set_xlim(xlim)
        ax4[int(count/9)][count%9].set_ylim(ylim)
        ax4[int(count/9)][count%9].set_xticks([])
        ax4[int(count/9)][count%9].set_yticks([])
        ax4[int(count/9)][count%9].minorticks_off()
        for axis in ['top','bottom','left','right']:
            ax2[int(count/9)][count%9].spines[axis].set_linewidth(0.1)
            ax3[int(count/9)][count%9].spines[axis].set_linewidth(0.1)
            ax4[int(count/9)][count%9].spines[axis].set_linewidth(0.1)
            ax2[int(count/9)][count%9].spines[axis].set_alpha(0.5)
            ax3[int(count/9)][count%9].spines[axis].set_alpha(0.5)
            ax4[int(count/9)][count%9].spines[axis].set_alpha(0.5)
        xlim = ax3[int(count/9)][count%9].get_xlim()
        ax3[int(count/9)][count%9].set_xlim(\
                                (-max(np.abs(xlim)),max(np.abs(xlim))))
        ax3[int(count/9)][count%9].set_xticks([0],[])
        ax3[int(count/9)][count%9].set_yticks([])
        ax3[int(count/9)][count%9].set_ylabel('')
        ax2[int(count/9)][count%9].axis('off')
        if sum(errpred_**2) > sum(errpred2_**2):
            for axis in ['top','bottom','left','right']:
                ax3[int(count/9)][count%9].spines[axis].set_color('green')
                ax3[int(count/9)][count%9].spines[axis].set_linewidth(1.0)
            better += 1
        count += 1

    # add annotation on error distribution figure
    ax3[int(count/9)][count%9].annotate(
        'The sum of squared errors is lower'+
         ' \n using the average technological slope'+\
        '\n'+' for '+\
        str(better)+ ' ('+str(round(100*better/86))+'%) technologies', 
        (0.5,0.065), ha='center', fontsize=10,
        xycoords='figure fraction')

    # graphical settings, axes labeling, and legends
    for x in range(int((90-len(uc))/2)):
        ax2[-1][-1-x].axis('off')
        ax3[-1][-1-x].axis('off')
        ax4[-1][-1-x].axis('off')
    fig2.suptitle('Predictions and observations')
    fig3.suptitle('Unit cost error distributions')
    fig4.suptitle('First and second half regression lines')
    ax2[0][0].annotate('Unit cost', (0.05,0.55), 
                    ha='center', rotation=90, 
                        xycoords='figure fraction')
    ax2[0][0].annotate('Cumulative production', 
                    (0.5,0.1), ha='center',
                    xycoords='figure fraction')
    ax4[0][0].annotate('Unit cost', (0.05,0.55), 
                    ha='center', rotation=90, 
                        xycoords='figure fraction')
    ax4[0][0].annotate('Cumulative production', 
                    (0.5,0.1), ha='center',
                    xycoords='figure fraction')
    legend_elements = [
        matplotlib.lines.Line2D([0],[0],color='k',
                                label='Technology-specific slope'),
        matplotlib.lines.Line2D([0],[0],color='g',
                                label='Mean technological slope')
    ]
    fig3.legend(handles=legend_elements, ncol=2, loc='lower center')
    legend_elements = [
        matplotlib.lines.Line2D([0],[0],color='firebrick', lw=0,
                                marker='o', markerfacecolor='None',
                                label='Observations'),
        matplotlib.lines.Line2D([0],[0],color='k',
                                label='Technology-specific slope'),
        matplotlib.lines.Line2D([0],[0],color='g',
                                label='Mean technological slope')
    ]
    fig2.legend(handles=legend_elements, ncol=3, loc='lower center')
    legend_elements = [
        matplotlib.lines.Line2D([0],[0],color='firebrick', lw=0,
                                marker='o', markerfacecolor='None',
                                label='Observations'),
        matplotlib.lines.Line2D([0],[0],color='k',
                                label='First half regression line'),
        matplotlib.lines.Line2D([0],[0],color='b',
                                label='Second half regression line')
    ]
    fig4.legend(handles=legend_elements, ncol=3, loc='lower center')
    fig2.subplots_adjust(bottom=0.15)
    fig3.subplots_adjust(bottom=0.15)
    fig4.subplots_adjust(bottom=0.15)
    return fig2, ax2, fig3, ax3, fig4, ax4

# produce figure with number of technologies with lower MSE
def barSectorMSE(errpred, errpred2, sectorsList):
    # prepare figure
    fig, ax = plt.subplots(2, 3, sharex=True, 
                           sharey=True, figsize=(10,6))
    # initialize lists
    avg, tech = [], []
    # iterate over errors and sector lists
    for item in zip(errpred, errpred2, sectorsList):
        # if the sector has not been initialized, initialize it
        if item[2] not in [x[0] for x in avg]:
            avg.append([item[2],0])
            tech.append([item[2],0])
        else:
            # if error using average slope is lower, update counter
            if sum(item[0]**2) > sum(item[1]**2):
                avg[[x[0] for x in avg].index(item[2])][1] += 1
            # else, update counter for technology-specific slope
            else:
                tech[[x[0] for x in tech].index(item[2])][1] += 1
    # plot bars for each sector
    for count in range(len(tech)):
        ax[count//3][count%3].set_title(tech[count][0])
        ax[count//3][count%3].bar(\
                            [0,1], [tech[count][1], avg[count][1]],
                            color = analysisFunctions\
                                .sectorsColor[tech[count][0]])
    # set labels and ticks
    ax[1][0].set_xticks([0,1],['Technology-specific','Average slope'])
    ax[1][1].set_xticks([0,1],['Technology-specific','Average slope'])
    ax[1][2].set_xticks([0,1],['Technology-specific','Average slope'])
    ax[0][0].set_ylabel('Number of technologies with lower MSE')
    ax[1][0].set_ylabel('Number of technologies with lower MSE')
    return fig, ax

# produce a figure plotting orders of magnitude available for each technology
def plotOrdersOfMagnitude(df):
    # initialize counter and lists for labels and orders of magnitude
    count = 0
    labs = []
    oOfM = []
    # open figure
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(8,10),
                           gridspec_kw={'height_ratios': [8,1]})
    # iterate over technologies
    for tech in df['Tech'].unique():
        sel = df.loc[df['Tech']==tech]
        x = np.log10(sel['Cumulative production'].values)
        x0 = x[0]
        x1 = x[-1]
        # append spanned orders of magnitude to list
        oOfM.append([x1-x0])
        # plot horizontal line
        ax[0].plot([0, x1-x0], [count, count], 
                   color='k',
                #    color = analysisFunctions\
                # .sectorsColor[analysisFunctions.sectorsinv[tech]],
                     lw=0.5)
        # update labels
        labs.append(tech + ' (' + str(sel.shape[0]) + ')')
        count += 1
    ax[0].set_yticks(\
        [x for x in range(len(df['Tech'].unique()))], labs, fontsize=5)
    ax[0].set_ylabel('Technology')
    # sort orders of magnitude to plot 
    # cumulative production orders of magnitude covered
    oOfM.sort()
    # compute number of samples as orders of magnitude covered increase
    density = [[0,86]]
    for el in oOfM:
        density.append([el[0], density[-1][1]-1])
    density = np.array(density)
    # plot number of technologies per 
    # cumulative production orders of magnitude covered
    ax[1].plot(density[:,0], density[:,1], color='k', lw=2)
    ax[1].set_xlabel('Orders of magnitude of cumulative production')
    ax[1].set_ylabel('Number of technologies available')
    plt.subplots_adjust(top=0.98, bottom=0.09, hspace=0.15, left=0.2)
    return fig, ax

def plotPercentiles(centered, pct, 
                    countPoints, countTechs, 
                    color, ax1, ax2, ax2b = None,
                    ):
    ax1.plot(\
        [10**x for x in centered],10**pct[:,4], 
        color=color, lw=2, zorder=-2)
    for r in range(2,-1,-1):
        ax1.fill_between(\
            [10**x for x in centered], 
            10**pct[:,1+r], 10**pct[:,-r-1], 
            alpha=0.1+0.2*r, color=color, zorder=-2-r, lw=0)
    ax1.plot([1,1],[0,100**100],'k', zorder=-2)
    ax2.plot([10**x for x in centered], countPoints, 
                color='k', lw=2, zorder=-2)
    ax2.set_ylim(0,max(countPoints)*1.1)
    if ax2b is None:
        ax2b = ax2.twinx()
    ax2b.plot([10**x for x in centered], countTechs, 
                color='red', lw=2, zorder=-2)    
    return ax2b

def plotBoxplotsArray(centered, stats, color, ax):
    for idx in range(len(stats)):
        ax.bxp([stats[idx]], positions = [10**centered[idx]], 
                widths = (10**centered[idx])/12,
                showfliers=False, boxprops=dict(color=color, lw=2), 
                manage_ticks=False)

def plotForecastErrors(dferrTech, dferrAvg,
                       fOrd, samplingPoints,
                       trainErr = None, tOrd = None):

    # create figures
    if trainErr is None:
        figsize = (7,8)
    else:
        figsize = (9,8)
    fig, ax = plt.subplots(3,1, sharex=True, figsize=figsize)

    if trainErr is not None:
        # analyze training errors
        pctTrain, _, trainIntAxis, \
            countPoints, countTechs, stats = \
                analysisFunctions.\
                    dataToPercentilesArray(trainErr,
                                        tOrd, samplingPoints,
                                        training=True)

        # plot percentiles and boxplots
        ax2b = plotPercentiles(trainIntAxis, pctTrain, 
                        countPoints, countTechs, 
                        color = 'b', ax1 = ax[0], ax2 = ax[2])
        plotBoxplotsArray(trainIntAxis, stats, 
                        color = 'b', ax = ax[0])
    else:
        ax2b = None

    # analyze forecast error
    pctTech, _, forecastIntAxis, \
        countPointsTech, countTechsTech, statsTech = \
            analysisFunctions.\
                dataToPercentilesArray(dferrTech,
                                     fOrd, samplingPoints)
    pctAvg, _, _, \
        _, _, statsAvg = \
            analysisFunctions.\
                dataToPercentilesArray(dferrAvg,
                                     fOrd, samplingPoints)

    ax2b = plotPercentiles(forecastIntAxis, pctTech,
                          countPointsTech, countTechsTech,
                          color = cmapp(0.7), ax1 = ax[0], 
                          ax2 = ax[2], ax2b = ax2b)
    ax2b = plotPercentiles(forecastIntAxis, pctAvg,
                          countPointsTech, countTechsTech,
                          color = cmapg(0.7), ax1 = ax[1], 
                          ax2 = ax[2], ax2b = ax2b)
    plotBoxplotsArray(forecastIntAxis, statsTech, 
                        color = cmapp(0.7), ax = ax[0])
    plotBoxplotsArray(forecastIntAxis, statsAvg, 
                        color = cmapg(0.7), ax = ax[1])

    # labeling and annotating graphs
    if tOrd is not None:
        ax[0].plot([1,1],[0,100],'k', zorder=-2)
        ax[1].plot([1,1],[0,100],'k', zorder=-2)
        ax[2].plot([1,1],[0,10**10],'k', zorder=-2)
        ax[0].annotate('Training', xy=(10**(-tOrd/2), 6), 
                       xycoords='data', ha='center', 
                       va='bottom', fontsize=12)
        ax[0].annotate('Forecast', xy=(10**(+fOrd/5), 6), 
                       xycoords='data', ha='center', 
                       va='bottom', fontsize=12)
        ax[0].set_xlim(10**-tOrd, 10**fOrd)
        ax[1].set_xlim(10**-tOrd, 10**fOrd)
        ax[1].annotate('Training', xy=(10**(-tOrd/2), 6), 
                       xycoords='data', ha='center', 
                       va='bottom', fontsize=12)
        ax[1].annotate('Forecast', xy=(10**(+fOrd/5), 6), 
                       xycoords='data', ha='center', 
                       va='bottom', fontsize=12)
        ax[2].set_xlim(10**-tOrd, 10**fOrd)
        ax[2].annotate('Training', xy=(10**(-tOrd/2), 6), 
                       xycoords='data', ha='center', 
                       va='bottom', fontsize=12)
        ax[2].annotate('Forecast', xy=(10**(+fOrd/5), 6), 
                       xycoords='data', ha='center', 
                       va='bottom', fontsize=12)
    else:
        ax[0].set_xlim(10**0, 10**fOrd)
        ax[1].set_xlim(10**0, 10**fOrd)
        ax[2].set_xlim(10**0, 10**fOrd)
    ax[0].set_ylim(0.1,10)
    ax[0].set_yscale('log', base=10)
    ax[0].set_xscale('log', base=10)
    ax[0].set_ylabel('Error (Actual/Predicted)')
    ax[0].set_title('Technologies available: ' + \
                    str(dferrAvg['Tech'].nunique()))
    ax[0].plot([0,10**10],[1,1],'k', zorder=-10)
    ax[1].set_ylim(0.1,10)
    ax[1].set_yscale('log', base=10)
    ax[1].set_xscale('log', base=10)
    ax[1].set_ylabel('Error (Actual/Predicted)')
    ax[1].plot([0,10**10],[1,1],'k', zorder=-10)
    ax[2].set_xscale('log', base=10)
    ax[2].set_xlabel(
        'Predicted cumulative production / Current cumulative production')
    ax[2].set_ylabel('Number of points to estimate error')
    ax[2].set_ylim(0, max(max(countPointsTech)*1.1, ax[2].get_ylim()[1]))
    if trainErr is not None:
        ax[2].set_ylim(0, 
                    max(
                        max(countPoints)*1.1, 
                            ax[2].get_ylim()[1]))

        


    if trainErr is not None:
        legend_elements = [
            matplotlib.lines.Line2D(\
                [0], [0], color='b', lw=2, label='Training error'),
            matplotlib.lines.Line2D(\
                [0], [0], color=cmapp(0.7), lw=2, 
                label='Forecast error - Technology-specific'),
            matplotlib.lines.Line2D(\
                [0], [0], color=cmapg(0.7), lw=2, 
                label='Forecast error - Average slope'),
                            ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3)
    else:
        legend_elements = [
            matplotlib.lines.Line2D(\
                [0], [0], color=cmapp(0.7), lw=2, 
                label='Forecast error - Technology-specific'),
            matplotlib.lines.Line2D(\
                [0], [0], color=cmapg(0.7), lw=2, 
                label='Forecast error - Average slope'),
                            ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2)      

    axes1=fig.add_axes([0.825,0.415,0.15,0.2])
    axes1.plot([-1,2],[0.5,0.5],'k', lw=2)
    axes1.fill_between([-1,2],[0.25,0.25],[0.75,0.75], 
                       color='k', alpha=0.3)
    axes1.fill_between([-1,2],[0.1,0.1],[0.9,0.9], 
                       color='k', alpha=0.3)
    axes1.fill_between([-1,2],[0,0],[1.0,1.0], 
                       color='k', alpha=0.1)
    axes1.annotate('10th percentile', 
                   xy=(4.0, 0.1), xycoords='data', 
                   ha='center', va='center', fontsize=7)
    axes1.annotate('25th percentile', 
                   xy=(4.0, 0.25), xycoords='data', 
                   ha='center', va='center', fontsize=7)
    axes1.annotate('Median', 
                   xy=(4.0, 0.5), xycoords='data', 
                   ha='center', va='center', fontsize=7)
    axes1.annotate('75th percentile', 
                   xy=(4.0, 0.75), xycoords='data', 
                   ha='center', va='center', fontsize=7)
    axes1.annotate('90th percentile', 
                   xy=(4.0, 0.9), xycoords='data', 
                   ha='center', va='center', fontsize=7)
    axes1.annotate('Max', xy=(4.0, 1.0), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)
    axes1.annotate('Min', xy=(4.0, 0.0), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)

    axes1.plot([0,1],[0.5,0.5],'k', lw=1)
    axes1.plot([0,1,1,0,0],[0.25,0.25,0.75,0.75,0.25], lw=2, color='k')
    axes1.plot([0,1],[0.5,0.5], lw=2, color='darkorange')
    axes1.plot([0,1],[0,0], color='k', lw=1)
    axes1.plot([0,1],[1,1], color='k', lw=1)
    axes1.plot([0.5,0.5],[0,0.25], color='k', lw=1)
    axes1.plot([0.5,0.5],[0.75,1], color='k', lw=1)

    axes1.set_xlim(-1,5)
    axes1.set_ylim(-0.2,1.2)
    axes1.set_xticks([])
    axes1.set_yticks([])
    axes1.axis('off')

    fig.subplots_adjust(top=0.92, bottom=0.11, right=0.8)

    return fig, ax

def summaryBoxplots(trForOrds, fErrsTech, fErrsAvg, Ranges):

    # create figure
    dim = int(len(trForOrds)**0.5)
    fig, ax = plt.subplots(dim, dim, figsize=(9,8), sharey=True)
    countax = 0 # counter for axes

    # for each provided training and forecast horizon pair
    for tf in trForOrds:

        # get training and forecast horizons
        tOrd = tf[0]
        fOrd = tf[1]
        counth = 0 # counter for horizons

        # list to store custom xticks and boxplots positions 
        # (obtained at first pass)
        if tf == trForOrds[0]:
            xticks = []

        # iterate over elements of Ranges
        for r in range(len(Ranges)):

            # if Ranges' element is in the examined horizon
            if Ranges[r] in trForOrds:

                # asymmetric horizons are dealt with a single case
                if Ranges[r][0] > Ranges[r][1]:
                    continue
                
                # if the horizon is not covered skip to the next
                # but consider asymmetric ones
                if (Ranges[r][0] < tOrd or Ranges[r][1] < fOrd) and \
                    not(Ranges[r][0]>=fOrd and Ranges[r][1]>=tOrd):
                    counth += 1
                    continue

                # get error for that horizon
                dferrTech = fErrsTech[r]
                dferrAvg = fErrsAvg[r]
                Techs = dferrTech['Tech'].unique()

                # if the horizon is not symmetric
                if Ranges[r][0] != Ranges[r][1]:

                    # get the symmetric horizon and its errors
                    r2 = Ranges.index([Ranges[r][1], Ranges[r][0]])
                    dferrTech2 = fErrsTech[r2]

                    # get list of technologies in both horizons
                    # and derive intersections
                    Techs = dferrTech['Tech'].unique()
                    Techs2 = dferrTech2['Tech'].unique()
                    Techs = np.intersect1d(Techs, Techs2)

                    # keep only technologies in the intersection
                    dferrTech = dferrTech.loc[\
                        dferrTech['Tech'].isin(Techs)]
                    dferrAvg = dferrAvg.loc[\
                        dferrAvg['Tech'].isin(Techs)]
                
                # if the horizon is not the examined one
                # select the original horizon and filter
                # based on the technologies that are in
                # the larger horizon
                if not(Ranges[r] == tf):
                    dferrTech = fErrsTech[Ranges.index(tf)]
                    dferrAvg = fErrsAvg[Ranges.index(tf)]
                    dferrTech = dferrTech.loc[\
                        dferrTech['Tech'].isin(Techs)]
                    dferrAvg = dferrAvg.loc[\
                        dferrAvg['Tech'].isin(Techs)]
            
                # compute percentiles and boxplots stats
                pctTech = analysisFunctions\
                    .computeTechWeightedPercentiles(dferrTech)
                pctAvg = analysisFunctions\
                    .computeTechWeightedPercentiles(dferrAvg)
                statsTech = analysisFunctions\
                    .computeBoxplots(np.array([[0,*pctTech]]),
                                    [0], positions=[1,3,4,5,7])
                statsAvg = analysisFunctions\
                    .computeBoxplots(np.array([[0,*pctAvg]]),
                                    [0], positions=[1,3,4,5,7])

                # plot boxplots
                ax[countax//dim][countax%dim].bxp(
                    [statsTech[0]], 
                    positions=[3*counth],
                    showfliers=False,
                    boxprops=dict(color=cmapp(0.7), lw=2),
                    manage_ticks=False,
                    widths=0.8)
                ax[countax//dim][countax%dim].bxp(
                    [statsAvg[0]], 
                    positions=[3*counth+1],
                    showfliers=False,
                    boxprops=dict(color=cmapg(0.7), lw=2),
                    manage_ticks=False,
                    widths=0.8)

                    # save xticks
                if tf == trForOrds[0] and \
                    Ranges[r][0] <= Ranges[r][1]:
                    xticks.append([3*counth+0.5,
                            '['+str(Ranges[r][0]) + \
                                ',' + str(Ranges[r][1])+']' + \
                                    '\nT=' + \
                                        str(dferrTech['Tech'].nunique())])

                # update horizons counter
                counth += 1
        
        # set scale, lims and ticks
        ax[countax//dim][countax%dim].plot([-10,100],[1,1], 'k', zorder=-10)
        ax[countax//dim][countax%dim].set_yscale('log', base=10)
        ax[countax//dim][countax%dim].set_ylim(0.1, 10)
        ax[countax//dim][countax%dim].set_xlim(-0.5, xticks[-1][0]+1)
        ax[countax//dim][countax%dim].set_xticks(
            [x[0] for x in xticks], [x[1] for x in xticks])
        
        #update axes counter
        countax += 1

    ## figure annotations
    ax[0][0].annotate('Training interval',
            xy=(0.025, .5), xycoords='figure fraction',
            horizontalalignment='center', verticalalignment='center',
            fontsize=12,
            rotation=90)
    for l in range(dim):
        ax[l][0].annotate("$10^{{{}}}$".format(trForOrds[l*dim][0]),
                xy=(-.1*dim, .5), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                )
    ax[0][0].annotate('Forecast interval',
                xy=(.5, .975), xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12
                )
    for l in range(dim):
        ax[0][l].annotate("$10^{{{}}}$".format(trForOrds[l][1]),
                xy=(.5, 1.1), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                )

    return fig, ax

