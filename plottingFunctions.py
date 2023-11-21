import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib, cmcrameri
import analysisFunctions
import pandas as pd

sns.color_palette('colorblind')
# # set color to be used to represent tech-specific slope
cmapp = matplotlib.colormaps['Purples']
cmapp = '#750FFA'
cmapp = 'C0'
# # set color to be used to represent average slope
cmapg = matplotlib.colormaps['Greens']
cmapg = '#057825'
# cmapg = 'C2'

# produce scatter figure with R2 values for 
# early slope and late slope
def scatterFigure(LR_cal, LR_val, selected=None, title=None):

    # create figure and plot iteratively
    figscatter , axscatter = plt.subplots(figsize=(8,7))
    colors = ['C9' for x in range(len(LR_cal))]
    if selected is not None:
        for x in selected:
            colors[x] = 'C4'
    # vals = axscatter.scatter(LR_cal, LR_val,
    #                 #   norm = norm,
    #                 #   cmap = cmap,
    #                   alpha = 0.7,
    #                   lw = 0.2,
    #                   s = 50,
    #                   edgecolor='k')
    
    vals = axscatter.scatter(LR_cal, LR_val,  c=colors,
                    #   c=length,
                    #   norm = norm,
                    #   cmap = cmap,
                      alpha = 0.8,
                      lw = 0.2,
                      s = 50,
                      edgecolor='k')
    
    # compute R2
    if len(LR_cal) > 2:
        model = sm.OLS(LR_val, sm.add_constant(LR_cal))
        result = model.fit()
        # axscatter.annotate('N = ' + str(len(LR_cal)),
        #         (-1.5,1))
        print('Explained variance (R2) = ',result.rsquared)
        model = sm.OLS([LR_val[x] for x in selected], sm.add_constant([LR_cal[x] for x in selected]))
        result = model.fit()
        print('Explained variance (R2) = ',result.rsquared)

    # # compute R2
    # if len(LR_cal) > 2:
    #     model = sm.OLS([LR_val[i] for i in selected], 
    #                    sm.add_constant([LR_cal[i] for i in selected]))
    #     result = model.fit()
    #     # axscatter.annotate('N = ' + str(len(LR_cal)),
    #     #         (-1.5,1))
    #     print('Explained variance (R2) for selected techs = ',result.rsquared)

    if selected is not None:
        axscatter.annotate('Selected technologies: \n'+\
                           'cumulative production range covers\n'+\
                            ' at least two orders of magnitude',
                        (-1,0.5),
                        ha='center',
                        va='center',
                        color='C4',
                        fontsize=11
                        )
    # axscatter.annotate('Non selected technologies',
    #                    (1.2,-0.4),
    #                    ha='center',
    #                    va='center',
    #                    color='C9')

    # plotting graphics
    axscatter.plot([0,0],[-3,3], color='k', alpha=0.8, lw=0.2, zorder=-30)
    axscatter.plot([-3,3],[0,0], color='k', alpha=0.8, lw=0.2, zorder=-30)
    # axscatter.set_xlim((-2,2))
    # axscatter.set_ylim((-2,2))
    axscatter.set_xlabel('Slope of the first half of the experience curve')
    axscatter.set_ylabel('Slope of the second half of the experience curve')
    axscatter.set_xlim((-2,1.5))
    axscatter.set_ylim((-2,1.5))
    axscatter.set_aspect('equal', 'box')
    # cbar = figscatter.colorbar(vals, 
    #                         cmap=cmap, norm=norm, 
    #                         orientation='horizontal', shrink = 0.6)
    # cbar.set_label('Final cumulative production / Initial cumulative production')
    # axscatter.set_title(title+' (N = '+str(len(LR_cal))+')')
    figscatter.tight_layout()
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
        
        model = sm.OLS(np.log10(uc_[len(cpcal_):]),
                          sm.add_constant(np.log10(cp_[len(cpcal_):])))
        result = model.fit()


        ax4[int(count/9)][count%9].plot(
            10**result.predict(
                sm.add_constant(np.log10(cp_[len(cpcal_):]))), 
            cp_[len(cpcal_):], 
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
    ax[0][0].set_ylabel('Technologies with lower MSE')
    ax[1][0].set_ylabel('Technologies with lower MSE')
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
                    countPoints = None, countTechs = None, 
                    color = 'k', ax1 = None, ax2 = None, ax2b = None,
                    ):
    ax1.plot(\
        [10**x for x in centered],10**pct[:,4], 
        color=color, lw=2, zorder=-2)
    for r in range(2,0,-1):
        ax1.fill_between(\
            [10**x for x in centered], 
            10**pct[:,1+r], 10**pct[:,-r-1], 
            alpha=0.1+0.2*r, color=color, zorder=-2-r, lw=0)
    # ax1.plot([1,1],[0,10],'k', zorder=-2)
    if ax2 is not None:
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
                medianprops=dict(color='#931621', lw=1.5, zorder=2),
                manage_ticks=False,
                capprops=dict(lw=0), whiskerprops=dict(lw=0))


def plotForecastErrorGrid(fErrsTech, fErrsAvg, Ranges, 
                            tfOrds, samplingPoints):
    
    dim = int(len(tfOrds)**0.5)

    # create figure
    fig, ax = plt.subplots(dim,2*dim, sharex='col', sharey=True, figsize=(12,7))

    # iterate over order of magnitude of interest
    for tf in tfOrds:

        # prepare axes
        ax1 = ax[int(tfOrds.index(tf)/dim)][(2*tfOrds.index(tf))%(2*dim)]
        ax2 = ax[int(tfOrds.index(tf)/dim)][(2*tfOrds.index(tf))%(2*dim)+1]

        # select data
        dferrTech = fErrsTech[Ranges.index(tf)]
        dferrAvg = fErrsAvg[Ranges.index(tf)]

        # get data to be plotted
        pctTech, _, forecastIntAxis, \
            _, _, statsTech = \
                analysisFunctions.\
                    dataToPercentilesArray(dferrTech,
                                        tf[1], samplingPoints)
        pctAvg, _, _, \
            _, _, statsAvg = \
                analysisFunctions.\
                    dataToPercentilesArray(dferrAvg,
                                        tf[1], samplingPoints)
        
        plotPercentiles(forecastIntAxis, pctTech,
                          color = cmapp, ax1 = ax1)
        plotPercentiles(forecastIntAxis, pctAvg,
                          color = cmapg, ax1 = ax2)
        plotBoxplotsArray(forecastIntAxis[1:-1], statsTech[1:-1], 
                        color = cmapp, ax = ax1)
        plotBoxplotsArray(forecastIntAxis[1:-1], statsAvg[1:-1], 
                        color = cmapg, ax = ax2)

        ax1.plot([1,1e10],[1,1],'k', zorder=-10)
        ax2.plot([1,1e10],[1,1],'k', zorder=-10)
        ax1.set_xscale('log', base=10)
        ax1.set_yscale('log', base=10)
        ax2.set_xscale('log', base=10)
        ax2.set_yscale('log', base=10)
        ax1.set_ylim(0.1,10)
        ax2.set_ylim(0.1,10)
        ax1.set_xlim(10**0, 10**tf[1])
        ax2.set_xlim(10**0, 10**tf[1])
        ax1.annotate('Technology-specific', 
                     xy=(1.1, 6), 
                     color=cmapp,
                     ha='left')
        ax2.annotate('Average slope', 
                     xy=(1.1, 6), 
                     color=cmapg,
                     ha='left')
        ax1.set_title('Techs = '+ str(dferrTech['Tech'].nunique()))
        ax2.set_title('Techs = '+ str(dferrTech['Tech'].nunique()))

    for x in range(dim):
        ax[x][0].set_ylabel('Unit cost error (actual/predicted)')
    ax[1][0].annotate('Future cumulative production' +\
                      ' / Current cumulative production',
                      xy=(0.5,0.05),
                        xycoords='figure fraction',
                        ha='center',
                        va='center')
    
    ax[0][0].annotate('Forecast horizon',
                      xy=(0.5,0.975),
                        xycoords='figure fraction',
                        ha='center',
                        va='center')
    ax[0][0].annotate('Training horizon',
                      xy=(0.025,0.5),
                        xycoords='figure fraction',
                        ha='center',
                        va='center',
                        rotation=90)

    ax[0][0].annotate('$10^{0.5}$',
                      xy=(1.1,1.2),
                        xycoords='axes fraction',
                        ha='center',
                        va='center')
    ax[0][2].annotate('$10^1$',
                      xy=(1.1,1.2),
                        xycoords='axes fraction',
                        ha='center',
                        va='center')
    ax[0][0].annotate('$10^{0.5}$',
                      xy=(-0.7,0.5),
                        xycoords='axes fraction',
                        ha='center',
                        va='center')
    ax[1][0].annotate('$10^1$',
                      xy=(-0.7,0.5),
                        xycoords='axes fraction',
                        ha='center',
                        va='center')
    if dim>2:
        ax[0][4].annotate('$10^2$',
                    xy=(1.1,1.2),
                    xycoords='axes fraction',
                    ha='center',
                    va='center')
        ax[2][0].annotate('$10^2$',
                    xy=(-0.7,0.5),
                    xycoords='axes fraction',
                    ha='center',
                    va='center')
    
    if dim>2:
        ax[0][0].set_ylabel('')
        ax[2][0].set_ylabel('')

    # ax[0][2].minorticks_off()
    # ax[0][3].minorticks_off()
    # ax[1][2].minorticks_off()
    # ax[1][3].minorticks_off()

    # ax[0][2].xminortickslabels([1,10],ax[0][2].get_xticklabels()[[0,-1]])
    # ax[0][3].xminortickslabels([1,10],ax[0][2].get_xticklabels()[[0,-1]])
    ax[1][2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1][3].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    
    axes1=fig.add_axes([0.9,0.35,0.08,0.3])
    axes1.plot([-0.5,1.5],[0.5,0.5],'k', lw=2, zorder=-2)
    axes1.fill_between([-0.5,1.5],[0.25,0.25],[0.75,0.75], 
                       color='k', alpha=0.3, zorder=-10)
    axes1.fill_between([-.5,1.5],[0.05,0.05],[0.95,0.95], 
                       color='k', alpha=0.3, zorder=-10)
    # axes1.fill_between([-.5,1.5],[0,0],[1.0,1.0], 
    #                    color='k', alpha=0.1)
    axes1.plot([1.5,2,2,1.5],[0.25,0.25,0.75,0.75], color='k',
               linestyle='--',lw=0.5)
    axes1.plot([1.5,4,4,1.5],[0.05,0.05,0.95,0.95], color='k',
               linestyle='--',lw=0.5)

    axes1.annotate('50%', xy=(3, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7,
                   rotation=90)
    axes1.annotate('90%', xy=(5, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7,
                   rotation=90)
    axes1.annotate('Median', xy=(-2.5, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)
    axes1.annotate('Max', xy=(0.5, 1.1), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)
    axes1.annotate('Min', xy=(0.5, -0.1), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)

    # axes1.plot([0,1],[0.5,0.5],'k', lw=1)
    axes1.plot([0,1,1,0,0],[0.25,0.25,0.75,0.75,0.25], lw=2, color='k')
    axes1.plot([0,1],[0.5,0.5], lw=2, color='#931621', zorder=-1)
    axes1.plot([0,1],[0,0], color='k', lw=1)
    axes1.plot([0,1],[1,1], color='k', lw=1)
    axes1.plot([0.5,0.5],[0,0.25], color='k', lw=1)
    axes1.plot([0.5,0.5],[0.75,1], color='k', lw=1)

    axes1.set_xlim(-3.5,5)
    axes1.set_ylim(-0.5,1.5)
    axes1.set_xticks([])
    axes1.set_yticks([])
    axes1.axis('off')

    fig.tight_layout()
    fig.subplots_adjust(left=0.125, right=0.875)
    return fig, ax

def plotForecastSlopeError(fErrsTech, fErrsAvg, 
                            tf, Ranges, 
                            samplingPoints, vert=True):
    # open figure
    # fig, ax = plt.subplots(1,4, width_ratios=[1,1,0.175,1.25], figsize=(12,4))
    fig, ax = plt.subplots(1,3, sharey=True, figsize=(15,5))

    # prepare axes
    ax1 = ax[0]
    ax2 = ax[1]

    # select data
    dferrTech = fErrsTech[Ranges.index(tf)]
    dferrAvg = fErrsAvg[Ranges.index(tf)]

    # get data to be plotted
    pctTech, _, forecastIntAxis, \
        _, _, statsTech = \
            analysisFunctions.\
                dataToPercentilesArray(dferrTech,
                                    tf[1], samplingPoints)
    pctAvg, _, _, \
        _, _, statsAvg = \
            analysisFunctions.\
                dataToPercentilesArray(dferrAvg,
                                    tf[1], samplingPoints)
    
    ax1.scatter(10**dferrTech['Forecast horizon'].values, 10**dferrTech['Error'].values,
                  marker='.', color='firebrick', zorder=-2, alpha=0.05)
    ax2.scatter(10**dferrAvg['Forecast horizon'].values, 10**dferrAvg['Error'].values,
                  marker='.', color='firebrick', zorder=-2, alpha=0.05)

    plotPercentiles(forecastIntAxis, pctTech,
                        color = 'firebrick', ax1 = ax1)
    plotPercentiles(forecastIntAxis, pctAvg,
                        color = 'firebrick', ax1 = ax2)
    plotBoxplotsArray(forecastIntAxis[1:-1], statsTech[1:-1], 
                    color = 'firebrick', ax = ax1)
    plotBoxplotsArray(forecastIntAxis[1:-1], statsAvg[1:-1], 
                    color = 'firebrick', ax = ax2)

    ax1.plot([1,15],[1,1],'C0', zorder=10, lw=2)
    ax2.plot([1,15],[1,1],'C2', zorder=10, lw=2)
    # ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=10)
    # ax2.set_xscale('log', base=10)
    # ax2.set_yscale('log', base=10)
    # ax1.set_ylim(0.1,10)
    # ax2.set_ylim(0.1,10)
    ax1.set_xlabel('Cumulative production / Last observed cumulative production')
    ax2.set_xlabel('Cumulative production / Last observed cumulative production')
    ax1.set_xlim(10**0, 10**tf[1])
    ax2.set_xlim(10**0, 10**tf[1])
    ax1.annotate('Technology-specific slope', 
                    xy=(5, 4), fontsize=12, 
                    color=cmapp,
                    ha='center',
                    va='center')
    ax2.annotate('Average slope', 
                    xy=(5, 4), fontsize=12,
                    color=cmapg,
                    ha='center',
                    va='center')

    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_ylabel('Unit cost error (actual/predicted)')
    ax[0].annotate('Future cumulative production'+\
                   ' / Current cumulative production',
                   (1.1,-.2), ha='center', va='center')
    
    # ax2.set_yticklabels([])

    lim = [0,0]

    if vert==True:
        positions = [1, 5]
    else:
        positions = [5, 1]

    # ax[2].axis('off')

    ax_ = ax[2]


    pctTech = np.array(2*pctTech[1:-1]).mean(axis=0).tolist()
    pctTech = [x for x in pctTech[1:]]
    pctAvg = np.array(2*pctAvg[1:-1]).mean(axis=0).tolist()
    pctAvg = [x for x in pctAvg[1:]]

    statsTech_, statsAvg_ = {}, {}
    for k in statsTech[0].keys():
        statsTech_[k] = 10**np.mean([(2*np.log10(s[k])) for s in statsTech[1:-1]])
        statsAvg_[k] = 10**np.mean([(2*np.log10(s[k])) for s in statsAvg[1:-1]])
    
    statsTech = [statsTech_]
    statsAvg = [statsAvg_]

    # plot boxplots and errorbars for 90% probability interval
    ax_.bxp([statsTech[0]], positions = [positions[0]],
            widths = 1, showfliers=False,
            boxprops=dict(color=cmapp, lw=2),
            manage_ticks=False,
            medianprops=dict(color='#931621', lw=1.5, zorder=2),
            capprops=dict(lw=0), whiskerprops=dict(lw=0),
            vert=vert)
    ax_.bxp([statsAvg[0]], positions = [positions[1]],
            widths = 1, showfliers=False,
            boxprops=dict(color=cmapg, lw=2),
            manage_ticks=False,
            medianprops=dict(color='#931621', lw=1.5, zorder=2),
            capprops=dict(lw=0), whiskerprops=dict(lw=0),
            vert=vert)
    
    ax_.errorbar(positions[0], 10**pctTech[1], xerr=0.5, c=cmapp)
    ax_.errorbar(positions[0], 10**pctTech[5], xerr=0.5, c=cmapp)
    ax_.errorbar(positions[1], 10**pctAvg[1], xerr=0.5, c=cmapg)
    ax_.errorbar(positions[1], 10**pctAvg[5], xerr=0.5, c=cmapg)
    ax_.fill_between([positions[0]-0.5,positions[0]+0.5], 
                        [10**pctTech[1],10**pctTech[1]],
                        [10**pctTech[5],10**pctTech[5]],
                        color=cmapp, alpha=0.1,
                        zorder=-2)
    ax_.fill_between([positions[0]-0.5,positions[0]+0.5], 
                        [10**pctTech[2],10**pctTech[2]],
                        [10**pctTech[4],10**pctTech[4]],
                        color=cmapp, alpha=0.4,
                        zorder=-2)
    ax_.fill_between([positions[1]-0.5,positions[1]+0.5], 
                        [10**pctAvg[1],10**pctAvg[1]],
                        [10**pctAvg[5],10**pctAvg[5]],
                        color=cmapg, alpha=0.1,
                        zorder=-2)
    ax_.fill_between([positions[1]-0.5,positions[1]+0.5], 
                        [10**pctAvg[2],10**pctAvg[2]],
                        [10**pctAvg[4],10**pctAvg[4]],
                        color=cmapg, alpha=0.4,
                        zorder=-2)            
    ax_.plot([-2,7], [1,1], color='k', zorder=-10, alpha=0.25)
    ax_.set_xlim(-1,7)
    ax_.set_xticks([positions[0],positions[1]],['Technology-specific','Average slope'])
    lim[0] = min(ax_.get_ylim()[0], lim[0])
    lim[1] = max(ax_.get_ylim()[1], lim[1])

    labs = ax_.get_xticklabels()
    if labs:
        labs[1].set_color(cmapg)
        labs[0].set_color(cmapp) 

    # lim[0] = min(lim[0], -lim[1])
    # lim[1] = max(lim[1], -lim[0])
    # if vert == False:
    #     ax_.set_xlim(lim)
    # else:
    #     ax_.set_ylim(lim)

    ax_.set_ylim(0.1,5)
    axes1=fig.add_axes([0.9,0.25,0.08,0.5])
    axes1.plot([-0.5,1.5],[0.5,0.5],'k', lw=2, zorder=-2)
    axes1.fill_between([-0.5,1.5],[0.25,0.25],[0.75,0.75], 
                       color='k', alpha=0.3, zorder=-10)
    axes1.fill_between([-.5,1.5],[0.05,0.05],[0.95,0.95], 
                       color='k', alpha=0.3, zorder=-10)
    # axes1.fill_between([-.5,1.5],[0,0],[1.0,1.0], 
    #                    color='k', alpha=0.1)
    axes1.plot([1.5,2,2,1.5],[0.25,0.25,0.75,0.75], color='k',
               linestyle='--',lw=0.5)
    axes1.plot([1.5,4,4,1.5],[0.05,0.05,0.95,0.95], color='k',
               linestyle='--',lw=0.5)

    axes1.annotate('50%', xy=(3, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7,
                   rotation=90)
    axes1.annotate('90%', xy=(5, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7,
                   rotation=90)
    axes1.annotate('Median', xy=(-2.5, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)
    # axes1.annotate('Max', xy=(0.5, 1.1), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7)
    # axes1.annotate('Min', xy=(0.5, -0.1), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7)

    # axes1.plot([0,1],[0.5,0.5],'k', lw=1)
    axes1.plot([0,1,1,0,0],[0.25,0.25,0.75,0.75,0.25], lw=2, color='k')
    axes1.plot([0,1],[0.5,0.5], lw=2, color='#931621', zorder=-1)
    # axes1.plot([0,1],[0,0], color='k', lw=1)
    # axes1.plot([0,1],[1,1], color='k', lw=1)
    # axes1.plot([0.5,0.5],[0,0.25], color='k', lw=1)
    # axes1.plot([0.5,0.5],[0.75,1], color='k', lw=1)

    axes1.set_xlim(-3.5,5)
    axes1.set_ylim(-0.5,1.5)
    axes1.set_xticks([])
    axes1.set_yticks([])
    axes1.axis('off')

    plt.subplots_adjust(bottom=0.15, left=0.075, right=0.875, wspace=0.15)

    return fig, ax


def plotForecastErrors(dferrTech, dferrAvg,
                       fOrd, samplingPoints,
                       trainErr = None, tOrd = None,
                       ):

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
                          color = cmapp, ax1 = ax[0], 
                          ax2 = ax[2], ax2b = ax2b)
    ax2b = plotPercentiles(forecastIntAxis, pctAvg,
                          countPointsTech, countTechsTech,
                          color = cmapg, ax1 = ax[1], 
                          ax2 = ax[2], ax2b = ax2b)
    plotBoxplotsArray(forecastIntAxis[1:-1], statsTech[1:-1], 
                        color = cmapp, ax = ax[0])
    plotBoxplotsArray(forecastIntAxis[1:-1], statsAvg[1:-1], 
                        color = cmapg, ax = ax[1])
    
    ax2b.yaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(integer=True))
    ax2b.minorticks_off()

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
                [0], [0], color=cmapp, lw=2, 
                label='Forecast error - Technology-specific'),
            matplotlib.lines.Line2D(\
                [0], [0], color=cmapg, lw=2, 
                label='Forecast error - Average slope'),
                            ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3)
    else:
        legend_elements = [
            matplotlib.lines.Line2D(\
                [0], [0], color=cmapp, lw=2, 
                label='Forecast error - Technology-specific'),
            matplotlib.lines.Line2D(\
                [0], [0], color=cmapg, lw=2, 
                label='Forecast error - Average slope'),
                            ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2)      

    axes1=fig.add_axes([0.85,0.35,0.125,0.3])
    axes1.plot([-0.5,1.5],[0.5,0.5],'k', lw=2, zorder=-2)
    axes1.fill_between([-0.5,1.5],[0.25,0.25],[0.75,0.75], 
                       color='k', alpha=0.3, zorder=-10)
    axes1.fill_between([-.5,1.5],[0.05,0.05],[0.95,0.95], 
                       color='k', alpha=0.2, zorder=-10)
    axes1.fill_between([-.5,1.5],[0,0],[1.0,1.0], 
                       color='k', alpha=0.1, zorder=-10)
    axes1.plot([1.5,2,2,1.5],[0.25,0.25,0.75,0.75], color='k',
               linestyle='--',lw=0.5)
    axes1.plot([1.5,4,4,1.5],[0.05,0.05,0.95,0.95], color='k',
               linestyle='--',lw=0.5)

    axes1.annotate('50%', xy=(3, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)
    axes1.annotate('90%', xy=(5, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)
    axes1.annotate('Median', xy=(-2.5, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)
    axes1.annotate('Max', xy=(0.5, 1.1), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)
    axes1.annotate('Min', xy=(0.5, -0.1), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)

    # axes1.plot([0,1],[0.5,0.5],'k', lw=1)
    axes1.plot([0,1,1,0,0],[0.25,0.25,0.75,0.75,0.25], lw=2, color='k')
    axes1.plot([0,1],[0.5,0.5], lw=2, color='#931621', zorder=-1)
    axes1.plot([0,1],[0,0], color='k', lw=1)
    axes1.plot([0,1],[1,1], color='k', lw=1)
    axes1.plot([0.5,0.5],[0,0.25], color='k', lw=1)
    axes1.plot([0.5,0.5],[0.75,1], color='k', lw=1)

    axes1.set_xlim(-3.5,5)
    axes1.set_ylim(-0.5,1.5)
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
                    boxprops=dict(color=cmapp, lw=2),
                    manage_ticks=False,
                    medianprops=dict(color='#931621', lw=1.5, zorder=2),
                    widths=0.8)
                ax[countax//dim][countax%dim].bxp(
                    [statsAvg[0]], 
                    positions=[3*counth+1],
                    showfliers=False,
                    boxprops=dict(color=cmapg, lw=2),
                    manage_ticks=False,
                    medianprops=dict(color='#931621', lw=1.5, zorder=2),
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

def plotBoxplotPvalues(t, t1, t2, z, z1, z2):
    
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    
    ax[0].boxplot(t, vert=False, showfliers=False)
    ax[0].plot([t1,t1],[0.5,1.5], 'r--', lw=1)
    ax[0].plot([t2,t2],[0.5,1.5], 'r--', lw=1)
    ax[0].set_title('Paired t-test')

    ax[1].boxplot(z, vert=False, showfliers=False)
    ax[1].plot([z1,z1], [0.5,1.5], 'r--', lw=1)
    ax[1].plot([z2,z2], [0.5,1.5], 'r--', lw=1)
    ax[1].set_title('Wilcoxon signed ranks test')

    return fig, ax

def plotRankTechnologies(fErrsTech, fErrsAvg,
                        tf, Ranges, 
                        samplingPoints, vert=True):

    positions = [-1,1]


    # select data
    dferrTech = fErrsTech[Ranges.index(tf)].copy()
    dferrAvg = fErrsAvg[Ranges.index(tf)].copy()

    fig, ax = plt.subplots(1,dferrTech['Tech'].nunique()+1,
                           sharey=True, figsize=(10,8))

    # get data to be plotted
    pctTech, _, forecastIntAxis, \
        _, _, statsTech = \
            analysisFunctions.\
                dataToPercentilesArray(dferrTech,
                                    tf[1], samplingPoints)
    pctAvg, _, _, \
        _, _, statsAvg = \
            analysisFunctions.\
                dataToPercentilesArray(dferrAvg,
                                    tf[1], samplingPoints)

    pctTech = np.array(2*pctTech[1:-1]).mean(axis=0).tolist()
    pctTech = [x for x in pctTech[1:]]
    pctAvg = np.array(2*pctAvg[1:-1]).mean(axis=0).tolist()
    pctAvg = [x for x in pctAvg[1:]]

    statsTech_, statsAvg_ = {}, {}
    for k in statsTech[0].keys():
        statsTech_[k] = 10**np.mean([(2*np.log10(s[k])) for s in statsTech[1:-1]])
        statsAvg_[k] = 10**np.mean([(2*np.log10(s[k])) for s in statsAvg[1:-1]])
    
    statsTech = [statsTech_]
    statsAvg = [statsAvg_]

    ax_ = ax[0]
    # plot boxplots and errorbars for 90% probability interval
    ax_.bxp([statsTech[0]], positions = [positions[0]],
            widths = 1, showfliers=False,
            boxprops=dict(color=cmapp, lw=2),
            manage_ticks=False,
            medianprops=dict(color='#931621', lw=1.5, zorder=2),
            capprops=dict(lw=0), whiskerprops=dict(lw=0),
            vert=vert)
    ax_.bxp([statsAvg[0]], positions = [positions[1]],
            widths = 1, showfliers=False,
            boxprops=dict(color=cmapg, lw=2),
            manage_ticks=False,
            medianprops=dict(color='#931621', lw=1.5, zorder=2),
            capprops=dict(lw=0), whiskerprops=dict(lw=0),
            vert=vert)
    
    ax_.errorbar(positions[0], 10**pctTech[1], xerr=0.5, c=cmapp)
    ax_.errorbar(positions[0], 10**pctTech[5], xerr=0.5, c=cmapp)
    ax_.errorbar(positions[1], 10**pctAvg[1], xerr=0.5, c=cmapg)
    ax_.errorbar(positions[1], 10**pctAvg[5], xerr=0.5, c=cmapg)
    ax_.fill_between([positions[0]-0.5,positions[0]+0.5], 
                        [10**pctTech[1],10**pctTech[1]],
                        [10**pctTech[5],10**pctTech[5]],
                        color=cmapp, alpha=0.1,
                        zorder=-2)
    ax_.fill_between([positions[0]-0.5,positions[0]+0.5], 
                        [10**pctTech[2],10**pctTech[2]],
                        [10**pctTech[4],10**pctTech[4]],
                        color=cmapp, alpha=0.4,
                        zorder=-2)
    ax_.fill_between([positions[1]-0.5,positions[1]+0.5], 
                        [10**pctAvg[1],10**pctAvg[1]],
                        [10**pctAvg[5],10**pctAvg[5]],
                        color=cmapg, alpha=0.1,
                        zorder=-2)
    ax_.fill_between([positions[1]-0.5,positions[1]+0.5], 
                        [10**pctAvg[2],10**pctAvg[2]],
                        [10**pctAvg[4],10**pctAvg[4]],
                        color=cmapg, alpha=0.4,
                        zorder=-2)

    ax_.set_xticklabels([])
    ax_.set_xticks([])
    ax_.set_xlabel('All technologies', 
                            rotation=90)
    ax_.plot([-2,2], [1,1], 'k--', lw=0.5, zorder=-10)
    ax_.set_xlim(-2,2)
    ax_.set_ylim(0.01,10)

    order = []
    techcount = []
    for tech in dferrTech['Tech'].unique():
        techcount.append([tech, dferrTech.loc[dferrTech['Tech']==tech].shape[0],
                          dferrTech.loc[dferrTech['Tech']==tech,'F p-value'].mean()])
    techcount = pd.DataFrame(techcount, columns=['Tech','count','F p-value'])
    for tech in techcount.sort_values(by='F p-value', ascending=False)['Tech'].tolist():
        sTech = dferrTech.loc[dferrTech['Tech']==tech]
        sAvg = dferrAvg.loc[dferrAvg['Tech']==tech]

        # get data to be plotted
        pctTech, _, forecastIntAxis, \
            _, _, statsTech = \
                analysisFunctions.\
                    dataToPercentilesArray(sTech,
                                        tf[1], samplingPoints)
        pctAvg, _, _, \
            _, _, statsAvg = \
                analysisFunctions.\
                    dataToPercentilesArray(sAvg,
                                        tf[1], samplingPoints)
    
        pctTech = np.array(2*pctTech[1:-1]).mean(axis=0).tolist()
        pctTech = [x for x in pctTech[1:]]
        pctAvg = np.array(2*pctAvg[1:-1]).mean(axis=0).tolist()
        pctAvg = [x for x in pctAvg[1:]]

        statsTech_, statsAvg_ = {}, {}
        for k in statsTech[0].keys():
            statsTech_[k] = 10**np.mean([(2*np.log10(s[k])) for s in statsTech[1:-1]])
            statsAvg_[k] = 10**np.mean([(2*np.log10(s[k])) for s in statsAvg[1:-1]])
        
        statsTech = [statsTech_]
        statsAvg = [statsAvg_]
        order.append([tech, statsTech[0]['med']])

    order = pd.DataFrame(order, columns=['Tech','med'])
    order = order.sort_values(by='med', ascending=False)

    # for tech in order['Tech'].tolist():
    count = 0
    for tech in techcount.sort_values(by='F p-value', ascending=False)['Tech'].tolist():
        count += 1
        sTech = dferrTech.loc[dferrTech['Tech']==tech]
        sAvg = dferrAvg.loc[dferrAvg['Tech']==tech]

        # get data to be plotted
        pctTech, _, forecastIntAxis, \
            _, _, statsTech = \
                analysisFunctions.\
                    dataToPercentilesArray(sTech,
                                        tf[1], samplingPoints)
        pctAvg, _, _, \
            _, _, statsAvg = \
                analysisFunctions.\
                    dataToPercentilesArray(sAvg,
                                        tf[1], samplingPoints)
    
        pctTech = np.array(2*pctTech[1:-1]).mean(axis=0).tolist()
        pctTech = [x for x in pctTech[1:]]
        pctAvg = np.array(2*pctAvg[1:-1]).mean(axis=0).tolist()
        pctAvg = [x for x in pctAvg[1:]]

        statsTech_, statsAvg_ = {}, {}
        for k in statsTech[0].keys():
            statsTech_[k] = 10**np.mean([(2*np.log10(s[k])) for s in statsTech[1:-1]])
            statsAvg_[k] = 10**np.mean([(2*np.log10(s[k])) for s in statsAvg[1:-1]])
        
        statsTech = [statsTech_]
        statsAvg = [statsAvg_]

        # ax_ = ax[order['Tech'].unique().tolist().index(tech)+1]
        ax_ = ax[count]
        # plot boxplots and errorbars for 90% probability interval
        ax_.bxp([statsTech[0]], positions = [positions[0]],
                widths = 1, showfliers=False,
                boxprops=dict(color=cmapp, lw=2),
                manage_ticks=False,
                medianprops=dict(color='#931621', lw=1.5, zorder=2),
                capprops=dict(lw=0), whiskerprops=dict(lw=0),
                vert=vert)
        ax_.bxp([statsAvg[0]], positions = [positions[1]],
                widths = 1, showfliers=False,
                boxprops=dict(color=cmapg, lw=2),
                manage_ticks=False,
                medianprops=dict(color='#931621', lw=1.5, zorder=2),
                capprops=dict(lw=0), whiskerprops=dict(lw=0),
                vert=vert)
        
        ax_.errorbar(positions[0], 10**pctTech[1], xerr=0.5, c=cmapp)
        ax_.errorbar(positions[0], 10**pctTech[5], xerr=0.5, c=cmapp)
        ax_.errorbar(positions[1], 10**pctAvg[1], xerr=0.5, c=cmapg)
        ax_.errorbar(positions[1], 10**pctAvg[5], xerr=0.5, c=cmapg)
        ax_.fill_between([positions[0]-0.5,positions[0]+0.5], 
                            [10**pctTech[1],10**pctTech[1]],
                            [10**pctTech[5],10**pctTech[5]],
                            color=cmapp, alpha=0.1,
                            zorder=-2)
        ax_.fill_between([positions[0]-0.5,positions[0]+0.5], 
                            [10**pctTech[2],10**pctTech[2]],
                            [10**pctTech[4],10**pctTech[4]],
                            color=cmapp, alpha=0.4,
                            zorder=-2)
        ax_.fill_between([positions[1]-0.5,positions[1]+0.5], 
                            [10**pctAvg[1],10**pctAvg[1]],
                            [10**pctAvg[5],10**pctAvg[5]],
                            color=cmapg, alpha=0.1,
                            zorder=-2)
        ax_.fill_between([positions[1]-0.5,positions[1]+0.5], 
                            [10**pctAvg[2],10**pctAvg[2]],
                            [10**pctAvg[4],10**pctAvg[4]],
                            color=cmapg, alpha=0.4,
                            zorder=-2)

        ax_.set_xticklabels([])
        ax_.set_xticks([])
        ax_.set_xlabel(tech.replace('_',' ')+' ('+str(round(techcount.loc[techcount['Tech']==tech,'F p-value'].values[0],3))+')', 
                                rotation=90)
        ax_.plot([-2,2], [1,1], 'k--', lw=0.5, zorder=-10)
        ax_.set_xlim(-2,2)

    plt.subplots_adjust(top=0.975, bottom=0.25, wspace=0.0, right=0.875, left=0.075)
    ax[0].set_ylabel('Unit cost error (Actual/Predicted)')
    ax[0].set_yscale('log', base=10)

    axes1=fig.add_axes([0.91,0.45,0.08,0.4])
    axes1.plot([-0.5,1.5],[0.5,0.5],'k', lw=2, zorder=-2)
    axes1.fill_between([-0.5,1.5],[0.25,0.25],[0.75,0.75], 
                       color='k', alpha=0.3, zorder=-10)
    axes1.fill_between([-.5,1.5],[0.05,0.05],[0.95,0.95], 
                       color='k', alpha=0.3, zorder=-10)
    # axes1.fill_between([-.5,1.5],[0,0],[1.0,1.0], 
    #                    color='k', alpha=0.1)
    axes1.plot([1.5,2,2,1.5],[0.25,0.25,0.75,0.75], color='k',
               linestyle='--',lw=0.5)
    axes1.plot([1.5,4,4,1.5],[0.05,0.05,0.95,0.95], color='k',
               linestyle='--',lw=0.5)

    axes1.annotate('50%', xy=(3, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7,
                   rotation=90)
    axes1.annotate('90%', xy=(5, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7,
                   rotation=90)
    axes1.annotate('Median', xy=(-2.5, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)
    # axes1.annotate('Max', xy=(0.5, 1.1), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7)
    # axes1.annotate('Min', xy=(0.5, -0.1), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7)

    # axes1.plot([0,1],[0.5,0.5],'k', lw=1)
    axes1.plot([0,1,1,0,0],[0.25,0.25,0.75,0.75,0.25], lw=2, color='k')
    axes1.plot([0,1],[0.5,0.5], lw=2, color='#931621', zorder=-1)
    # axes1.plot([0,1],[0,0], color='k', lw=1)
    # axes1.plot([0,1],[1,1], color='k', lw=1)
    # axes1.plot([0.5,0.5],[0,0.25], color='k', lw=1)
    # axes1.plot([0.5,0.5],[0.75,1], color='k', lw=1)

    axes1.set_xlim(-3.5,5)
    axes1.set_ylim(-0.5,1.5)
    axes1.set_xticks([])
    axes1.set_yticks([])
    axes1.axis('off')

    return fig, ax


    # # select data
    # dferrTech = fErrsTech[Ranges.index(tf)].copy()
    # dferrAvg = fErrsAvg[Ranges.index(tf)].copy()
    
    # dferrTech['Sector'] = [analysisFunctions.\
    #                        sectorsinv[tech] for tech in dferrTech['Tech']]
    # dferrTech = dferrTech.sort_values(by=['Sector','Tech'])

    # fig, ax = plt.subplots(1,dferrTech['Tech'].nunique()+1,
    #                        sharey=True, figsize=(10,8))

    # count = 0

    # # compute percentiles and boxplots stats
    # pctTech = analysisFunctions\
    #     .computeTechWeightedPercentiles(dferrTech)
    # pctAvg = analysisFunctions\
    #     .computeTechWeightedPercentiles(dferrAvg)
    # statsTech = analysisFunctions\
    #     .computeBoxplots(np.array([[0,*pctTech]]),
    #                     [0], positions=[1,3,4,5,7],
    #                     log = True)
    # statsAvg = analysisFunctions\
    #     .computeBoxplots(np.array([[0,*pctAvg]]),
    #                     [0], positions=[1,3,4,5,7],
    #                     log = True)

    # # plot boxplots and errorbars for 90% probability interval
    # ax[count].bxp([statsTech[0]], positions = [positions[0]],
    #         widths = 0.3, showfliers=False,
    #         boxprops=dict(color=cmapp, lw=2),
    #         manage_ticks=False,
    #         medianprops=dict(color='#931621', lw=1.5, zorder=2),
    #         capprops=dict(lw=0), whiskerprops=dict(lw=0))
    # ax[count].bxp([statsAvg[0]], positions = [positions[1]],
    #         widths = 0.3, showfliers=False,
    #         boxprops=dict(color=cmapg, lw=2),
    #         manage_ticks=False,
    #         medianprops=dict(color='#931621', lw=1.5, zorder=2),
    #         capprops=dict(lw=0), whiskerprops=dict(lw=0))
    # ax[count].set_xticklabels([])
    # ax[count].set_xticks([])
    # ax[count].set_xlabel('All technologies', 
    #                         rotation=90)
    # ax[count].plot([1,1],[-2,2], 'k--', lw=0.5, zorder=-10)
    # ax[count].set_xlim(-0.4,0.4)
    # ax[count].errorbar(positions[0], 10**np.array(pctTech[1]), xerr=0.15, c=cmapp)
    # ax[count].errorbar(positions[0], 10**np.array(pctTech[5]), xerr=0.15, c=cmapp)
    # ax[count].errorbar(positions[1], 10**np.array(pctAvg[1]), xerr=0.15, c=cmapg)
    # ax[count].errorbar(positions[1], 10**np.array(pctAvg[5]), xerr=0.15, c=cmapg)
    # ax[count].fill_between([positions[0]-0.15,positions[0]+0.15], 
    #                         [10**np.array(pctTech[1]),10**np.array(pctTech[1])],
    #                         [10**np.array(pctTech[5]),10**np.array(pctTech[5])],
    #                     color=cmapp, alpha=0.1,
    #                     zorder=-1)
    # ax[count].fill_between([positions[0]-0.15,positions[0]+0.15], 
    #                         [10**np.array(pctTech[2]),10**np.array(pctTech[2])],
    #                         [10**np.array(pctTech[4]),10**np.array(pctTech[4])],
    #                     color=cmapp, alpha=0.4,
    #                     zorder=-1)
    # ax[count].fill_between([positions[1]-0.15,positions[1]+0.15], 
    #                         [10**np.array(pctAvg[1]),10**np.array(pctAvg[1])],
    #                         [10**np.array(pctAvg[5]),10**np.array(pctAvg[5])],
    #                     color=cmapg, alpha=0.1,
    #                     zorder=-1)
    # ax[count].fill_between([positions[1]-0.15,positions[1]+0.15], 
    #                         [10**np.array(pctAvg[2]),10**np.array(pctAvg[2])],
    #                         [10**np.array(pctAvg[4]),10**np.array(pctAvg[4])],
    #                     color=cmapg, alpha=0.4,
    #                     zorder=-1) 

    # count += 1

    # for tech in dferrTech['Tech'].unique():

    #     selTech = dferrTech.loc[dferrTech['Tech'] == tech]
    #     selAvg = dferrAvg.loc[dferrAvg['Tech'] == tech]

    #     # compute percentiles and boxplots stats
    #     pctTech = analysisFunctions\
    #         .computeTechWeightedPercentiles(selTech)
    #     pctAvg = analysisFunctions\
    #         .computeTechWeightedPercentiles(selAvg)
    #     statsTech = analysisFunctions\
    #         .computeBoxplots(np.array([[0,*pctTech]]),
    #                         [0], positions=[1,3,4,5,7],
    #                         log = True)
    #     statsAvg = analysisFunctions\
    #         .computeBoxplots(np.array([[0,*pctAvg]]),
    #                         [0], positions=[1,3,4,5,7],
    #                         log = True)

    #     # plot boxplots and errorbars for 90% probability interval
    #     ax[count].bxp([statsTech[0]], positions = [positions[0]],
    #             widths = 0.3, showfliers=False,
    #             boxprops=dict(color=cmapp, lw=2),
    #             manage_ticks=False,
    #             medianprops=dict(color='#931621', lw=1.5, zorder=2),
    #             capprops=dict(lw=0), whiskerprops=dict(lw=0)),
    #     ax[count].bxp([statsAvg[0]], positions = [positions[1]],
    #             widths = 0.3, showfliers=False,
    #             boxprops=dict(color=cmapg, lw=2),
    #             manage_ticks=False,
    #             medianprops=dict(color='#931621', lw=1.5, zorder=2),
    #             capprops=dict(lw=0), whiskerprops=dict(lw=0))

    #     ax[count].errorbar(positions[0], 10**np.array(pctTech[1]), xerr=0.15, c=cmapp)
    #     ax[count].errorbar(positions[0], 10**np.array(pctTech[5]), xerr=0.15, c=cmapp)
    #     ax[count].errorbar(positions[1], 10**np.array(pctAvg[1]), xerr=0.15, c=cmapg)
    #     ax[count].errorbar(positions[1], 10**np.array(pctAvg[5]), xerr=0.15, c=cmapg)
    #     ax[count].fill_between([positions[0]-0.15,positions[0]+0.15], 
    #                            [10**np.array(pctTech[1]),10**np.array(pctTech[1])],
    #                            [10**np.array(pctTech[5]),10**np.array(pctTech[5])],
    #                         color=cmapp, alpha=0.1,
    #                         zorder=-1)
    #     ax[count].fill_between([positions[0]-0.15,positions[0]+0.15], 
    #                            [10**np.array(pctTech[2]),10**np.array(pctTech[2])],
    #                            [10**np.array(pctTech[4]),10**np.array(pctTech[4])],
    #                         color=cmapp, alpha=0.4,
    #                         zorder=-1)
    #     ax[count].fill_between([positions[1]-0.15,positions[1]+0.15], 
    #                            [10**np.array(pctAvg[1]),10**np.array(pctAvg[1])],
    #                            [10**np.array(pctAvg[5]),10**np.array(pctAvg[5])],
    #                         color=cmapg, alpha=0.1,
    #                         zorder=-1)
    #     ax[count].fill_between([positions[1]-0.15,positions[1]+0.15], 
    #                            [10**np.array(pctAvg[2]),10**np.array(pctAvg[2])],
    #                            [10**np.array(pctAvg[4]),10**np.array(pctAvg[4])],
    #                         color=cmapg, alpha=0.4,
    #                         zorder=-1)

    #     ax[count].set_xticklabels([])
    #     ax[count].set_xticks([])
    #     ax[count].set_xlabel(tech.replace('_',' '), 
    #                             rotation=90)
    #     ax[count].plot([-2,2],[-1,1], 'k--', lw=0.5, zorder=-10)
    #     ax[count].set_xlim(-0.4,0.4)

    #     count += 1
     
    # ax[0].set_yscale('log', base=10)


    # axes1=fig.add_axes([0.35,0.01,0.3,0.08])
    # axes1.fill_between([0.25,0.75],[0,0],[1,1],
    #                    color='k', alpha=0.3, zorder=-10)
    # axes1.fill_between([0.05,0.95],[0,0],[1,1],
    #                    color='k', alpha=0.3, zorder=-10)
    # axes1.plot([0.25,0.25,0.75,0.75], [-0.5,-2,-2,-0.5], color='k',
    #            linestyle='--',lw=0.5)
    # axes1.plot([0.05,0.05,0.95,0.95], [-0.5,-4,-4,-0.5], color='k',
    #            linestyle='--',lw=0.5)
    # axes1.plot([0.25,0.25,0.75,0.75,0.25], [0,1,1,0,0],lw=2, color='k')
    # axes1.plot([0.5,0.5], [0,1],lw=2, color='#931621', zorder=-1)
    # axes1.plot([0,0], [0,1],color='k', lw=1)
    # axes1.plot([1,1], [0,1],color='k', lw=1)
    # axes1.plot([0,0.25], [0.5,0.5],color='k', lw=1)
    # axes1.plot([0.75,1], [0.5,0.5],color='k', lw=1)

    # axes1.annotate('50%', xy=(0.5, -3), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7,
    #                )
    # axes1.annotate('90%', xy=(0.5,-5), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7,
    #                )
    # axes1.annotate('Median', xy=(0.5, 2.5), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7)
    # axes1.annotate('Max', xy=(1.1,0.5), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7)
    # axes1.annotate('Min', xy=(-0.1, 0.5), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7)

    # axes1.set_ylim(-6,3)
    # axes1.set_xlim(-0.5,1.5)
    # axes1.set_xticks([])
    # axes1.set_yticks([])
    # axes1.axis('off')

    #     ax[count].set_xticklabels([])
    #     ax[count].set_xticks([])
    #     ax[count].set_xlabel(tech.replace('_',' '), 
    #                             rotation=90)
    #     ax[count].plot([-2,2],[-1,1], 'k--', lw=0.5, zorder=-10)
    #     ax[count].set_xlim(-0.4,0.4)

    # plt.subplots_adjust(left=0.2, right=0.8, 
    #                     top=0.975, bottom=0.175, hspace=0.0)

    # return fig, ax


def plotObsPred(obsErr, predErr, pred2Err,
                              tf, Ranges, 
                            samplingPoints, ):
    

    obserr = obsErr[Ranges.index(tf)]
    prederr = predErr[Ranges.index(tf)]
    pred2err = pred2Err[Ranges.index(tf)]


    fig, ax = plt.subplots(1, 3, 
                           sharey=True, sharex=True,
                           figsize=(15,5))

    ax[0].scatter(10**obserr['Forecast horizon'].values,
                  10**obserr['Error'].values,
                  marker='.', color='firebrick',
                  edgecolor='None',
                  zorder=-2, alpha=0.05)
    ax[1].scatter(10**prederr['Forecast horizon'].values,
                  10**prederr['Error'].values,
                  marker='.', color='C0', 
                  edgecolor='None',
                  zorder=-2, alpha=0.05)

    # get data to be plotted
    pctObs, _, forecastIntAxis, \
        _, _, statsObs = \
            analysisFunctions.\
                dataToPercentilesArray(obserr,
                                    tf[1], samplingPoints)
    pctTech, _, forecastIntAxis, \
        _, _, statsTech = \
            analysisFunctions.\
                dataToPercentilesArray(prederr,
                                    tf[1], samplingPoints)
    pctAvg, _, _, \
        _, _, statsAvg = \
            analysisFunctions.\
                dataToPercentilesArray(pred2err,
                                    tf[1], samplingPoints)
    
    plotPercentiles(forecastIntAxis, pctObs,
                        color = 'firebrick', ax1 = ax[0])
    plotPercentiles(forecastIntAxis, pctTech,
                        color = cmapp, ax1 = ax[1])
    # plotPercentiles(forecastIntAxis, pctAvg,
    #                     color = cmapg, ax1 = ax[2])
    plotBoxplotsArray(forecastIntAxis[1:-1], statsObs[1:-1], 
                    color = 'firebrick', ax = ax[0])
    plotBoxplotsArray(forecastIntAxis[1:-1], statsTech[1:-1], 
                    color = cmapp, ax = ax[1])
    # plotBoxplotsArray(forecastIntAxis[1:-1], statsAvg[1:-1], 
    #                 color = cmapg, ax = ax[2])
    xint = np.linspace(0,tf[1],100)
    ax[2].plot(10**xint,[10**(x*-0.37274962196044475) for x in xint], lw=2, color = cmapg)

    ax[0].plot([1,1e6],[1,1],'k', zorder=-10)
    ax[1].plot([1,1e6],[1,1],'k', zorder=-10)
    ax[2].plot([1,1e6],[1,1],'k', zorder=-10)
    # ax[0].set_xscale('log', base=10)
    # ax1.set_yscale('log', base=10)
    # ax2.set_xscale('log', base=10)
    # ax2.set_yscale('log', base=10)
    # ax1.set_ylim(0.1,10)
    # ax2.set_ylim(0.1,10)
    ax[1].set_xlabel('Cumulative production / Last observed cumulative production')
    # ax2.set_xlabel('Cumulative production / Last observed cumulative production')
    ax[1].set_xlim(10**0, 10**tf[1])
    # ax2.set_xlim(10**0, 10**tf[1])
    ax[0].annotate('Observations', 
                    xy=(5, 1.75), fontsize=12, 
                    color='firebrick',
                    ha='center',
                    va='center')
    ax[1].annotate('Technology-specific slope\nForecast', 
                    xy=(5, 1.75), fontsize=12, 
                    color=cmapp,
                    ha='center',
                    va='center')
    ax[2].annotate('Average slope\nForecast', 
                    xy=(5, 1.75), fontsize=12,
                    color=cmapg,
                    ha='center',
                    va='center')

    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_ylabel('Unit cost / Last observed unit cost')
    # ax[0].annotate('Cumulative production'+\
    #                ' / Last observed cumulative production',
    #                (1.1,-.2), ha='center', va='center')
    ax[0].set_ylim(0,2)

    axes1=fig.add_axes([0.9,0.25,0.08,0.5])
    axes1.plot([-0.5,1.5],[0.5,0.5],'k', lw=2, zorder=-2)
    axes1.fill_between([-0.5,1.5],[0.25,0.25],[0.75,0.75], 
                       color='k', alpha=0.3, zorder=-10)
    axes1.fill_between([-.5,1.5],[0.05,0.05],[0.95,0.95], 
                       color='k', alpha=0.3, zorder=-10)
    # axes1.fill_between([-.5,1.5],[0,0],[1.0,1.0], 
    #                    color='k', alpha=0.1)
    axes1.plot([1.5,2,2,1.5],[0.25,0.25,0.75,0.75], color='k',
               linestyle='--',lw=0.5)
    axes1.plot([1.5,4,4,1.5],[0.05,0.05,0.95,0.95], color='k',
               linestyle='--',lw=0.5)

    axes1.annotate('50%', xy=(3, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7,
                   rotation=90)
    axes1.annotate('90%', xy=(5, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7,
                   rotation=90)
    axes1.annotate('Median', xy=(-2.5, 0.5), 
                   xycoords='data', ha='center', 
                   va='center', fontsize=7)
    # axes1.annotate('Max', xy=(0.5, 1.1), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7)
    # axes1.annotate('Min', xy=(0.5, -0.1), 
    #                xycoords='data', ha='center', 
    #                va='center', fontsize=7)

    # axes1.plot([0,1],[0.5,0.5],'k', lw=1)
    axes1.plot([0,1,1,0,0],[0.25,0.25,0.75,0.75,0.25], lw=2, color='k')
    axes1.plot([0,1],[0.5,0.5], lw=2, color='#931621', zorder=-1)
    # axes1.plot([0,1],[0,0], color='k', lw=1)
    # axes1.plot([0,1],[1,1], color='k', lw=1)
    # axes1.plot([0.5,0.5],[0,0.25], color='k', lw=1)
    # axes1.plot([0.5,0.5],[0.75,1], color='k', lw=1)

    axes1.set_xlim(-3.5,5)
    axes1.set_ylim(-0.5,1.5)
    axes1.set_xticks([])
    axes1.set_yticks([])
    axes1.axis('off')

    plt.subplots_adjust(bottom=0.15, left=0.075, right=0.875, wspace=0.15)


    return fig, ax


def plotR2Grid(slopeVals, Ranges):

    ndim = int(np.sqrt(len(Ranges)))
    fig, ax = plt.subplots(ndim, ndim, figsize=(15,10), sharex=True, sharey=True)

    count = 0
    for r in Ranges:
        slopevals = slopeVals[Ranges.index(r)]
        # slopevals = slopevals.loc[slopevals['Tech'].isin(
        #     ['Fotovoltaica','Transistor','DRAM','Laser_Diode','Hard_Disk_Drive']
        # ['Wind_Turbine_2_(Germany)', 'Fotovoltaica', 'Photovoltaics_2',
        # 'Titanium_Sponge', 'Wind_Electricity', 'Transistor', 'Photovoltaics_4',
        # 'DRAM', 'Ethanol_2', 'Monochrome_Television', 'Laser_Diode',
        # 'Capillary_DNA_Sequencing', 'Photovoltaics', 'Solar_Water_Heaters',
        # 'Wind_Turbine_(Denmark)', 'Hard_Disk_Drive', 'Primary_Magnesium',
        # 'Wheat_(US)', 'Wind_Power', 'Polystyrene']
                    # )]

        model = sm.OLS(slopevals['Slope val'], sm.add_constant(slopevals['Slope training']))
        result = model.fit()
        print(r, result.rsquared)
        ax[int(count/ndim)][count%ndim].scatter(slopevals['Slope training'], slopevals['Slope val'],
                                                color='none', edgecolor='firebrick', s=10)
        ax[int(count/ndim)][count%ndim].set_title('R2 = '+\
                                                  str(round(result.rsquared,2))+\
                                                    ', N = '+\
                                                        str(slopevals['Slope training'].count())+\
                                                            ', T = '+\
                                                                str(slopevals['Tech'].nunique()))
        ax[int(count/ndim)][count%ndim].plot([slopevals['Slope training'].min(),
                                              slopevals['Slope training'].max()],
                                              [slopevals['Slope val'].mean(),
                                               slopevals['Slope val'].mean()], 'k--', zorder=-2)
        print(slopevals['Tech'].unique())
        # model = sm.OLS(slopevals['Slope val'], sm.add_constant(slopevals['Average slope']))
        # result = model.fit()
        # ax[count%ndim][int(count/ndim)].scatter(slopevals['Average slope'], slopevals['Slope val'], color='red', alpha=0.5)
        count += 1
    
    ax[0][0].annotate('10$^{0.5}$', xy=(0.5, 1.2), xycoords='axes fraction', fontsize=12)
    ax[0][0].annotate('10$^{0.5}$', xy=(-.5, 0.5), xycoords='axes fraction', fontsize=12)
    ax[0][1].annotate('10$^{1}$', xy=(0.5, 1.2), xycoords='axes fraction', fontsize=12)
    ax[1][0].annotate('10$^{1}$', xy=(-.5, 0.5), xycoords='axes fraction', fontsize=12)
    ax[0][2].annotate('10$^{2}$', xy=(0.5, 1.2), xycoords='axes fraction', fontsize=12)
    ax[2][0].annotate('10$^{2}$', xy=(-.5, 0.5), xycoords='axes fraction', fontsize=12)
    ax[1][0].annotate('Traning horizon', xy=(-.75,0.5), 
                      xycoords='axes fraction', rotation=90, 
                      ha='center', va='center',
                      fontsize=12)
    ax[0][1].annotate('Forecast horizon', xy=(0.5,1.35), 
                      xycoords='axes fraction', ha='center', va='center',
                      fontsize=12)
    ax[1][0].set_ylabel('Slope of the second part')
    ax[2][1].set_xlabel('Slope of the first part')

    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.2, right=0.8)

    return fig, ax

def plotR2Contour(slopeVals, Ranges):

    fig, ax = plt.subplots(1,3,figsize=(12,5.25))

    for i in range(3):

        ax_ = ax[i]

        if i==1:
            techs = slopeVals[1]['Tech'].unique()
            title = 'Technologies covering 2 orders of magnitude ('+str(slopeVals[1]['Tech'].nunique())+')'
        elif i==2:
            techs = slopeVals[5]['Tech'].unique()
            title = 'Technologies covering 3 orders of magnitude ('+str(slopeVals[5]['Tech'].nunique())+')'
        else:
            techs = slopeVals[0]['Tech'].unique()
            title = 'Technologies covering 1 order of magnitude ('+str(slopeVals[0]['Tech'].nunique())+')'

        r2 = []
        for r in Ranges:
            if r[0]+r[1]>1 and i==0:
                r2.append(np.nan)
                continue
            if r[0]+r[1]>2 and i==1:
                r2.append(np.nan)
                continue
            if r[0]+r[1]>3 and i==2:
                r2.append(np.nan)
                continue
            slopevals = slopeVals[Ranges.index(r)]
            slopevals = slopevals.loc[slopevals['Tech'].isin(techs)]
            model = sm.OLS(slopevals['Slope val'], sm.add_constant(slopevals['Slope training']))
            result = model.fit()
            r2.append(result.rsquared)
        print(r2)
        r2 = np.array(r2).reshape(3,3)
        r2 = r2[::-1]
        print(r2)
        im = ax_.imshow(r2, cmap=cmcrameri.cm.batlow, vmin=0, vmax=0.43)
        ax_.set_xticks([0,1,2],[0.5,1,2])
        ax_.set_yticks([0,1,2],[2,1,0.5])
        ax_.set_title(title)
    ax[0].set_ylabel('Training horizon')
    ax[1].set_xlabel('Forecast horizon')
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Fraction of variance of future slope explained by measured slope')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)

    return fig, ax



def plotSignificance(testResDetail, Ranges):
    cmap = matplotlib.colors.ListedColormap([cmapg, 'firebrick', cmapp])
    fig, ax = plt.subplots(1,1)
    gridSign = []
    for ords in Ranges:
        sign = []
        sel = testResDetail.loc[(testResDetail['tOrd']==ords[0])&\
                                (testResDetail['fOrd']==ords[1])]
        for t in testResDetail['Tech'].unique():
            if not(sel.loc[sel['Tech']==t].empty):
                if sel.loc[sel['Tech']==t,'pvalue'].values[0] < 0.05:
                    if sel.loc[sel['Tech']==t,'diff'].values < 0 :
                        sign.append(1)
                    else:
                        sign.append(-1)
                else:
                    sign.append(0)
            else:
                sign.append(np.nan)
        gridSign.append(sign)
    gridSign = np.array(gridSign).transpose()
    im = ax.imshow(gridSign, cmap=cmap, aspect='auto', alpha=0.8)
    for i in range(gridSign.shape[1]):
        for j in range(gridSign.shape[0]):
            if gridSign[j,i] == 0:
                ax.plot([i-0.5,i+1-0.5],[j-0.5,j+1-0.5], color='grey')
                ax.plot([i+1-0.5,i-0.5],[j-0.5,j+1-0.5], color='grey')

    testResDetail['Label'] = testResDetail['Sector'] +' - '+ testResDetail['Tech']
    ax.set_yticks([x for x in range(testResDetail['Tech'].nunique())],[t.replace('_',' ') for t in testResDetail['Label'].unique()])
    ax.set_xticks([x for x in range(len(Ranges))],['('+str(int(10**x[0]))+', '+str(int(10**x[1]))+')' for x in Ranges], rotation=90)
    ax.set_ylabel('Technology')
    ax.set_xlabel('Multiplicative increase in cumulative production (Training, Forecast) ')
    plt.subplots_adjust(left=0.5, right=0.95, top=0.95, bottom=0.25)
    cbar = fig.colorbar(im, ax=ax, boundaries=[-1,-0.5,.5,1], 
                        values=[-1,0,1], orientation='horizontal',
                        cax=fig.add_axes([0.5,0.05,0.3,0.025]))
    cbar.set_ticks([-0.75,0,.75])
    cbar.ax.plot([-.5,.5],[0,1], color='grey')
    cbar.ax.plot([.5,-.5],[0,1], color='grey')
    cbar.set_ticklabels(['Avg','No difference','Tech'])

    return fig, ax