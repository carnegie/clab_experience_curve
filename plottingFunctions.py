import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib
import analysisFunctions

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
            matplotlib.lines.Line2D([0],[0],lw=0,color='royalblue',marker='o', label='Energy'),
            matplotlib.lines.Line2D([0],[0],lw=0,color='black',marker='o', label='Chemicals'),
            matplotlib.lines.Line2D([0],[0],lw=0,color='red',marker='o', label='Hardware'),  
            matplotlib.lines.Line2D([0],[0],lw=0,color='forestgreen',marker='o', label='Consumer goods'),
            matplotlib.lines.Line2D([0],[0],lw=0,color='cyan',marker='o', label='Food'),
            matplotlib.lines.Line2D([0],[0],lw=0,color='darkmagenta',marker='o', label='Genomics')
        ]
        figscatter.legend(handles=legend_elements, loc='center right', title='Sector')
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
    better = 0 # counter for number of technologies with lower MSE using average slope

    # create figures
    fig2 , ax2 = plt.subplots(10,9, figsize=(13,7))
    fig3 , ax3 = plt.subplots(10,9, figsize=(13,7))
    fig4, ax4 = plt.subplots(10,9, figsize=(13,7))

    while count - countt < len(uc):

        # extract relevant data
        uc_ = uc[count - countt]
        cp_ = np.concatenate([cpCal[count - countt], cpVal[count - countt]])
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
        ax3[int(count/9)][count%9].set_xlim((-max(np.abs(xlim)),max(np.abs(xlim))))
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
        'The sum of squared errors is lower \n using the average technological slope'+\
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
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10,6))
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
        ax[count//3][count%3].bar([0,1],
                                [tech[count][1], avg[count][1]], color=analysisFunctions.sectorsColor[tech[count][0]])
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
                #    color=analysisFunctions.sectorsColor[analysisFunctions.sectorsinv[tech]],
                     lw=0.5)
        # update labels
        labs.append(tech + ' (' + str(sel.shape[0]) + ')')
        count += 1
    ax[0].set_yticks([x for x in range(len(df['Tech'].unique()))], labs, fontsize=5)
    ax[0].set_ylabel('Technology')
    # sort orders of magnitude to plot cumulative production orders of magnitude covered
    oOfM.sort()
    # compute number of samples as orders of magnitude covered increase
    density = [[0,86]]
    for el in oOfM:
        density.append([el[0], density[-1][1]-1])
    density = np.array(density)
    # plot number of technologies per cumulative production orders of magnitude covered
    ax[1].plot(density[:,0], density[:,1], color='k', lw=2)
    ax[1].set_xlabel('Orders of magnitude of cumulative production')
    ax[1].set_ylabel('Number of technologies available')
    plt.subplots_adjust(top=0.98, bottom=0.09, hspace=0.15, left=0.2)
    return fig, ax

def plotForecastErrors(dferrTech, dferrAvg, trainErr, 
                       tOrd, fOrd, samplingPoints):

    # set colors to be used
    cmapp = matplotlib.colormaps['Purples']
    cmapg = matplotlib.colormaps['Greens']

    # create figures
    fig, ax = plt.subplots(3,1, sharex=True, figsize=(9,8))
    figb, axb = plt.subplots(3,1, sharex=True, figsize=(9,8))

    ### plot training error
    npoints = int(samplingPoints * tOrd)
    # create centered cumulative production array for plotting
    trainInt = np.linspace(-tOrd-tOrd/npoints/2, 0+ tOrd/npoints/2, npoints+2)
    trainInt = np.linspace(-tOrd-tOrd/npoints, 0, npoints+2)
    trainIntAxis = [trainInt[1]]
    for idx in range(1,len(trainInt)-1):
        trainIntAxis.append(trainInt[idx]+\
                            (trainInt[idx+1]-trainInt[idx])/2)
    trainIntAxis.append(0)
    countPoints = []
    countTechs = []
    pctTrain = []
    for i in range(len(trainInt)):
        if i == 0:
            sel = trainErr.loc[(trainErr['Forecast horizon']<=trainInt[i+1])].copy()
        elif i == len(trainInt)-1:
            sel = trainErr.loc[(trainErr['Forecast horizon']==trainInt[i])].copy()
        else:
            sel = trainErr.loc[(trainErr['Forecast horizon']>trainInt[i]) &\
                        (trainErr['Forecast horizon']<=trainInt[i+1])].copy()
        if sel.shape[0] == 0:
            pctTrain.append([trainInt[i],np.nan,np.nan,np.nan,np.nan,np.nan])
            countPoints.append(0)
            countTechs.append(0)
            continue
        countPoints.append(sel.shape[0])
        countTechs.append(sel['Tech'].nunique())
        for tt in sel['Tech'].unique():
            sel.loc[sel['Tech']==tt,'Weights'] = 1/sel.loc[sel['Tech']==tt].count()[0]
        sel = sel.sort_values(by='Error', ascending=True)
        cumsum = sel['Weights'].cumsum().round(4)
        pt = []
        for q in [0,10,25,50,75,90,100]:
        # for q in [0,10,20,30,40,50,60,70,80,90,100]:
            cutoff = sel['Weights'].sum() * q/100
            pt.append(sel['Error'][cumsum >= cutoff.round(4)].iloc[0])
        pctTrain.append([trainInt[i],*pt])
    pctTrain = np.array(pctTrain)
    ax[0].plot([10**x for x in trainIntAxis],10**pctTrain[:,4], color='b', lw=2)
    for r in range(2,-1,-1):
        ax[0].fill_between([10**x for x in trainIntAxis], 10**pctTrain[:,1+r], 10**pctTrain[:,-r-1], alpha=0.1+0.2*r, color='b', zorder=-2-r, lw=0)
    ax[2].plot([1,1],[0,100**100],'k')
    ax[2].plot([10**x for x in trainIntAxis],countPoints, color='k', lw=2)
    ax[2].set_ylim(0,max(countPoints)*1.1)
    ax2 = ax[2].twinx()
    ax2.plot([10**x for x in trainIntAxis],countTechs, color='red', lw=2)

    for x in trainIntAxis:
        stats = {}
        labs = ['whislo', 'q1', 'med', 'q3', 'whishi']
        count_ = 1
        for l in labs:
            stats[l] = 10**pctTrain[trainIntAxis.index(x),count_]
            count_ += 1
            if count_ == 2 or count_ == 6:
                count_ += 1
        axb[0].bxp([stats], positions = [10**x], widths = (10**x)/8, showfliers=False, boxprops=dict(color='b', lw=2), manage_ticks=False)
    axb[0].set_yscale('log', base=10)
    axb[2].plot([1,1],[0,100**100],'k')
    axb[2].plot([10**x for x in trainIntAxis],countPoints, color='k', lw=2)
    axb[2].set_ylim(0,max(countPoints)*1.1)
    axb[0].set_xscale('log', base=10)
    ax2b = axb[2].twinx()
    ax2b.plot([10**x for x in trainIntAxis],countTechs, color='red', lw=2)

    # plot forecast error
    npoints = int(samplingPoints * fOrd)
    forecastInt = np.linspace(0-fOrd/npoints, fOrd, npoints+2)
    forecastIntAxis = [0]
    for idx in range(1,len(forecastInt)-1):
        forecastIntAxis.append(forecastInt[idx]+\
                                (forecastInt[idx+1]-forecastInt[idx])/2)
    forecastIntAxis.append(fOrd)
    pctTech, pctAvg = [], []
    countPoints = []
    countTechs = []
    for i in range(len(forecastInt)):
        if i == 0:
            sel1 = dferrTech.loc[(dferrTech['Forecast horizon']==0)].copy()
            sel2 = dferrAvg.loc[(dferrAvg['Forecast horizon']==0)].copy()
        elif i == len(forecastInt)-1:
            sel1 = dferrTech.loc[(dferrTech['Forecast horizon']>=forecastInt[i])].copy()
            sel2 = dferrAvg.loc[(dferrAvg['Forecast horizon']>=forecastInt[i])].copy()
        else:
            sel1 = dferrTech.loc[(dferrTech['Forecast horizon']>forecastInt[i]) &\
                        (dferrTech['Forecast horizon']<=forecastInt[i+1])].copy()
            sel2 = dferrAvg.loc[(dferrAvg['Forecast horizon']>forecastInt[i]) &\
                        (dferrAvg['Forecast horizon']<=forecastInt[i+1])].copy()
        if sel1.shape[0] == 0:
            pctTech.append([forecastInt[i],np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
            pctAvg.append([forecastInt[i],np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
            countPoints.append([0,0])
            countTechs.append([0,0])
            continue
        countPoints.append([sel1.shape[0],sel2.shape[0]])
        countTechs.append([sel1['Tech'].nunique(),sel2['Tech'].nunique()])
        for tt in sel1['Tech'].unique():
            sel1.loc[sel1['Tech']==tt,'Weights'] = 1/sel1.loc[sel1['Tech']==tt].count()[0]
        for tt in sel2['Tech'].unique():
            sel2.loc[sel2['Tech']==tt,'Weights'] = 1/sel2.loc[sel2['Tech']==tt].count()[0]
        sel1 = sel1.sort_values(by='Error', ascending=True)
        sel2 = sel2.sort_values(by='Error', ascending=True)
        cumsum1 = sel1['Weights'].cumsum().round(4)
        cumsum2 = sel2['Weights'].cumsum().round(4)
        pt1, pt2 = [], []
        for q in [0,10,25,50,75,90,100]:
        # for q in [0,10,20,30,40,50,60,70,80,90,100]:
            cutoff1 = sel1['Weights'].sum() * q/100
            cutoff2 = sel2['Weights'].sum() * q/100
            pt1.append(sel1['Error'][cumsum1 >= cutoff1.round(4)].iloc[0])
            pt2.append(sel2['Error'][cumsum2 >= cutoff2.round(4)].iloc[0])
        pctTech.append([forecastInt[i],*pt1])
        pctAvg.append([forecastInt[i],*pt2])
    pctTech = np.array(pctTech)
    pctAvg = np.array(pctAvg)
    ax[0].plot([10**x for x in forecastIntAxis],10**pctTech[:,4], color=cmapp(0.7), lw=2)
    ax[1].plot([10**x for x in forecastIntAxis],10**pctAvg[:,4], color=cmapg(0.7), lw=2)
    for r in range(2,-1,-1):
        ax[0].fill_between([10**x for x in forecastIntAxis], 10**pctTech[:,1+r], 10**pctTech[:,-r-1], alpha=0.1+0.2*r, color=cmapp(0.7), zorder=-2-r, lw=0)
        ax[1].fill_between([10**x for x in forecastIntAxis], 10**pctAvg[:,1+r], 10**pctAvg[:,-r-1], alpha=0.1+0.2*r, color=cmapg(0.7), zorder=-2-r, lw=0)
    ax[2].plot([10**x for x in forecastIntAxis], np.asarray(countPoints)[:,0], color='k', lw=2)
    ax2.plot([10**x for x in forecastIntAxis], np.asarray(countTechs)[:,1], color='red', lw=2)
    ax2.set_ylabel('Number of technologies available', color='red')
    ax2b.set_ylabel('Number of technologies available', color='red')
    ax[0].plot([1,1],[0,100],'k')
    ax[1].plot([1,1],[0,100],'k')
    axb[0].plot([1,1],[0,100],'k')
    axb[1].plot([1,1],[0,100],'k')
    ax[0].annotate('Training', xy=(10**(-tOrd/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
    ax[0].annotate('Forecast', xy=(10**(+fOrd/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
    axb[0].annotate('Training', xy=(10**(-tOrd/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
    axb[0].annotate('Forecast', xy=(10**(+fOrd/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
    ax[0].set_ylim(0.1,10)
    ax[0].set_yscale('log', base=10)
    ax[0].set_xscale('log', base=10)
    ax[0].set_ylabel('Error (Actual/Predicted)')
    ax[0].set_title('Technologies available: ' + str(trainErr['Tech'].nunique())+
                    '\n Total points with '+ str(tOrd)+ ' orders of magnitude for training '+
                    ' and '+ str(fOrd)+' orders of magnitude for forecast: '+ str(dferrTech.shape[0]))
    axb[0].set_ylabel('Error (Actual/Predicted)')
    axb[0].set_title('Technologies available: ' + str(trainErr['Tech'].nunique())
                    #  +
                    # '\n Total points with '+ str(tOrd)+ ' orders of magnitude for training '+
                    # ' and '+ str(fOrd)+' orders of magnitude for forecast: '+ str(countTot)
                    )
    ax[0].plot([0,10**10],[1,1],'k', zorder=-10)
    ax[0].set_xlim(10**-tOrd, 10**fOrd)
    ax[1].annotate('Forecast', xy=(10**(+fOrd/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
    axb[1].annotate('Forecast', xy=(10**(+fOrd/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
    ax[1].set_ylim(0.1,10)
    ax[1].set_yscale('log', base=10)
    ax[1].set_xscale('log', base=10)
    ax[1].set_ylabel('Error (Actual/Predicted)')
    axb[1].set_ylabel('Error (Actual/Predicted)')
    ax[1].plot([0,10**10],[1,1],'k', zorder=-10)
    ax[1].set_xlim(10**-tOrd, 10**fOrd)

    ax[2].annotate('Training', xy=(10**(-tOrd/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
    ax[2].annotate('Forecast', xy=(10**(+fOrd/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
    axb[2].annotate('Training', xy=(10**(-tOrd/2), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
    axb[2].annotate('Forecast', xy=(10**(+fOrd/5), 6), xycoords='data', ha='center', va='bottom', fontsize=12)
    ax[2].set_xscale('log', base=10)
    ax[2].set_xlabel('Predicted cumulative production / Current cumulative production')
    ax[2].set_ylabel('Number of points to estimate error')
    axb[2].set_xlabel('Predicted cumulative production / Current cumulative production')
    axb[2].set_ylabel('Number of points to estimate error')
    ax[2].set_xlim(10**-tOrd,10**fOrd)
    countPoints = np.array(countPoints)
    ax[2].set_ylim(0,max(max(countPoints[:,0])*1.1, ax[2].get_ylim()[1]))

    legend_elements = [
                        matplotlib.lines.Line2D([0], [0], color='b', lw=2, label='Training error'),
                        matplotlib.lines.Line2D([0], [0], color=cmapp(0.7), lw=2, label='Forecast error - Technology-specific'),
                        matplotlib.lines.Line2D([0], [0], color=cmapg(0.7), lw=2, label='Forecast error - Average slope'),
                        ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3)

    axes1=fig.add_axes([0.875,0.415,0.12,0.2])
    axes1.plot([0,1],[0.5,0.5],'k', lw=2)
    axes1.fill_between([0,1],[0.25,0.25],[0.75,0.75], color='k', alpha=0.3)
    axes1.fill_between([0,1],[0.1,0.1],[0.9,0.9], color='k', alpha=0.3)
    axes1.fill_between([0,1],[0,0],[1.0,1.0], color='k', alpha=0.1)
    axes1.annotate('10th percentile', xy=(3.0, 0.1), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.annotate('25th percentile', xy=(3.0, 0.25), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.annotate('Median', xy=(3.0, 0.5), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.annotate('75th percentile', xy=(3.0, 0.75), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.annotate('90th percentile', xy=(3.0, 0.9), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.annotate('Max', xy=(3.0, 1.0), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.annotate('Min', xy=(3.0, 0.0), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.set_xlim(-1,5)
    axes1.set_ylim(-0.2,1.2)
    axes1.set_xticks([])
    axes1.set_yticks([])
    axes1.axis('off')

    fig.subplots_adjust(top=0.92, bottom=0.11, right=0.85)
    
    for x in forecastIntAxis[1:]:
        stats1 = {}
        labs = ['whislo', 'q1', 'med', 'q3', 'whishi']
        count_ = 1
        for l in labs:
            stats1[l] = 10**pctTech[forecastIntAxis.index(x),count_]
            count_ += 1
            if count_ == 2 or count_ == 5:
                count_ += 1
        axb[0].bxp([stats1], positions = [10**x], widths = (10**x)/8, showfliers=False, boxprops=dict(color=cmapp(0.7), lw=2), manage_ticks=False)
        stats2 = {}
        count_ = 1
        for l in labs:
            stats2[l] = 10**pctAvg[forecastIntAxis.index(x),count_]
            count_ += 1
            if count_ == 2:
                count_ += 1
        axb[1].bxp([stats2], positions = [10**x], widths = (10**x)/8, showfliers=False, boxprops=dict(color=cmapg(0.7), lw=2), manage_ticks=False)
    axb[1].set_yscale('log', base=10)
    axb[1].set_ylim(0.1,10)
    axb[0].set_ylim(0.1,10)
    axb[0].plot([0,10**10],[1,1],'k', zorder=-10)
    axb[1].plot([0,10**10],[1,1],'k', zorder=-10)
    axb[2].plot([1,1],[0,100**100],'k')
    axb[2].plot([10**x for x in forecastIntAxis],countPoints, color='k', lw=2)
    axb[2].set_ylim(0,max(np.asarray(countPoints)[:,0])*1.1)
    ax2b.plot([10**x for x in forecastIntAxis],countTechs, color='red', lw=2)
    axb[1].set_xlim(10**-tOrd, 10**fOrd)
    axb[2].set_ylim(0,max(max(countPoints[:,0])*1.1, ax[2].get_ylim()[1]))

    figb.legend(handles=legend_elements, loc='lower center', ncol=3)

    axes1=figb.add_axes([0.875,0.415,0.12,0.2])
    axes1.plot([0,1],[0.5,0.5],'k', lw=1)
    axes1.plot([0,1,1,0,0],[0.25,0.25,0.75,0.75,0.25], lw=2, color='k')
    # axes1.fill_between([0,1],[0.25,0.25],[0.75,0.75], color='k', alpha=0.3)
    # axes1.fill_between([0,1],[0.1,0.1],[0.9,0.9], color='k', alpha=0.3)
    axes1.plot([0,1],[0,0], color='k', lw=1)
    axes1.plot([0,1],[1,1], color='k', lw=1)
    axes1.plot([0.5,0.5],[0,0.25], color='k', lw=1)
    axes1.plot([0.5,0.5],[0.75,1], color='k', lw=1)
    axes1.annotate('25th percentile', xy=(3.0, 0.25), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.annotate('Median', xy=(3.0, 0.5), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.annotate('75th percentile', xy=(3.0, 0.75), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.annotate('Max', xy=(3.0, 1.0), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.annotate('Min', xy=(3.0, 0.0), xycoords='data', ha='center', va='center', fontsize=7)
    axes1.set_xlim(-1,5)
    axes1.set_ylim(-0.2,1.2)
    axes1.set_xticks([])
    axes1.set_yticks([])
    axes1.axis('off')

    figb.subplots_adjust(top=0.92, bottom=0.11, right=0.85)
    return fig, ax, figb, axb