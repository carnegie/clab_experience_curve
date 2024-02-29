import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import piecewise_regression as pw
import statsmodels.api as sm
import os

# set figures' parameters
sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300
sns.color_palette('colorblind')

# Define a function to compute the log likelihood function
def compute_log_likelihood(rss, n):
    """
    Compute the log likelihood function

    Parameters
    ----------
    rss : float
        Residual sum of squares
    n : int
        Number of observations
    
    Returns
    -------
    llf : float
        Log likelihood function
    """
    s = np.sqrt(rss / n)
    llf = - n/2 * np.log(2 * np.pi) - \
            n * np.log(s) - 1/(2 * s**2) * rss
    return llf

# define a function to compute the Akaike Information Criterion
def computeAIC(rss, n, k):
    """
    Compute the Akaike Information Criterion

    Parameters
    ----------
    rss : float
        Residual sum of squares
    n : int
        Number of observations
    k : int
        Number of parameters
    
    Returns
    -------
    aic : float
        Akaike Information Criterion
    """
    aic = -2 * compute_log_likelihood(rss, n) + 2 * k
    return aic

# define a function to compute the Bayesian Information Criterion
def computeBIC(rss, n, k):
    """
    Compute the Bayesian Information Criterion
    
    Parameters
    ----------
    rss : float
        Residual sum of squares
    n : int
        Number of observations
    k : int
        Number of parameters

    Returns
    -------
    bic : float
        Bayesian Information Criterion    
    """
    bic = -2 * compute_log_likelihood(rss, n) + np.log(n) * k
    return bic


def main():

    # set to true to plot a figure for each technology
    plotFigTech = False

    # set to True if the regression dataset needs to be built
    BuildRegDataset = False

    # set the maximum number of breakpoints
    max_breakpoints = 6

    try:
        IC = pd.read_csv('IC.csv')
    except FileNotFoundError:
        IC = None
    
    if IC is None or BuildRegDataset is True:
        print('Performing piecewise linear regression',
                'on the experience curve dataset')

        # Load the data
        df = pd.read_csv('ExpCurves.csv')

        # Create a list to store the information criteria
        IC = []

        # Iterate over the technologies
        for t in df['Tech'].unique():
            
            # extract log10 of cumulative production and unit cost
            x = np.log10(df[df['Tech'] == t]\
                         ['Cumulative production'].values)
            y = np.log10(df[df['Tech'] == t]\
                         ['Unit cost'].values)

            if plotFigTech:
                plt.figure()
                plt.title(t)
                plt.plot(x, y, 'o')

            # iterate over the number of breakpoints
            for n_breaks in range(max_breakpoints + 1):

                # store breakpoints
                breaks = []
                slopes = []

                # handle case with no breakpoints 
                # (i.e., simple linear regression)
                if n_breaks == 0:

                    # fit a simple linear regression
                    res = sm.OLS(y, sm.add_constant(x)).fit()

                    # store the sum of squared residuals
                    rss = res.ssr
                    # store the slope
                    slopes.append( 100 * (1 - 2 ** res.params[1]) )

                    # fill in the remaining slots with NaN
                    for i in range(n_breaks + 1, max_breakpoints + 1):
                        slopes.append(np.nan)
                        breaks.append(np.nan)
 
                    if plotFigTech:
                        plt.plot(x, res.predict(sm.add_constant(x)))
                
                # handle case with one or more breakpoints
                else:

                    # fit a piecewise linear regression
                    res = pw.Fit(x, y, n_breakpoints=n_breaks)

                    # check if the optimization converged
                    if res.get_results()['converged'] is True:

                        # store the sum of squared residuals
                        rss = res.get_results()['rss']


                        # store breakpoints and slopes
                        for i in range(n_breaks):
                            breaks.append(res.get_results()\
                                          ['estimates']\
                                            ['breakpoint'+str(i+1)]\
                                                ['estimate'])
                            slopes.append(100 * ( 1 - \
                                            2 ** (res.get_results()\
                                                    ['estimates']\
                                                    ['alpha'+str(i+1)]\
                                                    ['estimate'])
                                            ))

                        # sort slopes
                        slopes = [x for _,x in sorted(zip(breaks, slopes))]
                        # add last computed slope
                        slopes.append(100 * (1 - \
                                        2 ** (res.get_results()\
                                            ['estimates']\
                                            ['alpha'+str(n_breaks+1)]\
                                            ['estimate'])
                                        ))
                        # sort breakpoints
                        breaks.sort()

                        # fill in the remaining slots with NaN
                        for i in range(n_breaks + 1, max_breakpoints + 1):
                            breaks.append(np.nan)
                            slopes.append(np.nan)
                        
                        if plotFigTech:
                            res.plot_fit()
                    
                    # if optimization has not converged, 
                    # skip to the next number of breakpoints
                    else:
                        continue

                # define parameters for information criteria computation
                    
                # number of observations
                n = x.shape[0] 
                # number of parameters
                # intercept and slope for 0 breaks
                # + additional slope for each break
                k = 2 + n_breaks * 2 

                # calculate the Akaike and Bayesian information criteria
                aic = computeAIC(rss, n, k)
                bic = computeBIC(rss, n, k)

                # store the information criteria values
                IC.append([t, n_breaks, aic, bic, x[0], *breaks, *slopes, x.shape[0]])
            
            if plotFigTech:
                plt.show()

        # convert the list of information criteria to a pandas DataFrame
        IC = pd.DataFrame(IC, columns=['Tech', 'n_breaks', 'AIC', 'BIC',
                                    'Initial production', 'Breakpoint 1',
                                    'Breakpoint 2', 'Breakpoint 3', 
                                    'Breakpoint 4','Breakpoint 5', 
                                    'Breakpoint 6',
                                    'LR 1', 'LR 2', 'LR 3', 'LR 4',
                                    'LR 5', 'LR 6', 'LR 7', 'Number of observations'])
        IC.to_csv('IC.csv', index=False)

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

    # plot the distribution of technologies over number of segments
    fig, ax = plt.subplots(1,1, figsize=(9,6), sharex=True, sharey=True)
    sns.barplot(data=metrics, 
            x = 'Number of segments',
            y = 'Count',
            hue='Metric',
            ax = ax,
            legend=False)
    ax.annotate('Akaike\nInformation\nCriterion', (-.25, 25),
                xycoords='data', color=sns.color_palette()[0],
                ha='center', va='center')
    ax.annotate('Bayesian\nInformation\nCriterion', (2.25, 25),
                xycoords='data', color=sns.color_palette()[1],
                ha='center', va='center')
    
    ax.set_xlim(-1,7)
    ax.set_xlabel('Optimal number of segments')
    ax.set_ylabel('Number of technologies')

    fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95)

    if not os.path.exists('figs' + os.path.sep + 'SupplementaryFigures'):
        os.makedirs('figs' + os.path.sep + 'SupplementaryFigures')
    fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                 os.path.sep +
                'TechCounytVsOptimalSegments.png')
    fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                 os.path.sep +
                'TechCounytVsOptimalSegments.eps')


    # get the number of technologies 
    AIC = IC.loc[IC.groupby('Tech')['AIC'].idxmin()].reset_index()
    BIC = IC.loc[IC.groupby('Tech')['BIC'].idxmin()].reset_index()

    [print(n, AIC.loc[AIC['n_breaks'] == n, 'Tech'].unique()) for n in range(max_breakpoints + 1)]
    
    
    # rename metrics
    AIC['metric'] = 'Akaike'
    BIC['metric'] = 'Bayesian'

    # create new dataframe with tech counts and metrics
    metrics = pd.concat([AIC, BIC]).reset_index(drop=True)

    metrics = metrics[['n_breaks','metric', 'Number of observations']]

    # rename columns and add number of segments
    metrics.columns = ['n_breaks', 'Metric', 'Number of observations']
    metrics['Number of segments'] = metrics['n_breaks'] + 1

    fig, ax = plt.subplots(1,1, figsize=(9,5), sharex=True, sharey=True)
    sns.scatterplot(data=metrics, 
            x = 'Number of observations',
            y = 'Number of segments',
            hue='Metric',
            ax = ax,
            alpha=0.5,
            legend=False)
    # ax.annotate('Akaike\nInformation\nCriterion', (-.25, 25),
    #             xycoords='data', color=sns.color_palette()[0],
    #             ha='center', va='center')
    # ax.annotate('Bayesian\nInformation\nCriterion', (2.25, 25),
    #             xycoords='data', color=sns.color_palette()[1],
    #             ha='center', va='center')
    
    # ax.set_xlim(-1,7)
    
    # ax.set_title('Distribution of technologies over number of segments minimizing error')
    ax.set_xlabel('Number of observations')
    ax.set_ylabel('Number of segments')
    ax.set_xscale('log', base=10)
    
    fig.subplots_adjust(bottom=0.15)


    # sel1 = IC.loc[(IC['n_breaks'] == 1) & \
    #               ((IC.index.isin(IC.groupby('Tech')['AIC'].idxmin().values)) | \
    #                 (IC.index.isin(IC.groupby('Tech')['BIC'].idxmin().values)))]
    # sel2 = IC.loc[(IC['n_breaks'] == 2) & \
    #               ((IC.index.isin(IC.groupby('Tech')['AIC'].idxmin().values)) | \
    #                 (IC.index.isin(IC.groupby('Tech')['BIC'].idxmin().values)))]


    # sel1['Breakpoint divided by initial production'] = \
    #     10 ** (sel1['Breakpoint 1'] - sel1['Initial production'])
    # sel2['2nd breakpoint divided by 1st breakpoint'] = \
    #     10 ** (sel2['Breakpoint 2'] - sel2['Breakpoint 1'])
    # sel2['1st breakpoint divided by initial production'] = \
    #     10 ** (sel2['Breakpoint 1'] - sel2['Initial production'])
    

    # fig, ax = plt.subplots(figsize=(6,6))
    # sns.boxplot(data=sel1, y='Breakpoint divided by initial production', ax=ax)
    # plt.gca().set_yscale('log', base=10)

    # plt.subplots_adjust(bottom=0.15, left=0.2, right=0.9)


    # fig, ax = plt.subplots(1,3, figsize=(15,6), width_ratios=[2,1,1])
    # sns.scatterplot(data=sel2, 
    #                 x='1st breakpoint divided by initial production', 
    #                 y='2nd breakpoint divided by 1st breakpoint', ax=ax[0])
    # ax[0].set_yscale('log', base=10)
    # ax[0].set_xscale('log', base=10)
    # ax[0].set_aspect('equal')
    # xlim = ax[0].get_xlim()
    # ylim = ax[0].get_ylim()
    # ax[0].plot([min(xlim[0], ylim[0]), max(xlim[1], ylim[1])], 
    #             [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])], 
    #             '--', lw=1)
    # ax[0].annotate('1:1', (0.45, 0.55), rotation=45,
    #                 xycoords='axes fraction', 
    #                 ha='center', va='center')

    # ax[0].set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    # ax[0].set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))

    # sns.boxplot(data=sel2, y='1st breakpoint divided by initial production', ax=ax[1])
    # ax[1].set_yscale('log', base=10)
    # ax[1].set_ylim(0.9, 1e8)
    # sns.boxplot(data=sel2, y='2nd breakpoint divided by 1st breakpoint', ax=ax[2])
    # ax[2].set_yscale('log', base=10)
    # ax[2].set_ylim(0.9, 1e8)
    
    # fig.subplots_adjust(wspace=0.3, bottom=0.15, left=0.1, right=0.95)


    fig, ax = plt.subplots(max_breakpoints, 2, figsize=(15,10), 
                           sharex=True, sharey='col')
    

    for n_break in range(1, max_breakpoints + 1):
        sel = IC.loc[(IC['n_breaks'] == n_break) & \
                ((IC.index.isin(IC.groupby('Tech')['AIC'].idxmin().values)) | \
                (IC.index.isin(IC.groupby('Tech')['BIC'].idxmin().values)))]

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
                                 sel[sel['Tech'] == t][['LR '+str(i+1) for i in range(n_break + 1)]].values[0],
                                 color=sns.color_palette()[0],
                                 lw=.5,)
            
        ax[n_break-1][0].axhline(0, color='k', ls='--', lw=.5, alpha=.5, zorder=-5)
        ax[n_break-1][1].axhline(0, color='k', ls='--', lw=.5, alpha=.5, zorder=-5)
        # ax[n_break-1][0].set_title(str(n_break+1) + ' segments')

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

if __name__ == "__main__":
    main()
