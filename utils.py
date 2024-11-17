import numpy as np
import statsmodels.api as sm
import matplotlib, os
import matplotlib.pyplot as plt
import seaborn as sns
import piecewise_regression as pw
import pandas as pd
import scipy.stats

### sectors dictionary
sectors = {'Energy':['Wind_Turbine_2_(Germany)', 
                    'Fotovoltaica',
                    'Crude_Oil', 
                    'Photovoltaics_2', 
                    'Onshore_Gas_Pipeline', 
                    'Wind_Electricity',
                    'Photovoltaics_4',
                    'Geothermal_Electricity',
                    'Solar_Thermal',
                    'Nuclear_Electricity',
                    'Solar_Thermal_Electricity', 
                    'Electric_Power',
                    'SCGT', 
                    'Photovoltaics', 
                    'Solar_Water_Heaters', 
                    'Wind_Turbine_(Denmark)',
                    'CCGT_Power',
                    'Nuclear_Power_(OECD)',
                    'CCGT_Electricity', 
                    'Offshore_Gas_Pipeline',
                    'Wind_Power'], 
        'Chemicals':['Titanium_Sponge',
                    'CarbonDisulfide',
                    'Primary_Aluminum', 
                    'Acrylonitrile',
                    'HydrofluoricAcid',
                    'PolyesterFiber',
                    'SodiumHydrosulfite', 
                    'EthylAlcohol',
                    'Ethanol_2', 
                    'Cyclohexane',
                    'Polyvinylchloride',
                    'PolyethyleneLD',
                    'Trichloroethane', 
                    'Polypropylene',
                    'Pentaerythritol',
                    'Ethylene_2',
                    'VinylAcetate', 
                    'CarbonBlack',
                    'Aniline', 
                    'PhthalicAnhydride',
                    'Magnesium', 
                    'MaleicAnhydride',
                    'TitaniumDioxide', 
                    'Paraxylene',
                    'Ammonia', 
                    'VinylChloride',
                    'Sorbitol', 
                    'Styrene',
                    'Aluminum', 
                    'Polystyrene',
                    'Phenol', 
                    'BisphenolA',
                    'EthyleneGlycol', 
                    'Methanol',
                    'PolyethyleneHD', 
                    'Low_Density_Polyethylene',
                    'Urea', 
                    'Sodium',
                    'Ethanolamine', 
                    'SodiumChlorate',
                    'Primary_Magnesium', 
                    'NeopreneRubber',
                    'Ethylene', 
                    'AcrylicFiber',
                    'Formaldehyde', 
                    'Benzene',
                    'Ethanol_(Brazil)', 
                    'IsopropylAlcohol',
                    'Motor_Gasoline', 
                    'Caprolactam'],
        'Hardware': ['Transistor', 
                    'DRAM', 
                    'Hard_Disk_Drive', 
                    'Laser_Diode'],
        'Consumer goods': ['Monochrome_Television', 
                           'Automotive_(US)' ,
                           'Ford_Model-T',
                           'Electric_Range', 
                           'Free_Standing_Gas_Range'],
        'Food': ['Milk_(US)', 
                 'Refined_Cane_Sugar',
                 'Wheat_(US)', 
                 'Beer_(Japan)',
                 'Corn_(US)'],
        'Genomics':['Shotgun_Sanger_DNA_Sequencing',
                     'Capillary_DNA_Sequencing']
}

# define colors for sectors
sectors_colors = {'Chemicals':'#DE196B',
                    'Consumer goods':'#640FBF',
                    'Energy':'#FF9100',
                    'Food':'#048E2E',
                    'Genomics':'#632E0D',
                    'Hardware':'#1F92F0',
                    }

# invert dictionary (from tech to sector)
sectorsinv = {v:k for k, vlist in sectors.items() for v in vlist}


def computeSlope(df, cumprod_col=None, unitcost_col=None):
    
    """
    Computes slope of experience curve (learning exponent)
    using Wright's model

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing technology data
    
    cumprod_col : str
        Column name containing cumulative production data
    
    unitcost_col : str
        Column name containing unit cost data

    Returns
    -------
    slope : float
        Slope of experience curve (learning exponent)

    """

    if cumprod_col is None:
        cumprod_col = 'Cumulative production'
    if unitcost_col is None:
        unitcost_col = 'Unit cost'

    # extract technology data
    x, y = np.log10(\
        df[cumprod_col].values), \
        np.log10(df[unitcost_col].values)

    # build linear regression model and fit it to data
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()

    # return slope
    return result.params[1] 

def computeSlopeLafond(df, cumprod_col=None, unitcost_col=None):

    """
    Computes slope of experience curve (learning exponent)
    using first difference Wright's model

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing technology data

    cumprod_col : str
        Column name containing cumulative production data
    
    unitcost_col : str
        Column name containing unit cost data

    Returns
    -------
    slope : float
        Slope of experience curve (learning exponent)

    """

    if cumprod_col is None:
        cumprod_col = 'Cumulative production'
    if unitcost_col is None:
        unitcost_col = 'Unit cost'

    # extract technology data
    x, y = np.log10(\
        df[cumprod_col].values), \
        np.log10(df[unitcost_col].values)

    x_d, y_d = np.diff(x), np.diff(y)

    slope = sum(y_d * x_d) / sum(x_d**2)

    # return slope
    return slope

def computeNoiseVariance(df, slope, cumprod_col=None, unitcost_col=None):

    """
    Computes variance of the noise in the first different wright's law
    
    Parameters
    ----------    
    df : pandas.DataFrame
        Dataframe containing technology data
    
    slope : float
        Learning exponent
        
    cumprod_col : str
        Column name containing cumulative production data
        
    unitcost_col : str
        Column name containing unit cost data
    
    Returns
    -------
    noise_variance : float
        Variance of the noise in the data
    
    """

    if cumprod_col is None:
        cumprod_col = 'Cumulative production'
    if unitcost_col is None:
        unitcost_col = 'Unit cost'

    # extract technology data
    x, y = np.log10(\
        df[cumprod_col].values), \
        np.log10(df[unitcost_col].values)

    x_d, y_d = np.diff(x), np.diff(y)

    noise = y_d - slope * x_d

    noise_variance = 1 / (y_d.shape[0] - 1) * np.sum(noise**2)

    return noise_variance


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
    ll : float
        Log likelihood
    """

    s = np.sqrt(rss / n)
    ll = - n/2 * np.log(2 * np.pi) - \
            n * np.log(s) - 1/(2 * s**2) * rss
    return ll

def computeAIC(ll, k):

    """
    Compute the Akaike Information Criterion

    Parameters
    ----------
    ll : float
        Log likelihood 
    k : int
        Number of parameters
    
    Returns
    -------
    aic : float
        Akaike Information Criterion
    """

    aic = -2 * ll + 2 * k
    return aic

# define a function to compute the Bayesian Information Criterion
def computeBIC(ll, n, k):

    """
    Compute the Bayesian Information Criterion
    
    Parameters
    ----------
    ll : float
        Log likelihood
    n : int
        Number of observations
    k : int
        Number of parameters

    Returns
    -------
    bic : float
        Bayesian Information Criterion    
    """

    bic = -2 * ll + np.log(n) * k
    return bic


def plot_cost_prod_learning_dynamics(df,
                                     tech,
                                     min_points=20,
                                     lafond=True,
                                     time_range=None,
                                     fig=None,
                                     ax=None,
                                     cmap='viridis',
                                     savefig=True,
                                     cbar_kws=None
                                     ):
    
    """
    Plot cost-production learning dynamics for a given technology

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing technology data
    
    tech : str
        Technology name

    min_points : int
        Minimum number of points to plot data
    
    lafond : bool
        If True, compute learning exponent using first difference
        Wright's model
    
    time_range : list
        Range of years to plot data
    
    fig : matplotlib.figure.Figure
        Figure object

    ax : matplotlib.axes.Axes
        Axes object
    
    savefig : bool
        If True, save figure as .png and .pdf file
    
    cbar_kws : dict
        Dictionary containing colorbar parameters
            
    """        
    
    if fig is None:
        # create figure
        fig, ax = plt.subplots(1,2, figsize=(12,6))

    # set column names 
    cols = ['Unit cost', 'Year', 'Production',
            'Cumulative production']

    # check if there are enough points
    if df.shape[0] < min_points:
        plt.close(fig)
        print('Not enough points for ' + tech)
        return

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
    if time_range is None:
        norm = matplotlib.colors.Normalize(
                vmin=df[df.columns[1]].unique()[0],
                vmax=df[df.columns[1]].unique()[-1])   
    else:
        norm = matplotlib.colors.Normalize(
                vmin=time_range[0],
                vmax=time_range[1])
    
    # create figure
    sns.scatterplot(data=df, x=df.columns[3], y=df.columns[0],
                    hue=df.columns[1], ax=ax[0], palette=cmap,
                    hue_norm=norm,
                    legend=False, edgecolor='k', s=100)
    
    # set log-log scale and label axes
    ax[0].set_xscale('log', base=10)
    ax[0].set_yscale('log', base=10)
    ax[0].set_xlabel(df.columns[3])
    ax[0].set_ylabel(df.columns[0])

    # # add star representing technology-specific learning exponent
    ax[1].set_xlabel('Past learning rate [%]')
    ax[1].set_ylabel('Future learning rate [%]')

    # iterate over all years from 2nd to second to last
    for i in range(df[df.columns[1]].unique()[0],
                   df[df.columns[1]].unique()[-1]+1):
        
        # split data into calibration and validation sets
        cal = df[df[df.columns[1]]<=i]
        val = df[df[df.columns[1]]>=i]

        if len(cal) < 5 or len(val) < 5:
            continue

        if not(lafond):
            # compute learning exponents
            lexp_past = computeSlope(cal,
                                        cumprod_col=cal.columns[3],
                                        unitcost_col=cal.columns[0])
            lexp_future = computeSlope(val,
                                        cumprod_col=val.columns[3],
                                        unitcost_col=val.columns[0])
            
            # add star for central point
            if len(cal) == len(val) or len(cal) == len(val) + 1:
                sns.scatterplot(x=[100*(1 - 2**lexp_past)], 
                                y=[100*(1 - 2**lexp_future)],
                                color=cmap(norm(i)), edgecolor='k', s=100,
                                markers='*',
                                ax=ax[1], zorder=10, legend=False)
            else:
                # add points to learning exponent dynamics
                sns.scatterplot(x=[100*(1 - 2**lexp_past)], 
                                y=[100*(1 - 2**lexp_future)],
                                color=cmap(norm(i)), edgecolor='k', s=100,
                                ax=ax[1], zorder=1, legend=False)

        else:
            lexp_past = computeSlopeLafond(cal,
                                              cumprod_col=cal.columns[3],
                                              unitcost_col=cal.columns[0])
            lexp_future = computeSlopeLafond(val,
                                                cumprod_col=val.columns[3],
                                                unitcost_col=val.columns[0])

            # add star for central point
            if len(cal) == len(val) or len(cal) == len(val) + 1:
                sns.scatterplot(x=[100*(1 - 2**lexp_past)], 
                                y=[100*(1 - 2**lexp_future)],
                                # color=cmap(norm(i)), ,
                                color='gold',
                                edgecolor='k', s=200,
                                marker='*',
                                ax=ax[1], zorder=10, legend=False)
            else:
                # add points to learning exponent dynamics
                sns.scatterplot(x=[100*(1 - 2**lexp_past)], 
                                y=[100*(1 - 2**lexp_future)],
                                color=cmap(norm(i)), edgecolor='k', s=100,
                                alpha=0.75,
                                ax=ax[1], zorder=1, legend=False)
                
            try:
                axlim_min = min(axlim_min,
                                 100 * (1 - 2**max(lexp_past, lexp_future)))
                axlim_max = max(axlim_max, 
                                100 * (1 - 2**min(lexp_past, lexp_future)))
            except:
                axlim_min = 100 * (1 - 2**max(lexp_past, lexp_future))
                axlim_max = 100 * (1 - 2**min(lexp_past, lexp_future))
    

    # add colorbar
    if time_range is None:
        smap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        smap.set_array([])
        fig.subplots_adjust(right=0.9, top=0.9, bottom=0.35, left=0.15,
                            hspace=0.25)

        if cbar_kws is None:
            cbar_loc = [0.1, 0.15, 0.8, 0.02]
            orientation = 'horizontal'
        else:
            cbar_loc = cbar_kws['loc']
            orientation = cbar_kws['orientation']

        cbar_ax = fig.add_axes(cbar_loc)
        cbar = fig.colorbar(smap, cax=cbar_ax, label='Year', 
                            orientation=orientation)

        cbar.set_ticks([df[df.columns[1]].unique()[0], 
                        df[df.columns[1]].unique()[-1]])
        cbar.set_ticklabels([str(df[df.columns[1]].unique()[0]),
                            str(df[df.columns[1]].unique()[-1])])

    # add identity line to learning exponent dynamics
    lims = [ax[1].get_xlim(), ax[1].get_ylim()]
    lims = [min(lims[0][0],lims[1][0]), max(lims[0][1],lims[1][1])]
    ax[1].plot([-200,100],[-200,100], color='k', ls='--', lw=1, zorder=-10)
    ax[1].set_xlim(0.9*axlim_min, 1.1*axlim_max)
    ax[1].set_ylim(0.9*axlim_min, 1.1*axlim_max)
    ax[1].set_yticks(ax[1].get_xticks())
    ax[1].set_xticks(ax[1].get_yticks())
    ax[1].set_aspect('equal')
    
    if savefig:

        ## annotate panels
        ax[0].annotate('a', xy=(0.15, 0.05),
                        xycoords='axes fraction',
                        ha='center', va='center')
        ax[1].annotate('b', xy=(0.15, 0.05),
                        xycoords='axes fraction',
                        ha='center', va='center')

        fig.suptitle(tech)
    
        if not os.path.exists('figs' + 
                            os.path.sep + 
                            'supplementaryFigures'+
                            os.path.sep +
                            'learningRateDynamics'):
            os.makedirs('figs' +
                        os.path.sep +
                        'supplementaryFigures'+
                        os.path.sep +
                        'learningRateDynamics')
        plt.savefig('figs' +
                    os.path.sep +
                    'supplementaryFigures'+
                    os.path.sep +
                    'learningRateDynamics'+
                    os.path.sep +
                    tech + '.png')
        plt.savefig('figs' +
                    os.path.sep +
                    'supplementaryFigures'+
                    os.path.sep +
                    'learningRateDynamics'+
                    os.path.sep +
                    tech + '.pdf')

        plt.close(fig)    


def build_piecewise_regression_dataset(df, 
                                       max_breakpoints=6,
                                       min_dist=np.log10(2),
                                       plot_fig_tech=False,
                                       half_data=False,
                                       half_techs=False,
                                       remove_sector=None,
                                       output_file=None
                                       ):
    
    """
    Build dataset for piecewise regression analysis

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing technology data

    max_breakpoints : int
        Maximum number of breakpoints

    min_dist : float
        Minimum distance between breakpoints

    plot_fig_tech : bool
        If True, plot data for each technology

    half_data : bool
        If True, use only half of the data for each time series
    
    half_techs : bool
        If True, use only half of the technologies in the dataset (randomly selected)

    remove_sector : str
        Name of the sector to remove from the dataset

    output_file : str
        Name of the output file to append to "IC" (Information Criteria)

    Returns
    -------
    IC : list
        List containing information criteria values

    """
    
    # Create a list to store the information criteria
    IC = []

    # if half_techs is True, randomly select half of the technologies
    if half_techs:
        techs = np.random.choice(df['Tech'].unique(),
                                 round(df['Tech'].unique().shape[0]/2),
                                 replace=False)
        df = df[df['Tech'].isin(techs)]

    # Iterate over the technologies
    for t in df['Tech'].unique():

        if remove_sector is not None:
            if sectorsinv[t] == remove_sector:
                continue
        
        # extract log10 of cumulative production and unit cost
        x = np.log10(df[df['Tech'] == t]\
                        ['Cumulative production'].values)
        y = np.log10(df[df['Tech'] == t]\
                        ['Unit cost'].values)
        
        # if half_data is True, use only half of the data
        if half_data:
            x = x[:round(x.shape[0]/2)]
            y = y[:round(y.shape[0]/2)]

        if plot_fig_tech:
            plt.figure()
            plt.title(t)
            plt.plot(x, y, 'o')

        # iterate over the number of breakpoints
        for n_breaks in range(max_breakpoints + 1):

            # create empty lists to store breakpoints and learning exponents
            breaks = []
            slopes = []

            # handle case with no breakpoints 
            # (i.e., simple linear regression)
            if n_breaks == 0:


                # fit linear regression after differencing
                res = sm.OLS(np.diff(y), np.diff(x)).fit()

                # store the sum of squared residuals
                rss = res.ssr

                # store slope, constant is nan
                slopes.append(100 * (1 - 2 ** res.params[0]) )
                const = np.nan


                # fill in the remaining slots with NaN
                for i in range(n_breaks + 1, max_breakpoints + 1):
                    slopes.append(np.nan)
                    breaks.append(np.nan)

                if plot_fig_tech:
                    plt.plot(x, res.predict(sm.add_constant(x)))

                # define parameters for information criteria computation
                n = x.shape[0]
                k = 1

                # compute information criteria
                aic = computeAIC(compute_log_likelihood(rss, n-1), k)
                bic = computeBIC(compute_log_likelihood(rss, n-1), n-1, k)

                # append this data to output list
                IC.append([t, n_breaks, 1, aic, bic,
                            np.nan,
                            x[0], *breaks, x[-1], *slopes, x.shape[0]])
 
                # fit linear regression
                res = sm.OLS(y, sm.add_constant(x)).fit()

                # store the sum of squared residuals
                rss = res.ssr

                # store the slope
                slopes, breaks = [], []
                slopes.append( 100 * (1 - 2 ** res.params[1]) )
                const = res.params[0]
            
                # fill in the remaining slots with NaN
                for i in range(n_breaks + 1, max_breakpoints + 1):
                    slopes.append(np.nan)
                    breaks.append(np.nan)

            
            # handle case with one or more breakpoints
            else:

                # fit a piecewise linear regression
                res = pw.Fit(x, y, n_breakpoints=n_breaks,
                                min_distance_between_breakpoints=\
                                    min(0.99, min_dist/(x[-1] - x[0])),
                                min_distance_to_edge=\
                                    min(0.99, min_dist/(x[-1] - x[0]))
                                )

                # check if the optimization converged
                if res.get_results()['converged'] is True:

                    # store the sum of squared residuals
                    rss = res.get_results()['rss']

                    const = res.get_results()['estimates']['const']['estimate']

                    slopes = []

                    # store breakpoints and slopes
                    for i in range(n_breaks):
                        breaks.append(res.get_results()\
                                        ['estimates']\
                                        ['breakpoint'+str(i+1)]\
                                            ['estimate'])
                        slopes.append(res.get_results()\
                                                ['estimates']\
                                                ['beta'+str(i+1)]\
                                                ['estimate'])

                    # sort slopes
                    slopes = [x for _,x in sorted(zip(breaks, slopes))]
                    slopes.insert(0, res.get_results()\
                                            ['estimates']\
                                            ['alpha1']['estimate'])
                    slopes = list(np.cumsum(slopes))
                    slopes = list(100 * (1 - 2 ** np.array(slopes)))

                    # sort breakpoints
                    breaks.sort()

                    # fill in the remaining slots with NaN
                    for i in range(n_breaks + 1, max_breakpoints + 1):
                        breaks.append(np.nan)
                        slopes.append(np.nan)
                    
                    if plot_fig_tech:
                        res.plot_fit()
                        print(breaks)
                        print(slopes)

                        costs = []
                        bp=0
                        for xi in np.arange(x[0], x[-1], 0.01):
                            if bp >= len(breaks):
                                costs.append(costbp + \
                                                np.log2(1-slopes[-1]/100) * \
                                                    (xi-breaks[bp-1]))
                                continue
                            if np.isnan(breaks[bp]) or xi < breaks[bp]:
                                if bp == 0:
                                    costs.append(const + \
                                                 np.log2(1-slopes[0]/100)*xi)
                                else:
                                    costs.append(costbp + \
                                            np.log2(1 - slopes[bp]/100) * \
                                                    (xi - breaks[bp-1]))
                            else:
                                if bp == 0:
                                    costbp = const + \
                                        np.log2(1 - slopes[0]/100) * breaks[0]
                                else:
                                    costbp = costbp + \
                                        np.log2(1-slopes[bp]/100) * \
                                                (breaks[bp] - breaks[bp-1])
                                bp += 1
                                costs.append(costbp + \
                                             np.log2(1 - slopes[bp]/100) * \
                                                    (xi - breaks[bp-1]))
                        plt.plot([x for x in np.arange(x[0], x[-1], 0.01)], 
                                    costs, 'k.')                            
                
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
            aic = computeAIC(compute_log_likelihood(rss, n), k)
            bic = computeBIC(compute_log_likelihood(rss, n), n , k)
            
            # store the information criteria values
            IC.append([t, n_breaks, 0, aic, bic,
                        const,
                        x[0], *breaks, x[-1], *slopes, x.shape[0]])
        
        if plot_fig_tech:
            plt.show()

    # convert the list of information criteria to a pandas DataFrame
    IC = pd.DataFrame(IC, columns=['Tech', 'n_breaks', 'First Diff.', 
                                   'AIC', 'BIC',
                                    'Intercept',
                                    'Initial production', 'Breakpoint 1',
                                    'Breakpoint 2', 'Breakpoint 3', 
                                    'Breakpoint 4','Breakpoint 5', 
                                    'Breakpoint 6',
                                    'Final production',
                                    'LR 1', 'LR 2', 'LR 3', 'LR 4',
                                    'LR 5', 'LR 6', 'LR 7', 
                                    'Number of observations'])

    IC.to_csv('IC' + output_file + '.csv', index=False)
    
    return IC

def fit_probability_dist(data, floc=None):
    """
    Fit multiple probability distributions to data
    and return paremters and Bayesian Information Criterion

    Parameters
    ----------
    data : numpy.array
        Data to fit

    Returns
    -------
    summary : Pandas Dataframe 
        Dataframe containing log likelihood, BIC, number of samples,
         and parameters for each fitted distribution

    """

    # list of distributions
    dists = [scipy.stats.norm, scipy.stats.expon, 
             scipy.stats.lognorm, scipy.stats.uniform]
    
    # empty list to store results
    l = []

    # for the distributions considered
    for d in dists:

        # handle case with fixed location parameter
        if d.name in ['lognorm','expon'] and floc is not None:
            data_ = data[data > floc]
            params = d.fit(data_, floc=floc)
            ll = np.sum(d.logpdf(data_, *params))
            bic = computeBIC(
                ll, data_.shape[0], len(params))
            ssize = data_.shape[0]
        
        # all other cases
        else:
            params = d.fit(data)
            ll = np.sum(d.logpdf(data, *params))
            bic = computeBIC(
                ll, data.shape[0], len(params))
            ssize = data.shape[0]
        
        # append to list
        l.append([d.name, ll, bic, ssize, *params])

    # create summary dataframe
    summary = pd.DataFrame(l, columns=['Distribution', 'Log likelihood', 'BIC', 
                                 'Sample size', 
                                 'Param1', 'Param2', 'Param3'])
    # sort by BIC
    summary = summary.sort_values('BIC')

    return summary


def forecast(initial_cost, initial_prod,
             future_prod, model_params, 
             last_lexp=None,
             last_break=None,
             last_cost=None,
             noise_std=0,
             model='pwlinear'):
    
    """
    
    Forecast future cost given initial cost,
    initial production, future production, and model parameters
    using the piecewise linear experience curve model or
    the first difference wright's model

    Parameters
    ----------
    initial_cost : float
        Initial cost
    
    initial_prod : float
        Initial production

    future_prod : np.array
        Production values for which to produce a unit cost forecast
    
    model_params : pandas DataFrame
        List containing model parameters
        This DataFrame should have the columns:
        - Variable, Distribution, Loc, Scale
        One variable is 'breaks', the other one is 'lexp'.
        The distribution for 'breaks' is 'expon'.
        The distribution for 'lexp' is 'norm'.

    last_lexp : float
        Last learning exponent (if available, otherwise sampled later)

    last_break : float
        Last breakpoint (if available, otherwise the starting point is used)

    last_break_cost : float
        cost at last break (if available, otherwise the initial cost is used)
        Only used for the piecewise linear model

    model : str
        Model to use for forecasting
        Options are 'pwlinear' or 'lafond'
    
    Returns
    -------

    forecast : np.array
        Forecasted unit costs at future production levels

        
    """

    # extract parameters
    breaks = model_params[model_params['Variable'] == 'breaks']
    lexp = model_params[model_params['Variable'] == 'lexp']

    # build distributions
    breaks_dist = scipy.stats.expon(loc=breaks['Loc'].values[0],
                                    scale=breaks['Scale'].values[0])
    lexp_dist = scipy.stats.norm(loc=lexp['Loc'].values[0],
                                scale=lexp['Scale'].values[0])
    
    # handle cases of no previous information on the data series

    if last_lexp is None:
        last_lexp = lexp_dist.rvs()
    
    if last_break is None:
        last_break = initial_prod
    
    if last_cost is None and model == 'pwlinear':
        last_cost = initial_cost
    
    if model=='lafond':
        noise = 0
        last_forecast = initial_cost
        last_prod = initial_prod

    # sample next breakpoint
    next_break = last_break + max(np.log10(2), breaks_dist.rvs())

    # start the loop
    forecast = np.empty(0)
    for i in future_prod.shape[[0]]:

        if model == 'pwlinear':
            # check if we need to update the learning exponent
            if future_prod[i] >= next_break:
                last_lexp = lexp_dist.rvs()
                last_break = next_break
                next_break = last_break + max(np.log10(2), breaks_dist.rvs())
            
            # forecast the cost
            forecast = np.append(forecast, 
                                 last_cost + last_lexp * \
                                        (future_prod[i] - last_break) + 
                                        noise_std * np.random.randn()
                                    )
        
        elif model == 'lafond':
            noise = 0.19 * noise + noise_std * np.random.randn()
            forecast = np.append(forecast,
                                 last_forecast + last_lexp * \
                                    (future_prod[i] - last_prod) + noise)
            last_forecast = forecast[-1]
            last_prod = future_prod[i]

    return forecast 
                                 




    