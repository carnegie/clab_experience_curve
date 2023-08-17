import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy, seaborn

### sectors dictionary
sectors = {'Energy':['Wind_Turbine_2_(Germany)', 'Fotovoltaica',
                    'Crude_Oil', 'Photovoltaics_2', 
                    'Onshore_Gas_Pipeline', 'Wind_Electricity',
                    'Photovoltaics_4','Geothermal_Electricity',
                    'Solar_Thermal','Nuclear_Electricity',
                    'Solar_Thermal_Electricity', 'Electric_Power',
                    'SCGT', 'Photovoltaics', 
                    'Solar_Water_Heaters', 'Wind_Turbine_(Denmark)',
                    'CCGT_Power','Nuclear_Power_(OECD)',
                    'CCGT_Electricity', 'Offshore_Gas_Pipeline',
                    'Wind_Power'], 
        'Chemicals':['Titanium_Sponge','CarbonDisulfide',
                    'Primary_Aluminum', 'Acrylonitrile',
                    'HydrofluoricAcid','PolyesterFiber',
                    'SodiumHydrosulfite', 'EthylAlcohol',
                    'Ethanol_2', 'Cyclohexane',
                    'Polyvinylchloride','PolyethyleneLD',
                    'Trichloroethane', 'Polypropylene',
                    'Pentaerythritol','Ethylene_2',
                    'VinylAcetate', 'CarbonBlack',
                    'Aniline', 'PhthalicAnhydride',
                    'Magnesium', 'MaleicAnhydride',
                    'TitaniumDioxide', 'Paraxylene',
                    'Ammonia', 'VinylChloride',
                    'Sorbitol', 'Styrene',
                    'Aluminum', 'Polystyrene',
                    'Phenol', 'BisphenolA',
                    'EthyleneGlycol', 'Methanol',
                    'PolyethyleneHD', 'Low_Density_Polyethylene',
                    'Urea', 'Sodium',
                    'Ethanolamine', 'SodiumChlorate',
                    'Primary_Magnesium', 'NeopreneRubber',
                    'Ethylene', 'AcrylicFiber',
                    'Formaldehyde', 'Benzene',
                    'Ethanol_(Brazil)', 'IsopropylAlcohol',
                    'Motor_Gasoline', 'Caprolactam'],
        'Hardware': ['Transistor', 'DRAM', 
                     'Hard_Disk_Drive', 'Laser_Diode'],
        'Consumer goods': ['Monochrome_Television', 'Automotive_(US)' ,
                           'Electric_Range', 'Free_Standing_Gas_Range'],
        'Food': ['Milk_(US)', 'Refined_Cane_Sugar',
                 'Wheat_(US)', 'Beer_(Japan)',
                 'Corn_(US)'],
        'Genomics':['Shotgun_Sanger_DNA_Sequencing',
                     'Capillary_DNA_Sequencing']
}
sectorsColor = {'Energy':'royalblue', 'Chemicals':'black',
               'Hardware':'red', 'Consumer goods':'forestgreen',
               'Food':'cyan', 'Genomics':'darkmagenta'}

cmap = seaborn.color_palette("colorblind")
sectorsColor = {'Energy': cmap[0], 'Chemicals': cmap[1],
               'Hardware': cmap[2], 'Consumer goods': cmap[3],
               'Food': cmap[4], 'Genomics': cmap[8]}

sectorsinv = {v:k for k, vlist in sectors.items() for v in vlist}

# read cumulative production and unit cost
# split them into calibration and validation datasets
# according to the options defined in the main script
def splitData(sel, fraction, points):
    
    # read cumulative production and unit cost
    x = np.log10(sel['Cumulative production'].values)
    y = np.log10(sel['Unit cost'].values)
    
    if points == True:
        # separate calibration and validation datasets 
        # based on number of points
        x_cal = x[:round(x.shape[0]*fraction)+1]
        x_val = x[round(x.shape[0]*fraction)+1:]
        y_cal = y[:round(y.shape[0]*fraction)+1]
        y_val = y[round(y.shape[0]*fraction)+1:]
    
    else:
        # separate calibration and validation datasets 
        # based on cumulative production range
        idxcal = np.where(np.array(x)<=x[0]+(x[-1]-x[0])*fraction)
        idxval = np.where(np.array(x)>x[0]+(x[-1]-x[0])*fraction)
        x_cal = x[idxcal]
        x_val = x[idxval]
        y_cal = y[idxcal]
        y_val = y[idxval]
    return x, y, x_cal, x_val, y_cal, y_val


# compute slope using ordinary least squares regression
def computeSlope(df):

    # extract x and y
    x = np.log10(df['Cumulative production'].values)
    y = np.log10(df['Unit cost'].values)

    # fit linear regression model
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()

    return result.params[1]


# compute mean slope of technologies in dataset
def computeMeanSlope(df):

    # create list to store results
    slopeall = []

    # compute slope of each technology
    for tech in df['Tech'].unique():
        s = df.loc[df['Tech'] == tech]
        slopeall.append(computeSlope(s))
    
    return np.mean(slopeall)

# For all technologies, compute regression model,
# predictions, and estimate errors using technology-specific data.
# Compute predictions and errors using 
# the average slope of all remaining technologies.
def computeRegPredError(df, fraction, points):

    # create lists to store results
    LR_cal, LR_val, slopesall = [], [], []
    uc, cpCal, cpVal, ucpred, \
        errpred, ucpred2, errpred2 = [], [], [], [], \
                                        [], [], []
    slopeErr1, slopeErr2 = [], []

    for tech in df['Tech'].unique():

        # read data for specific technology
        sel = df.loc[df['Tech'] == tech].copy()

        # split data into calibration and validation datasets
        x, y, x_cal, x_val, y_cal, y_val = splitData(sel, fraction, points)

        # compute mean slope of all other technologies
        slopeall = computeMeanSlope(df.loc[~(df['Tech'] == tech)].copy())
        slopesall.append(slopeall)
        
        # perform regression on all datasets to obtain slopes
        model_cal = sm.OLS(y_cal, sm.add_constant(x_cal))
        result_cal = model_cal.fit()
        model_val = sm.OLS(y_val, sm.add_constant(x_val))
        result_val = model_val.fit()
        
        # store slopes
        LR_cal.append(result_cal.params[1])
        LR_val.append(result_val.params[1])

        # append data to lists
        uc.append(10**y)
        cpCal.append(10**x_cal)
        cpVal.append(10**x_val)

        # compute predictions and errors using technology-specific data
        # intercept and slope
        intercept = result_cal.params[0]
        slope = result_cal.params[1]
        ucpred.append(10**
            (intercept + slope * np.concatenate([x_cal, x_val]))
                      ) 
        
        # (error is maintained in log space)
        errpred.append(np.log10(ucpred[-1][len(x_cal):]) - \
            np.log10(sel['Unit cost'].values[len(x_cal):]))

        # compute predictions and errors using average slope
        # prediction starts from last observation and uses average slope
        ucpred2.append(10**(
            y_cal[-1] +  slopeall * \
            (np.concatenate([np.array([x_cal[-1]]), x_val]) - x_cal[-1])
            )) 
        # (error is maintained in log space)
        errpred2.append(np.log10(ucpred2[-1][1:]) - \
            np.log10(sel['Unit cost'].values[len(x_cal):]))
        
        slopeErr1.append(result_val.params[1] - slope)
        slopeErr2.append(result_val.params[1] - slopeall)

    return LR_cal, LR_val, slopesall, \
            uc, cpCal, cpVal, \
            ucpred, errpred, ucpred2, errpred2, \
            slopeErr1, slopeErr2

# compute R2 values using Monte Carlo sampling with replacement
def computeR2MonteCarlo(LR_cal, LR_val, techsList, iter=1000):

    # initialize R2 list
    R2 = []
    lenTechs = len(techsList)
    
    # iterations
    for MN_N in range(iter):
    
        # initialize list of technologies sampled
        techs = []
        # sample technologies with replacement
        for MN_T in range(lenTechs):
            idx = np.random.randint(0, lenTechs)
            techs.append([techsList[idx], LR_cal[idx], LR_val[idx]])

        # create dataframe and compute R2 after linear regression
        techs = pd.DataFrame(techs, columns=['Tech', 'LR_cal', 'LR_val'])
        model = sm.OLS(techs['LR_val'].values, 
                       sm.add_constant(techs['LR_cal'].values))
        result = model.fit()

        # if the samples have zero error 
        # (i.e. small sample and all points are the same)
        # the R2 is set to 1 
        if result.centered_tss == 0:
            R2.append(1)
        # else store the R2 value
        else:
            R2.append(result.rsquared)

    return R2

# compute errors over specified ranges of 
# cumulative production for training and validation
# using technology-specific and average slopes
def computeErrors(df, trainingOrdMag, forecastOrdMag):

    # initialize lists to store results
    trainErr, dferrTech, dferrAvg = [], [], []

    # iterate over all technologies
    for tech in df['Tech'].unique():

        # computing average technological slope
        #  based on all other technologies
        slopeall = computeMeanSlope(df.loc[~(df['Tech'] == tech)].copy())

        # select technology
        sel = df.loc[df['Tech']==tech].copy()

        # retrieve cumulative production and unit cost
        x = np.log10(sel['Cumulative production'].values)
        y = np.log10(sel['Unit cost'].values)

        # get length of the dataset
        H = len(x)

        # iterate over all points in the dataset
        for i in range(1,H):

            # use N to explore the training range 
            # (N recedes from i-1 to 0)
            N = i - 1
            while N > 0 and x[i] - x[N] < trainingOrdMag:
                N -= 1
            
            # if training range is large enough
            if x[i] - x[N] >= trainingOrdMag:

                # use M to explore the forecast range 
                # (M ranges from i+1 to H-1)
                M = min(i + 1, H - 1)
                while M <= H - 2 and x[M] - x[i] < forecastOrdMag:
                    M += 1

                # if training and forecast ranges are large enough
                if x[M] - x[i] >= forecastOrdMag:
                   
                    # derive linear regression model
                    model = sm.OLS(y[N:i+1], sm.add_constant(x[N:i+1]))
                    result = model.fit()
                    intercept = result.params[0]
                    slope = result.params[1]
                    
                    # compute training error
                    terr = y[N:i+1] - (intercept + slope * x[N:i+1])
                    for idx in range(len(terr)):
                        trainErr.append(
                            [x[N+idx] - x[i], terr[idx], tech])
                        
                    # compute forecast error associated 
                    # using slope M points after midpoint

                    #### NEED TO DISCUSS THESE LINES 
                    
                    # pred =  y[i] + slope * (x[i:M+1] - x[i])
                    
                    pred =  result.predict(sm.add_constant(x[i:M+1]))

                    pred2 =  y[i] + slopeall * (x[i:M+1] - x[i])
                    
                    # compute errors and store data
                    error = (y[i:M+1] - (pred)) 
                    error2 = (y[i:M+1] - (pred2)) 
                    for idx in range(len(error)):
                        dferrTech.append(
                            [x[i+idx] - x[i], error[idx], tech])
                        dferrAvg.append(
                            [x[i+idx] - x[i], error2[idx], tech])

    return trainErr, dferrTech, dferrAvg

# built arrays of breaks and centered breaks for forecast errors plot
def builtBreakCenteredArrays(ordOfMag, npoints, training=False):
    
    # use number of points and order of magnitudes
    # to build break and centered arrays
    if training == True:
        breaks = np.linspace(-ordOfMag-ordOfMag/npoints, 0, npoints+2)
        centered = [breaks[1]]
        for idx in range(1,len(breaks)-1):
            centered.append(breaks[idx]+\
                    (breaks[idx+1]-breaks[idx])/2)
        centered.append(0)
    
    else:
        breaks = np.linspace(0-ordOfMag/npoints, ordOfMag, npoints+2)
        centered = [0]
        for idx in range(1,len(breaks)-1):
            centered.append(breaks[idx]+\
                    (breaks[idx+1]-breaks[idx])/2)
        centered.append(ordOfMag)

    return breaks, centered


# def compute Technology weighted percentiles
def computeTechWeightedPercentiles(df, percentiles=[0,5,25,50,75,95,100]):

    df = df.copy()
    # count data points per technology and assign inverse weight
    for tt in df['Tech'].unique():
        df.loc[df['Tech']==tt,'Weights'] =  \
              1/df.loc[df['Tech']==tt].count()[0]
    
    # sort data by error
    df = df.sort_values(by='Error', ascending=True)
    
    # assign percentiles by summing weights
    cumsum = df['Weights'].cumsum().round(4)
    pt = []

    # save percentile (i.e. the first value above cutoff)
    for q in percentiles:
        cutoff = df['Weights'].sum() * q/100
        pt.append(df['Error'][cumsum >= cutoff.round(4)].iloc[0])
    
    return pt

# compute percentiles of forecast errors at specified points
def computePercentilesArray(dferr, breaks,
                             percentiles=[0,5,25,50,75,95,100], 
                             training=False):
    
    # initialize lists to count points, 
    # technologies, and store percentiles
    countPoints = []
    countTechs = []
    pct = []
    
    # iterate over the length of training interval breaks
    for i in range(len(breaks)):

        # select relevant data 
        # using breaks arrays 
        # (different if training or forecast)
        if i == 0:
            if training == False:
                sel = dferr.loc[(dferr['Forecast horizon']==0)].copy()
            else:
                sel = dferr.loc[\
                    (dferr['Forecast horizon']<=breaks[i+1]) &\
                    (dferr['Forecast horizon']>=breaks[i])].copy()
        elif i == len(breaks)-1:
            if training == False:
                sel = dferr.loc[\
                    (dferr['Forecast horizon']>=breaks[i])].copy()
            else:
                sel = dferr.loc[\
                    (dferr['Forecast horizon']==breaks[i])].copy()
        else:
            sel = dferr.loc[\
                    (dferr['Forecast horizon']>breaks[i]) &\
                    (dferr['Forecast horizon']<=breaks[i+1])].copy()
        
        # if no data is selected, append NaNs
        if sel.shape[0] == 0:
            pct.append([breaks[i],np.nan,np.nan,np.nan,np.nan,np.nan])
            countPoints.append(0)
            countTechs.append(0)
            continue

        # else, append number of points and technologies
        countPoints.append(sel.shape[0])
        countTechs.append(sel['Tech'].nunique())
        
        # compute technology weighted percentiles for selected data
        pt = computeTechWeightedPercentiles(sel, percentiles)
        pct.append([breaks[i],*pt])  

    pct = np.array(pct)
    return pct, countPoints, countTechs

# compute boxplots statistics for forecast errors
def computeBoxplots(pct, centered, positions=None):
    
    # initialize list to store dicts and labels for dict
    statss = []
    labs = ['whislo', 'q1', 'med', 'q3', 'whishi']

    # iterate over centered breaks
    for x in centered:
        # initialize dict to store stats
        stats = {}
        # if positions is empty,
        # assume to have the list of needed percentiles
        if positions is None:
            positions = [x for x in range(len(labs))]
        for l in range(len(labs)):
            stats[labs[l]] = 10**pct[centered.index(x),positions[l]]
        statss.append(stats)

    return statss

# analyze training data
def dataToPercentilesArray(df, ordOfMag, 
                           samplingPoints, training=False):
    
    ### plot training error
    npoints = int(samplingPoints * ordOfMag)

    # create centered cumulative production array for plotting
    breaks, centered = \
        builtBreakCenteredArrays(ordOfMag, npoints, training=training)
    
    # compute percentiles and boxplots stats
    pct, countPoints, countTechs = \
        computePercentilesArray(df, breaks, training=training)
    stats = \
        computeBoxplots(pct, centered, positions=[1,3,4,5,7])
    
    return pct, breaks, centered, \
            countPoints, countTechs, stats

# def computeSlopeErrors(df, fraction, points)

def performTPairedTest(errpred, errpred2):
    # compute RMSE and difference in RMSE
    RMSEdiff = []
    for t,a in zip(errpred, errpred2):
        e1 = np.mean([x**2 for x in t])**0.5
        e2 = np.mean([x**2 for x in a])**0.5
        RMSEdiff.append(e1 - e2)
    RMSEdiff = pd.DataFrame(RMSEdiff, columns=['diff'])
    N = RMSEdiff['diff'].nunique()

    # report statistics and p-value
    print('Paired t-test: null hypothesis ' + \
          'rejected if value is outside [' + \
        str(scipy.stats.t.ppf(0.025, N-1).round(3)) + \
            ',' + str(scipy.stats.t.ppf(0.975, N-1).round(3))+']')
    mu = np.mean(RMSEdiff['diff'].values)
    std = np.std(RMSEdiff['diff'].values) / (RMSEdiff.shape[0])**0.5
    print('\t The value is ', mu/std)
    print('\t The p-value is ', scipy.stats.t.sf(np.abs(mu/std), N-1)*2)

    return mu/std, scipy.stats.t.ppf(0.025, N-1), scipy.stats.t.ppf(0.975, N-1)

def performWilcoxonSignedRankTest(errpred, errpred2):
    # compute RMSE and difference in RMSE
    RMSEdiff = []
    for t,a in zip(errpred, errpred2):
        e1 = np.mean([x**2 for x in t])**0.5
        e2 = np.mean([x**2 for x in a])**0.5
        RMSEdiff.append(e1 - e2)
    RMSEdiff = pd.DataFrame(RMSEdiff, columns=['diff'])
    N = RMSEdiff['diff'].nunique()

    # rank RMSE differences
    print('Wilcoxon signed rank test: null hypothesis rejected if value is outside [-1.96,1.96]')
    RMSEdiff['abs'] = np.abs(RMSEdiff['diff'].values)
    RMSEdiff = RMSEdiff.sort_values(by='abs', ascending=True)
    RMSEdiff = RMSEdiff.reset_index()
    Rp, Rm = 0, 0
    count = 0
    for i in range(RMSEdiff.shape[0]):
        if RMSEdiff['diff'].values[i] > 0:
            Rp += i+1
            count += 1
        elif RMSEdiff['diff'].values[i] == 0:
            Rp += 1/2*(i+1)
            Rm += 1/2*(i+1)
        else:
            Rm += i+1

        ## uncomment below to understand how it works
        ## these lines print:
        ## 1) the difference in RMSE
        ## 2) the sum of ranks for the positive difference
        ## 3) the sum of ranks for the negative difference
        ## 4) the count of times where average slope is better
        ## 5) the total count of technologies
        print(RMSEdiff['diff'].values[i],
            '\t',
            Rp,
            '\t',
            Rm,
            '\t',
            count, 
            '\t',
            i+1)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # sns.kdeplot(RMSEdiff['diff'].values, ax=ax)
    # ax.plot(RMSEdiff['diff'].values,[0 for x in RMSEdiff['diff'].values], 'o')
    # plt.show()

    # compute statistics and report it
    T = min(Rp,Rm)
    z = (T - 1/4*N*(N+1)) / ((1/24*N*(N+1)*(2*N+1))**0.5)
    print('\tThe value is ', z)
    print('\t The p-value is ', scipy.stats.norm.sf(np.abs(z))*2)
    return z