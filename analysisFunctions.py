import numpy as np
import pandas as pd
import statsmodels.api as sm

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
        'Genomics':['Shotgun_Sanger_DNA_Sequencing', 'Capillary_DNA_Sequencing']
}
sectorsColor = {'Energy':'royalblue', 'Chemicals':'black',
               'Hardware':'red', 'Consumer goods':'forestgreen',
               'Food':'cyan', 'Genomics':'darkmagenta'}
sectorsinv = {v:k for k, vlist in sectors.items() for v in vlist}

# read cumulative production and unit cost
# split them into calibration and validation datasets
# according to the options defined in the main script
def splitData(sel, fraction, points):
    # read cumulative production and unit cost
    x = np.log10(sel['Cumulative production'].values)
    y = np.log10(sel['Unit cost'].values)
    if points == True:
        # separate calibration and validation datasets based on number of points
        x_cal = x[:round(x.shape[0]*fraction)]
        x_val = x[round(x.shape[0]*fraction):]
        y_cal = y[:round(y.shape[0]*fraction)]
        y_val = y[round(y.shape[0]*fraction):]
    else:
        # separate calibration and validation datasets based on cumulative production range
        idxcal = np.where(np.array(x)<=x[0]+(x[-1]-x[0])*fraction)
        idxval = np.where(np.array(x)>x[0]+(x[-1]-x[0])*fraction)
        x_cal = x[idxcal]
        x_val = x[idxval]
        y_cal = y[idxcal]
        y_val = y[idxval]
    return x, y, x_cal, x_val, y_cal, y_val


# compute slope using ordinary least squares regression
def computeSlope(df):
    x = np.log10(df['Cumulative production'].values)
    y = np.log10(df['Unit cost'].values)
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()
    return result.params[1]


# compute mean slope of technologies in dataset
def computeMeanSlope(df):
    slopeall = []
    for tech in df['Tech'].unique():
        s = df.loc[df['Tech'] == tech]
        slopeall.append(computeSlope(s))
    slopeall = np.mean(slopeall)
    return slopeall

# For all technologies, compute regression model,
# predictions, and estimate errors using technology-specific data.
# Compute predictions and errors using 
# the average slope of all remaining technologies.
def computeRegPredError(df, fraction, points):

    # create lists to store results
    LR_cal, LR_val, slopesall = [], [], []
    uc, cpCal, cpVal, ucpred, errpred, ucpred2, errpred2 = [], [], [], [], [], [], []

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

    return LR_cal, LR_val, slopesall, \
            uc, cpCal, cpVal, \
            ucpred, errpred, ucpred2, errpred2

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
        model = sm.OLS(techs['LR_val'].values, sm.add_constant(techs['LR_cal'].values))
        result = model.fit()
        # if the samples have zero error (i.e. small sample and all points are the same)
        # the R2 is set to 1 
        # else store the R2 value
        if result.centered_tss == 0:
            R2.append(1)
        else:
            R2.append(result.rsquared)
    return R2

# compute errors over specified ranges of 
# cumulative production for training and validation
# using technology-specific and average slopes
def computeErrors(df, trainingOrdMag, forecastOrdMag):
    trainErr, dferrTech, dferrAvg = [], [], []
    for tech in df['Tech'].unique():
        # computing average technological slope based on all other technologies
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
            # use N to explore the training range (N recedes from i-1 to 0)
            N = i - 1
            while N > 0 and x[i] - x[N] < trainingOrdMag:
                N -= 1
            if x[i] - x[N] >= trainingOrdMag:
                # use M to explore the forecast range (M ranges from i+1 to H-1)
                M = min(i + 1, H - 1)
                while M < H - 2 and x[M] - x[i] < forecastOrdMag:
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
                        trainErr.append([x[N+idx] - x[i], terr[idx], tech])
                    # compute forecast error associated using slope M points after midpoint

                    #### NEED TO DISCUSS THESE LINES WITH 
                    
                    pred =  y[i] + slope * (x[i:M+1] - x[i])
                    
                    # pred =  result.predict(sm.add_constant(x[i:M+1]))

                    pred2 =  y[i] + slopeall * (x[i:M+1] - x[i])
                    error = (y[i:M+1] - (pred)) 
                    error2 = (y[i:M+1] - (pred2)) 
                    for idx in range(len(error)):
                        dferrTech.append([x[i+idx] - x[i], error[idx], tech])
                        dferrAvg.append([x[i+idx] - x[i], error2[idx], tech])

    return trainErr, dferrTech, dferrAvg
 