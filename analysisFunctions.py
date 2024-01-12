import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy, seaborn

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

sectorsinv = {v:k for k, vlist in sectors.items() for v in vlist}

# compute mean slope of technologies in dataset
def computeMeanSlope(df):

    # create list to store results
    slopeall = []

    # compute slope of each technology
    for tech in df['Tech'].unique():
        s = df.loc[df['Tech'] == tech]
        slopeall.append(computeSlope(s))
    
    return np.mean(slopeall)

def computeSlope(df):
    
    # extract technology data
    x, y = np.log10(\
        df['Cumulative production'].values), \
        np.log10(df['Unit cost'].values)

    # build linear regression model and fit it to data
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()

    # return slope
    return result.params[1] 

def computeAllErrors(df, removeTechExamined=False):

    # initialize lists to store results
    dfObsErr = []

    # iterate over all technologies
    for t in df['Tech'].unique():

        if removeTechExamined:
            # computing average technological slope
            # based on all other technologies
            slopeall = computeMeanSlope(df.loc[~(df['Tech'] == t)].copy())
        else:
            slopeall = computeMeanSlope(df)
        # print(slopeall)

        # extract technology data
        x, y = np.log10(\
            df.loc[\
                df['Tech']==t, 'Cumulative production'].values), \
            np.log10(df.loc[df['Tech']==t, 'Unit cost'].values)
        
        # iterate over all points 
        # where prediction is meaningful in the dataset
        # i.e., the first two points are needed to make a prediction
        # and the last two points to have something
        for i in range(1, len(x)-1):
            # iterate over all points before 
            # to obtain different observed calibration sets
            for M in range(i-1,-1,-1):

                # build linear regression model and fit it to data
                model = sm.OLS(y[M:i+1], sm.add_constant(x[M:i+1]))
                result = model.fit()

                # iterate over all following points
                for N in range(i+1, len(x)):

                    # compute prediction using technology-specific slope
                    predtech = y[i] + \
                        result.params[1] * (x[N]-x[i])
                    
                    # compute prediction using average slope
                    predavg = y[i] + \
                        slopeall * (x[N]-x[i])

                    # compute error using technology-specific slope
                    errtech = y[N] - predtech

                    # compute error using average slope
                    erravg = y[N] - predavg
                    
                    # store data
                    dfObsErr.append([x[i] - x[M],
                                    x[N] - x[i],
                                    y[N] - y[i],
                                    predtech - y[i],
                                    predavg - y[i],
                                    errtech,
                                    erravg,
                                    x[i] - x[0],
                                    x[-1] - x[i],
                                    t])
                    
    return dfObsErr
