import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn, scipy

df = pd.read_csv('ExpCurves.csv')
# df = pd.read_csv('NormalizedExpCurves.csv')

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
df['Sector'] = [sectorsinv[tech] for tech in df['Tech']]

method='regression'

# get slope for all technologies
slopes = []
# slopes_pl = []
for tech in df['Tech'].unique():
    sel = df.loc[df['Tech']==tech]
    x = np.log10(sel['Cumulative production'].values)
    y = np.log10(sel['Unit cost'].values)
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()
    slopes.append([tech, result.params[1]])
slopes = pd.DataFrame(slopes, columns=['Tech', 'Slope'])

dferr = []
dferr2 = []
counterr = 0

RMSEdiff = []
for tech in df['Tech'].unique():
    # errordiff = []
    rmse1 = []
    rmse2 = []
    # computing average technological slope based on all other technologies
    slopeall = np.mean(slopes.loc[slopes['Tech'] != tech,'Slope'].values)
    # computing technology specific slope
    sel = df.loc[df['Tech']==tech]
    x = np.log10(sel['Cumulative production'].values)
    y = np.log10(sel['Unit cost'].values)
    H = len(x)
    # calibrate model over first set of points
    # for i in range(H):
    #     for N in range(0 - 1*(i==0), -1, -1):
        # for N in range(i-1, -1, -1):
    for i in range(round(H/2),round(H/2)+1):
    # for i in range(round(0.9*H),H):
    # for i in range(H-2,H):
        for N in range(0, -1, -1):
            slope = (y[i] - y[N]) /\
                (x[i] - x[N])
            # add linear regression method
            if method=='regression':
                model = sm.OLS(y[N:i+1], sm.add_constant(x[N:i+1]))
                result = model.fit()
                slope = result.params[1]
            # compute error associated using slope M points after midpoint
            for M in range(i+1, H):
                pred =  y[i] + slope * (x[M] - x[i])
                # if method=='regression':
                # 	pred = result.params[0] + slope * x[M]
                pred2 =  y[i] + slopeall * (x[M] - x[i])
                error = (y[M] - (pred))
                error2 = (y[M] - (pred2))
                rmse1.append(error**2)
                rmse2.append(error2**2)
                # error point by point
                # RMSEdiff.append((error**2)**0.5-(error2**2)**0.5)
    # error by technology
    RMSEdiff.append(np.mean(rmse1)**0.5-np.mean(rmse2)**0.5)
RMSEdiff = pd.DataFrame(RMSEdiff, columns=['diff'])
N = RMSEdiff['diff'].nunique()

# # number of wins assuming binomial distribution
# print('Assuming wins have a binomial distribution, number of wins for each  method should be between ', \
#      round( RMSEdiff['diff'].nunique()/2-RMSEdiff['diff'].nunique()**0.5), ' and ', \
#         round( RMSEdiff['diff'].nunique()/2+RMSEdiff['diff'].nunique()**0.5))
# print('\t Number of wins: Technology specific (', sum(RMSEdiff['diff'].values < 0), \
#         '), Average slope (', sum(RMSEdiff['diff'].values > 0), ')')

# plt.hist(RMSEdiff['diff'].values, bins=1000)

print('Paired t-test: null hypothesis rejected if value is above +/- 1.990')
mu = np.mean(RMSEdiff['diff'].values)
std = np.std(RMSEdiff['diff'].values) / (RMSEdiff.shape[0])**0.5
print(mu, std)
print('\t The value is ', mu/std)
# print('The mean difference is ', mu)   
# print('\t The confidence interval for the mean is (', mu-1.990*std,', ', mu+1.990*std,')')


print('Wilcoxon signed rank test: null hypothesis rejected if value is below -1.96')
RMSEdiff['abs'] = np.abs(RMSEdiff['diff'].values)
RMSEdiff = RMSEdiff.sort_values(by='abs', ascending=False)
RMSEdiff = RMSEdiff.reset_index()
Rp, Rm = 0, 0
for i in range(RMSEdiff.shape[0]):
    if RMSEdiff['diff'].values[i] > 0:
        Rp += i+1
    elif RMSEdiff['diff'].values[i] == 0:
        Rp += 1/2*(i+1)
        Rm += 1/2*(i+1)
    else:
        Rm += i+1
T = min(Rp,Rm)
z = (T - 1/4*N*(N+1)) / (1/24*N*(N+1)*(2*N+1))**0.5
print('\tThe value is ', z)

for sector in sectors:
    print('\n\n',sector)
    RMSEdiff = []
    for tech in sectors[sector]:
        rmse1 = []
        rmse2 = []
        # computing average technological slope based on all other technologies
        slopeall = np.mean(slopes.loc[slopes['Tech'] != tech,'Slope'].values)
        # computing technology specific slope
        sel = df.loc[df['Tech']==tech]
        x = np.log10(sel['Cumulative production'].values)
        y = np.log10(sel['Unit cost'].values)
        H = len(x)
        # calibrate model over first set of points
        # for i in range(H):
        #     for N in range(0 - 1*(i==0), -1, -1):
            # for N in range(i-1, -1, -1):
        for i in range(round(H/2),round(H/2)+1):
        # for i in range(round(0.9*H),H):
        # for i in range(H-2,H):
            for N in range(0, -1, -1):
                slope = (y[i] - y[N]) /\
                    (x[i] - x[N])
                # add linear regression method
                if method=='regression':
                    model = sm.OLS(y[N:i+1], sm.add_constant(x[N:i+1]))
                    result = model.fit()
                    slope = result.params[1]
                # compute error associated using slope M points after midpoint
                for M in range(i+1, H):
                    pred =  y[i] + slope * (x[M] - x[i])
                    # if method=='regression':
                    # 	pred = result.params[0] + slope * x[M]
                    pred2 =  y[i] + slopeall * (x[M] - x[i])
                    error = (y[M] - (pred))
                    error2 = (y[M] - (pred2))
                    rmse1.append(error**2)
                    rmse2.append(error2**2)
                    # error point by point
                    # RMSEdiff.append((error**2)**0.5-(error2**2)**0.5)
        # error by technology
        RMSEdiff.append(np.mean(rmse1)**0.5-np.mean(rmse2)**0.5)
    RMSEdiff = pd.DataFrame(RMSEdiff, columns=['diff'])
    N = RMSEdiff['diff'].nunique()

    # # number of wins assuming binomial distribution
    # print('Assuming wins have a binomial distribution, number of wins for each  method should be between ', \
    #      round( RMSEdiff['diff'].nunique()/2-RMSEdiff['diff'].nunique()**0.5), ' and ', \
    #         round( RMSEdiff['diff'].nunique()/2+RMSEdiff['diff'].nunique()**0.5))
    # print('\t Number of wins: Technology specific (', sum(RMSEdiff['diff'].values < 0), \
    #         '), Average slope (', sum(RMSEdiff['diff'].values > 0), ')')

    # plt.hist(RMSEdiff['diff'].values, bins=1000)

    print('Paired t-test: null hypothesis rejected if value is above +/- 1.990')
    mu = np.mean(RMSEdiff['diff'].values)
    std = np.std(RMSEdiff['diff'].values) / (RMSEdiff.shape[0])**0.5
    print(mu, std)
    print('\t The value is ', mu/std)
    # print('The mean difference is ', mu)   
    # print('\t The confidence interval for the mean is (', mu-1.990*std,', ', mu+1.990*std,')')


    print('Wilcoxon signed rank test: null hypothesis rejected if value is below -1.96')
    RMSEdiff['abs'] = np.abs(RMSEdiff['diff'].values)
    RMSEdiff = RMSEdiff.sort_values(by='abs', ascending=False)
    RMSEdiff = RMSEdiff.reset_index()
    Rp, Rm = 0, 0
    for i in range(RMSEdiff.shape[0]):
        if RMSEdiff['diff'].values[i] > 0:
            Rp += i+1
        elif RMSEdiff['diff'].values[i] == 0:
            Rp += 1/2*(i+1)
            Rm += 1/2*(i+1)
        else:
            Rm += i+1
    T = min(Rp,Rm)
    z = (T - 1/4*N*(N+1)) / (1/24*N*(N+1)*(2*N+1))**0.5
    print('\tThe value is ', z)

plt.show()
