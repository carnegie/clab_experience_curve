import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy, analysisFunctions, plottingFunctions

df = pd.read_csv('ExpCurves.csv')

### SELECT SCRIPT OPTIONS
# fraction of dataset used for calibration
fraction = 1/2
# split the dataset based on points (True) 
# or cumulative production interval (False)
points = True
# include nuclear technologies (True) or not (False)
nuclearIncluded = True

if nuclearIncluded == False:
    df = df.loc[~(df['Tech'].str.contains('Nuclear'))]

df['Sector'] = [analysisFunctions.sectorsinv[tech] for tech in df['Tech']]
sectorTech = [analysisFunctions\
              .sectorsinv[tech] for tech in df['Tech'].unique()]

# # compute regression model and predicition errors for each technology
# LR_cal, LR_val, slopesall, \
#     uc, cpCal, cpVal, \
#     ucpred, errpred, ucpred2, errpred2, \
#     slopeErrTech, slopeErrAvg = \
#         analysisFunctions.computeRegPredError(df, fraction, points)

# analysisFunctions.performTPairedTest(errpred, errpred2)

# analysisFunctions.performWilcoxonSignedRankTest(errpred, errpred2)

# analysisFunctions.performTPairedTest(slopeErrTech, slopeErrAvg)

# analysisFunctions.performWilcoxonSignedRankTest(slopeErrTech, slopeErrAvg)

# t, t1, t2, z, z1, z2 = analysisFunctions.performMonteCarloTests(errpred, errpred2)

# plottingFunctions.plotBoxplotPvalues(t, t1, t2, z, z1, z2)
# plt.show()


# repeat the analysis with changing forecast and training horizon
trOrds = [0.5, 1, 2]
forOrds = [0.5, 1, 2]

for tOrd in trOrds:
    for fOrd in forOrds:

        # compute points errors for each training and forecast range 
        trainErr, dferrTech, dferrAvg, \
            slopeErrTech, slopeErrAvg, _, _ = \
            analysisFunctions.computeErrors(df, tOrd, fOrd)

        columns = ['Forecast horizon', 'Error', 'Tech']
        dferrTech = pd.DataFrame(dferrTech, columns = columns)
        dferrAvg = pd.DataFrame(dferrAvg, columns=columns)
        slopeErrTech = pd.DataFrame(slopeErrTech, columns = columns)
        slopeErrAvg = pd.DataFrame(slopeErrAvg, columns=columns)
        
        errpred = []
        errpred2 = []
        slopeErrTech_ = []
        slopeErrAvg_ = []
        # for tech in dferrTech['Tech'].unique():
        for tech in ['Fotovoltaica','Transistor','DRAM','Hard_Disk_Drive']:
            errpred.append(dferrTech.loc[dferrTech['Tech']==tech, 'Error'].values)
            errpred2.append(dferrAvg.loc[dferrAvg['Tech']==tech, 'Error'].values)
            # slopeErrTech_.append(slopeErrTech.loc[slopeErrTech['Tech']==tech, 'Error'].values)
            # slopeErrAvg_.append(slopeErrAvg.loc[slopeErrAvg['Tech']==tech, 'Error'].values)

        print('\n\n\n')
        print(tOrd, fOrd)
        print(dferrTech['Tech'].nunique(), ' Techs')
        if dferrTech['Tech'].nunique() <= 1:
            print('Not enough technologies')
            continue
        analysisFunctions.performTPairedTest(errpred, errpred2)

        analysisFunctions.performWilcoxonSignedRankTest(errpred, errpred2)

        # analysisFunctions.performTPairedTest(slopeErrTech_, slopeErrAvg_)

        # analysisFunctions.performWilcoxonSignedRankTest(slopeErrTech_, slopeErrAvg_)

exit()


method = 'regression'

# get slope for all technologies
slopes = []
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
    e1 = []
    e2 = []
    # computing average technological slope based on all other technologies
    slopeall = np.mean(slopes.loc[slopes['Tech'] != tech,'Slope'].values)
    # computing technology specific slope
    sel = df.loc[df['Tech']==tech]
    x = np.log10(sel['Cumulative production'].values)
    y = np.log10(sel['Unit cost'].values)
    H = len(x)
    # calibrate model over first set of points
    # for i in range(H):
        # for N in range(0 - 1*(i==0), -1, -1):
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
                # pred =  y[i] + slope * (x[M] - x[i])
                if method=='regression':
                    pred = result.params[0] + slope * x[M]
                pred2 =  y[i] + slopeall * (x[M] - x[i])
                error = (y[M] - (pred))
                error2 = (y[M] - (pred2))
                e1.append(error**2)
                e2.append(error2**2)
                # error point by point
                # RMSEdiff.append((error**2)**0.5-(error2**2)**0.5)
    # error by technology
    RMSEdiff.append(np.mean(e1)**0.5-np.mean(e2)**0.5)
RMSEdiff = pd.DataFrame(RMSEdiff, columns=['diff'])
N = RMSEdiff['diff'].nunique()

print('Paired t-test: null hypothesis rejected if value is outside [' + \
      str(scipy.stats.t.ppf(0.025, N-1).round(3))+ ','+str(scipy.stats.t.ppf(0.975, N-1).round(3))+']')
mu = np.mean(RMSEdiff['diff'].values)
std = np.std(RMSEdiff['diff'].values) / (RMSEdiff.shape[0])**0.5
print('\t The value is ', mu/std)

exit()
print('Wilcoxon signed rank test: null hypothesis rejected if value is outside [-1.96,1.96]')
RMSEdiff['abs'] = np.abs(RMSEdiff['diff'].values)
RMSEdiff = RMSEdiff.sort_values(by='abs', ascending=True)
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

for sector in analysisFunctions.sectors:
    print('\n\n',sector)
    RMSEdiff = []
    for tech in analysisFunctions.sectors[sector]:
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

    print('Paired t-test: null hypothesis rejected if value is outside [' + \
        str(scipy.stats.t.ppf(0.025, N-1).round(3))+ ','+str(scipy.stats.t.ppf(0.975, N-1).round(3))+']')
    mu = np.mean(RMSEdiff['diff'].values)
    std = np.std(RMSEdiff['diff'].values) / (RMSEdiff.shape[0])**0.5
    print(mu, std)
    print('\t The value is ', mu/std)
    # print('The mean difference is ', mu)   
    # print('\t The confidence interval for the mean is (', mu-1.990*std,', ', mu+1.990*std,')')


    print('Wilcoxon signed rank test: null hypothesis rejected if value is outside [-1.96,1.96]')
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
