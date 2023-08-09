import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, analysisFunctions, plottingFunctions

matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('ExpCurves.csv')

### SELECT SCRIPT OPTIONS
# fraction of points or cumulative production interval used for calibration
fraction = 1/2
# split the dataset based on points (True) or cumulative production interval (False)
points = True
# include nuclear technologies (True) or not (False)
nuclearIncluded = True

if nuclearIncluded == False:
    df = df.loc[~(df['Tech'].str.contains('Nuclear'))]

df['Sector'] = [analysisFunctions.sectorsinv[tech] for tech in df['Tech']]
sectorTech = [analysisFunctions.sectorsinv[tech] for tech in df['Tech'].unique()]

LR_cal, LR_val, slopesall, \
    uc, cpCal, cpVal, \
    ucpred, errpred, ucpred2, errpred2 = \
        analysisFunctions.computeRegPredError(df, fraction, points)

print("Average Wright's exponent: ",np.mean(slopesall), 
    "\nAverage percentage cost reduction per doubling of cumulative production: ", 
    100 * (1 - 2**(np.mean(slopesall))), "%")

plottingFunctions.scatterFigure(LR_cal, LR_val, sectorTech)
plt.show()

## supplementary figures
for sector in df['Sector'].unique():
    LR_cal_ = []
    LR_val_ = []
    sectorTech_ = []
    for item in zip(LR_cal, LR_val, sectorTech):
        if item[2] == sector:
            LR_cal_.append(item[0])
            LR_val_.append(item[1])
            sectorTech_.append(item[2])
    plottingFunctions.scatterFigure(LR_cal_, LR_val_, sectorTech_)


plottingFunctions.gridPlots(uc, cpCal, cpVal, ucpred, errpred, ucpred2, errpred2)

plottingFunctions.barSectorMSE(errpred, errpred2, sectorTech)


# Monte Carlo sampling with replacement
R2 = {}
label = 'All Technologies (N = '+str(df['Tech'].nunique())+')'
R2[label] = analysisFunctions.computeR2MonteCarlo(LR_cal, LR_val, sectorTech)
for sector in sectorTech:
    if sector == 'Genomics':
        continue
    LR_cal_ = []
    LR_val_ = []
    sectorTech_ = []
    for item in zip(LR_cal, LR_val, sectorTech):
        if item[2] == sector:
            LR_cal_.append(item[0])
            LR_val_.append(item[1])
            sectorTech_.append(item[2])
    label = sector+' (N = '+str(len(LR_cal_))+')'
    R2[label] = analysisFunctions.computeR2MonteCarlo(LR_cal_, LR_val_, sectorTech_)
R2 = pd.DataFrame(R2)
fig, ax = plt.subplots(1,1, figsize=(11,6))
R2.boxplot(ax=ax, whis=99)
ax.grid(False)
ax.set_ylabel('R2')
plt.subplots_adjust(right=0.95, left=0.05, bottom=0.1, top=0.95)

plt.show()