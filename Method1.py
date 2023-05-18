import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('ExpCurves.csv')
df = pd.read_csv('NormalizedExpCurves.csv')

fraction = 1/2

LR_cal = []
LR_val = []
for tech in df['Tech'].unique():
    # read data for specific technology
    sel = df.loc[df['Tech'] == tech]
    try:
        x = np.log10(sel['Cumulative production'].values)
        y = np.log10(sel['Unit cost'].values)
    except KeyError:
        x = np.log10(sel['Normalized cumulative production'].values)
        y = np.log10(sel['Normalized unit cost'].values)
    # separate calibration and validation datasets based on points
    x_cal = x[:int(x.shape[0]*fraction)]
    x_val = x[int(x.shape[0]*fraction):]
    y_cal = y[:int(y.shape[0]*fraction)]
    y_val = y[int(y.shape[0]*fraction):]
    # alternative - separate calibration and validation datasets based on fraction of cumulative production
    x_cal = x[x<(x[-1]-x[0])*fraction]
    x_val = x[x>=(x[-1]-x[0])*fraction]
    y_cal = y[x<(x[-1]-x[0])*fraction]
    y_val = y[x>=(x[-1]-x[0])*fraction]
    # perform regression on both datasets separately
    model_cal = sm.OLS(y_cal, sm.add_constant(x_cal))
    result_cal = model_cal.fit()
    model_val = sm.OLS(y_val, sm.add_constant(x_val))
    result_val = model_val.fit()
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()
    LRall = result.params[1]
    plt.scatter(result_cal.params[1], result_val.params[1], 
                color='tab:red', facecolor='None', alpha = 0.6)
    LR_cal.append(result_cal.params[1])
    LR_val.append(result_val.params[1])
plt.xlabel('Early learning rate')
plt.ylabel('Late learning rate')

model = sm.OLS(LR_val, sm.add_constant(LR_cal))
result = model.fit()
print(result.summary())
plt.plot(np.linspace(-2,2,100), 
         result.params[0] + result.params[1] * np.linspace(-1,1,100),
         zorder = -1, label='Regression')
plt.plot([-2,2],[np.mean(LR_cal),np.mean(LR_cal)], label='Mean past learning rate')
plt.plot([-2,2],[np.mean(LR_val),np.mean(LR_val)], label='Mean future learning rate')
plt.gca().axis('equal')
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.annotate('R2 = ' + str(round(result.rsquared,2)) + \
             '\n N = ' + str(len(df['Tech'].unique())),
             (-1.5,1))
plt.annotate('late LR = ' + str(round(result.params[0],2)) + \
             '+ ' + str(round(result.params[1],2)) +'* early LR'+\
                '\n Significancy (F test p-value): '+str(round(result.f_pvalue,3)),
                (1.2,-1.25), ha='center')
ssetot = 0.0
ssereg = 0.0
ssecal = 0.0
ssecal2 = 0.0
sseval = 0.0
for lrv, lrc in zip(LR_val, LR_cal):
    ssetot += (lrv-np.mean(LR_cal))**2
    # ssetot += (lrv-np.mean(LRall))**2
    ssereg += (lrv-result.params[0]-result.params[1]*lrc)**2
    ssecal += (lrv-np.mean(LR_cal))**2
    ssecal2 += (lrv-lrc)**2
    sseval += (lrv-np.mean(LR_val))**2
print('Mean past and future learning rates: ')
print(np.mean(LR_cal), np.mean(LR_val))
print('R2 for regression and for prediction using mean past learning rate for all technologies: ')
print(1 - ssereg/ssetot)
print(1 - ssecal/ssetot)
print(1 - ssecal2/ssetot)
print(1 - sseval/ssetot)
plt.legend()
plt.show()