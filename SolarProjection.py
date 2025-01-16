import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy, os, utils
import piecewise_regression as pw
import statsmodels.api as sm
import cmcrameri as cm

sns.set_style("ticks")
sns.set_context("talk")
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300

color_pw = cm.cm.batlow(180)
color_l = cm.cm.batlow(50)
alpha=0.2

boxplot = True

validation = True
val_year = 2000

# load data
df = pd.read_csv('ExpCurves.csv')

# select pv
df = df.loc[df['Tech']=='Photovoltaics_2']

# set random seed
np.random.seed(0)

## read production data data
newdata_prod = pd.read_csv(
        'AdditionalData' + os.path.sep + 
        'Electricity_generation_by_source_World_IEA.csv',
                           header=2)
newdata_prod = newdata_prod[[newdata_prod.columns[0], 'Solar PV']]
newdata_prod.columns = ['Year','Solar PV']
newdata_prod['Solar PV'] *= 1e3 # transform GWh to MWh
newdata_prod = newdata_prod['Solar PV'].values 

# starting from 1990 cuumlative production add the new production
# this data series ranges from 1990 to 2021
# allows to compute cumulative production 1991-2022
newdata_prod = (df.loc[df['Year']==1990, 
                      'Cumulative production'].values[0]*1e-3 
                      + np.cumsum(newdata_prod))

# read cost data
newdata_cost = pd.read_excel(
        'AdditionalData' + os.path.sep + 
        'IRENA_RenewablePowerGenerationCosts_2022.xlsx',
                             sheet_name='Fig 3.1', header=22, index_col=1)
# get average LCOE
newdata_cost = newdata_cost.loc['Weighted average'].values[1:]

# insert deflated cost (1 2005USD = 1.5 2022USD) to the left
# this covers value from 1991 to 2009
# IEA data is used from 2010 to 2022
for x in df.loc[df['Year']>1990,'Unit cost'].values[::-1]:
    newdata_cost = np.insert(newdata_cost, 0, x*1.5) 
newdata_cost *= 1e3

# combine into new data series
prod = df.loc[df['Year']<=1990,'Cumulative production'].values*1e-3
prod = np.concatenate([prod,newdata_prod])

# convert cost from 2005USD to 2022USD
cost = df.loc[df['Year']<=1990,'Unit cost'].values * 1.5 * 1e3 
cost = np.concatenate([cost,newdata_cost])

new_solar_series = pd.DataFrame({'Unit cost (2022 USD/MWh)': cost,
                                 'Time (Year)': np.arange(1977,2023),
                                 'Production (MWh)': np.diff(prod, append=0),
                                'Cumulative production (MWh)': prod})

# save combined solar pv data series
new_solar_series.to_csv('solar_pv_PCDB_IEA_IRENA.csv', index=False)

# run multiple piecewise regression experiments to find the best model
nexp = 10 # number of experiments to evaluate best models
best = utils.repeat_piecewise_model_selection(
    new_solar_series, nexp, val_year, validation)

# select best model
IC = best.loc[best['RSS']==best['RSS'].min(), 
              best.columns[1:]].reset_index(drop=True)

# split data into calibration and validation
prod, cost, prod_val, cost_val = utils.split_calibration_validation(
    new_solar_series, val_year)

# plot calibration
fig, ax = utils.plot_calibration_piecewise(
    new_solar_series, val_year, IC, color_pw)

# compute residuals of piecewise regression
resid = utils.compute_residuals_piecewise(prod, cost, IC)

# evaluate different models for residuals
modelerr = sm.tsa.SARIMAX(resid, order=(1,0,0)).fit()
print(modelerr.summary())
modelerr = sm.tsa.SARIMAX(resid, order=(0,0,0)).fit()
print(modelerr.summary())
modelerrp = modelerr.params
# no autocorrelation is better (lower BIC)
modelerrp = np.insert(modelerrp, 0, 0)

# read parameters of distribution of distance betweeen breakpoints
params_breaks_lexp = pd.read_csv('params_breaks_lexp.csv')
# parameters for distance between breakpoints
params_breaks = (params_breaks_lexp
                 .loc[params_breaks_lexp['Variable']=='breaks'])
dist_breaks = getattr(scipy.stats, params_breaks['Distribution'].values[0])
dist_breaks = dist_breaks(params_breaks['Loc'].values[0],
                          params_breaks['Scale'].values[0])
# parameters for learning exponent
params_lexp = params_breaks_lexp.loc[params_breaks_lexp['Variable']=='lexp']
dist_lexp = getattr(scipy.stats, params_lexp['Distribution'].values[0])
dist_lexp = dist_lexp(params_lexp['Loc'].values[0],
                          params_lexp['Scale'].values[0])

## simulate future costs
# select number of simulations
nsim = 10000
proj, fut_prod = utils.ensemble_forecast_piecewise(new_solar_series, 
                                                   IC,
                                                   validation,
                                                   val_year,
                                                   dist_breaks,
                                                   dist_lexp,
                                                   modelerrp, 
                                                   nsim)

# compute crps
crps_piecewise = []
for i in range(proj.shape[1]):
    crps_piecewise.append(
        utils.compute_crps(proj[:,i], np.log10(cost_val[i])))
crps_piecewise = np.array(crps_piecewise)
                  
ax.fill_between(10**fut_prod, 10**np.percentile(proj, 5, axis=0),
                10**np.percentile(proj, 95, axis=0), 
                color=color_pw, alpha=alpha, zorder=-10, lw=0)
ax.plot(10**fut_prod, 10**np.percentile(proj, 5, axis=0), 
        color=color_pw, ls=':', zorder=-8, lw=1)
ax.plot(10**fut_prod, 10**np.percentile(proj, 95, axis=0), 
        color=color_pw, ls=':', zorder=-8, lw=1)
ax.plot(10**fut_prod, 10**np.median(proj, axis=0), 
        color=color_pw, lw=2, zorder=-5,
        label='Piecewise linear experience curve')

## overlay first difference wright's law projection
cost_d = np.diff(np.log10(cost))
prod_d = np.diff(np.log10(prod))

slope = np.sum(cost_d * prod_d) / np.sum(prod_d**2)

residuals = []
for i in range(len(cost_d)):
    residuals.append(cost_d[i] - slope * prod_d[i])

residuals = np.array(residuals)

# build ar1 model of residuals
res = sm.tsa.SARIMAX(residuals, order=(1,0,0)).fit()
print(res.summary())
ar1 = res.params[0]
sigma = res.params[1]

# parameters from Way et al., 2022
if validation is False:
    slope = -0.421
    ar1 = .19
    sigma = 0.103
else:
    ar1 = .19
    sigma = (np.var(residuals, ddof=1)/(1+ar1**2))**0.5

## build projections using linear regression model
proj = []

if validation:
    starting_point = 1
    fut_prod = np.log10(prod_val)

for s in range(nsim):
    yi = [np.log10(cost[-starting_point])]
    residual = residuals[-1]
    for xi in np.diff(fut_prod):
        residual = ar1 * residual + sigma * np.random.randn()
        yi.append(yi[-1] + slope * xi + residual)

    proj.append(yi)
proj = np.array(proj)

crps_wright = []
for i in range(proj.shape[1]):
    crps_wright.append(
        utils.compute_crps(proj[:,i], np.log10(cost_val[i])))
crps_wright = np.array(crps_wright)

# print results
print('CRPS piecewise:', crps_piecewise.mean())
print('CRPS wright:', crps_wright.mean())
print('CRPSs piecewise:', crps_piecewise)
print('CRPSs wright:', crps_wright)
print('Paired t test:', scipy.stats.ttest_rel(crps_piecewise, crps_wright))    

ax.fill_between(10**fut_prod, 10**np.percentile(proj, 5, axis=0),
                10**np.percentile(proj, 95, axis=0), 
                color=color_l, alpha=alpha, zorder=-10, lw=0)
ax.plot(10**fut_prod, 10**np.percentile(proj, 5, axis=0), 
        color=color_l, ls=':', zorder=-8,lw=1)
ax.plot(10**fut_prod, 10**np.percentile(proj, 95, axis=0), 
        color=color_l, ls=':', zorder=-8, lw=1)
ax.plot(10**fut_prod, 10**np.median(proj, axis=0), 
        color=color_l, lw=2, zorder=-5,
        label='First difference Wright\'s law')

ax.minorticks_off()
plt.legend(loc='lower left')
plt.title('Solar photovoltaics')
plt.tight_layout()
axes = fig.add_axes([0.3, 0.4, 0.15, 0.25])
axes.axis('off')
axes.fill_between([0.5,1], [0,0], [1,1], color='#47666F', alpha=.2, lw=0)
axes.plot([0.5,1], [0.5,0.5], color='#47666F', lw=2)
axes.plot([0.5,1], [1,1], color='#47666F', ls=':', lw=2)
axes.plot([0.5,1], [0,0], color='#47666F', ls=':', lw=2)
axes.annotate('90%', xy=(1.5,1.05), xycoords='data', 
            ha='center', va='center', color='k',
            fontsize=12)
axes.annotate('Median', xy=(1.5,0.5), xycoords='data',
            ha='center', va='center', color='k',
            fontsize=12)

axes.plot([1.1,2.1,2.1,1.1], [0,0,1,1], color='k', lw=.2)

if not os.path.exists('figs'):
    os.makedirs('figs')

plt.gcf().savefig('figs' + os.path.sep + 
                    'SolarPVProjection' + 
                    '_val' * validation + '.png')
plt.gcf().savefig('figs' + os.path.sep + 
                    'SolarPVProjection' + 
                    '_val' * validation + '.pdf')

plt.show()