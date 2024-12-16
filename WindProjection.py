import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, scipy, utils
import statsmodels.api as sm
import piecewise_regression as pw
import cmcrameri as cm

sns.set_style("ticks")
sns.set_context("talk")
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300

color_pw = cm.cm.batlow(180)
color_l = cm.cm.batlow(30)
alpha=0.2

boxplot = True

validation = True
val_year = 2000

# read original wind electricity data from Performance Curve Database
df = pd.read_csv('expCurveData' + os.path.sep + 'Wind_Electricity.csv')
df.columns = ['Unit cost', 'Year', 'Production', 'Cumulative production']

# from cents 2005 USD / kWh to 2005 USD / MWh
df['Unit cost'] = df['Unit cost'] / 100 * 1e3 

# read generation data from IEA
newdata_prod = pd.read_csv(
    'AdditionalData' + os.path.sep + 
    'Electricity_generation_by_source_World_IEA.csv',
                           header=2)
newdata_prod = newdata_prod[[newdata_prod.columns[0], 'Wind']]
newdata_prod.columns = ['Year','Wind']
newdata_prod['Wind'] *= 1e3 # transform GWh to MWh
newdata_prod = newdata_prod['Wind'].values


# read LCOE data from IEA
newdata_cost = pd.read_excel(
    'AdditionalData' + os.path.sep + 
    'IRENA_Costs_in_2022_rawdatafile.xlsx',
                             sheet_name='Fig 2.11', header=3)
newdata_cost = newdata_cost.iloc[:,:]
newdata_cost = newdata_cost['Weighted average'].values

# read generation data from Bolinger et al., 2022
add_prod_data = pd.read_csv(
    'AdditionalData' + os.path.sep + 
    'BolingerEtAl2022' + os.path.sep + 
    'Data_and_Scripts' + os.path.sep + 
    'Data' + os.path.sep + 'lcoe_learning_data_wind.csv')

add_prod_data = add_prod_data[['year','wind_global_gwh','raw_lcoe']]
add_prod_data_1984_1990 = add_prod_data.loc[(add_prod_data['year']<=1990)
                                             & (add_prod_data['year']>=1984)]

# put together cumulative production from Wiser (1984-1990) 
# and production IEA (1990-2021 -> 1991-2022)
prod = add_prod_data_1984_1990['wind_global_gwh'].values*1000 #gwh to mwh
prod1990_2021 = np.cumsum(newdata_prod)
prod1990_2021 = prod1990_2021 + prod[-1]
prod = np.concatenate([prod, prod1990_2021])

year = np.arange(1984,2023)

cost = newdata_cost * 1000.0 # from 2022 USD / kWh to 2022 USD / MWh

# build dataframe
newdf = pd.DataFrame({ 'Unit cost (2022 USD/MWh)':cost, 
                      'Time (Year)':year, 
                      'Production (MWh)': np.insert(np.diff(prod),0,np.nan), 
                      'Cumulative production (MWh)':prod})
# save new wind series
newdf.to_csv('wind_Bolinger_IEA_IRENA.csv', index=False)

# run multiple piecewise regression experiments to find the best model
nexp = 10 # number of experiments to evaluate best models
best = utils.repeat_piecewise_model_selection(
    newdf, nexp, val_year, validation)

# select best model
IC = best.loc[best['RSS']==best['RSS'].min(), 
              best.columns[1:]].reset_index(drop=True)

# split data into calibration and validation
prod, cost, prod_val, cost_val = utils.split_calibration_validation(
    newdf, val_year)

# plot calibration
fig, ax = utils.plot_calibration_piecewise(
    newdf, val_year, IC, color_pw)

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
proj, fut_prod = utils.ensemble_forecast_piecewise(newdf, 
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
        color=color_pw, ls=':', zorder=-8,lw=1)
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

## simulate future costs
proj = []
nsim = 10000

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

# compute crps
crps_wright = []
for i in range(proj.shape[1]):
    crps_wright.append(
        utils.compute_crps(proj[:,i], np.log10(cost_val[i])))
crps_wright = np.array(crps_wright)

print('CRPS piecewise:', crps_piecewise.mean())
print('CRPS wright:', crps_wright.mean())
print('CRPSs piecewise:', crps_piecewise)
print('CRPSs wright:', crps_wright)
print('Paired t test:', scipy.stats.ttest_rel(crps_piecewise, crps_wright))    

ax.fill_between(10**fut_prod, 10**np.percentile(proj, 5, axis=0),
                10**np.percentile(proj, 95, axis=0), 
                color=color_l, alpha=alpha, zorder=-10, lw=0)
ax.plot(10**fut_prod, 10**np.percentile(proj, 5, axis=0), 
        color=color_l, ls=':', zorder=-8, lw=1)
ax.plot(10**fut_prod, 10**np.percentile(proj, 95, axis=0), 
        color=color_l, ls=':', zorder=-8, lw=1)
ax.plot(10**fut_prod, 10**np.median(proj, axis=0), 
        color=color_l, lw=2, zorder=-5,
        label='First difference Wright\'s law')

plt.legend()
plt.title('Wind power')
plt.tight_layout()

ax.minorticks_off()

plt.legend(loc='lower left')
plt.title('Wind power')
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

plt.gcf().savefig('figs' + os.path.sep + 
                  'WindProjection' +
                  '_val' * validation + '.png')
plt.gcf().savefig('figs' + os.path.sep + 
                  'WindProjection' +
                  '_val' * validation + '.pdf')

plt.show()