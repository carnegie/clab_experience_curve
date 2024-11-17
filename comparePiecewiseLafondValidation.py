import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy, matplotlib, utils

def compute_crps(forecasts, obs):
    fcst = np.sort(forecasts)
    n = fcst.shape[0]

    # CRPS integral between -inf and +inf of:
    #    \int_{-inf}^{+inf} { (F(x) - H(x>=y))**2 dx }

    # e.g. y = 2, x = [1, 1]
    # \int_{-inf}^{1}{0} + \int{1}{2}{1} + \int{2}{inf}{0}
    # is this equal to one? yes

    # e.g. y = 2, x = [2, 2]
    # \int_{-inf}^{2}{0} + \int{2}{inf}{0}
    # is this equal to zero? yes

    # get values for integration domain
    domain = np.copy(fcst)
    domain = np.append(fcst, obs)
    domain = np.sort(domain)

    crps = 0
    for i in range(domain.shape[0]-1):
        cdf_fcst = sum(domain[i]>= fcst)/n
        cdf_obs = 1 if domain[i] >= obs else 0
        crps += (domain[i+1] - domain[i]) * (cdf_fcst - cdf_obs)**2
    return crps

# set figures' parameters
sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300
sns.color_palette('colorblind')
palette = 'crest'

# select input data
half_data = True
half_techs = False
remove_sector = None

# compose name of input file
input_file = ''
input_file += '_half_data' if half_data else ''
input_file += '_half_techs' if half_techs else ''
input_file += '_no'+remove_sector if remove_sector is not None else ''

# set random seed
np.random.seed(0)

# load data
df = pd.read_csv('ExpCurves.csv')

# load IC data
IC_half_data = pd.read_csv('IC'+input_file+'.csv')

# read params from analysis of piecewise regression
params_breaks_lexp = pd.read_csv('params_breaks_lexp'+input_file+'.csv')
breaks_dist = scipy.stats.expon(loc = params_breaks_lexp\
                                .loc[params_breaks_lexp['Variable'] == 'breaks', 'Loc'].values[0],
                                scale = params_breaks_lexp\
                                .loc[params_breaks_lexp['Variable'] == 'breaks', 'Scale'].values[0])
lexp_dist = scipy.stats.norm(loc = params_breaks_lexp\
                                .loc[params_breaks_lexp['Variable'] == 'lexp', 'Loc'].values[0],
                                scale = params_breaks_lexp\
                                .loc[params_breaks_lexp['Variable'] == 'lexp', 'Scale'].values[0])

# compare piecewise linear experience curves with 
# first difference wrights law using half of points for all techs
crps, picps = [], []
plotFigTech = False
nsim = 100

# iterate for each technology
for t in df['Tech'].unique():

    # remove prespecified sector if available
    if remove_sector is not None:
        if utils.sectorsinv[t] == remove_sector:
            continue

    # get cost and production data for technology
    x = df[df['Tech'] == t]['Cumulative production'].values
    y = df[df['Tech'] == t]['Unit cost'].values

    # plot data
    if plotFigTech:
        plt.figure(figsize=(10,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(x,y, label='Data')

    # LINEAR REGRESSION MODEL
    # use half of the points to compute error of the model
    # using first difference wright's law

    # compute first differenced time series
    x_diff_cal = np.diff(np.log10(x[:round(x.shape[0]/2)]))
    y_diff_cal = np.diff(np.log10(y[:round(y.shape[0]/2)]))

    # build linear regression model, fit it, and save learning exponent
    model = sm.OLS(y_diff_cal, x_diff_cal).fit()
    lexp = model.params[0]

    # store error parameters
    noise = model.resid
    model = sm.OLS(noise[1:], noise[:-1]).fit()
    ar1_noise = model.params[0]
    residuals = model.resid
    noise_std = np.std(residuals, ddof=1)
    if np.isnan(noise_std):
        noise_std = np.std(residuals)

    # compute cost during calibration period using calibrated model
    y_diff_cal_pred = np.zeros(y_diff_cal.shape[0])
    noise = np.random.randn() * noise_std
    for i in range(y_diff_cal.shape[0]):
        noise = ar1_noise * noise + np.random.randn() * noise_std
        y_diff_cal_pred[i] = lexp * x_diff_cal[i] #+ noise
    y_diff_cal_pred = 10**(np.log10(y[:round(x.shape[0]/2)-1]) + y_diff_cal_pred)
    y_diff_cal_pred = np.insert(y_diff_cal_pred,0,y[0])
                           
    # y_diff_cal_pred = 10**(np.log10(y[0]) + np.insert(np.cumsum(results.predict()),0,0))
    # plt.scatter(x[:round(x.shape[0]/2)], y_diff_cal_pred, color='red')

    # PIECEWISE REGRESSION MODEL
    # use half of the points to compute error of the model
    # using piecewise linear regression

    x_cal = np.log10(x[:round(x.shape[0]/2)])
    y_cal = np.log10(y[:round(y.shape[0]/2)])

    # get model parameters from IC
    sel = IC_half_data.loc[(IC_half_data['Tech'] == t) & \
                           (IC_half_data['First Diff.']==0)]
    sel = sel.loc[sel['BIC'].idxmin()]

    # get breakpoints
    breaks = np.array([sel['Breakpoint '+str(i)] for i in range(1,7)])
    breaks = breaks[~np.isnan(breaks)]

    # get learning rates
    lrs = np.array([sel['LR '+str(i)] for i in range(1,8)])
    lrs = lrs[~np.isnan(lrs)]
    lexps = np.log2(-(lrs/100 - 1))

    # # compute error of the model for each x_cal
    y_cal_pred = []
    for i in range(x_cal.shape[0]):
        if breaks.shape[0] == 0 or x_cal[i] < breaks[0]:
            cost = sel['Intercept'] + lexps[0] * x_cal[i]
        else:
            cost = sel['Intercept'] + \
                lexps[0] * breaks[0]
            j = 1
            while j < breaks.shape[0] and breaks[j] < x_cal[i]:
                cost += lexps[j] * \
                        (breaks[j] - breaks[j-1])
                j += 1
            else:
                cost += lexps[j] * \
                            (x_cal[i] - breaks[j-1])
        y_cal_pred.append(cost)

    y_cal_pred = np.array(y_cal_pred)

    noise_pw = np.std(y_cal_pred - y_cal, ddof=1)

    # if plotFigTech:
    #     plt.scatter(10**x_cal, 10**y_cal_pred, color='blue')

    ## PREDICT

    # use the remaining points for prediction and error comparison

    x_val = np.log10(x[round(x.shape[0]/2)-1:])
    y_val = np.log10(y[round(y.shape[0]/2)-1:])

    # LINEAR REGRESSON MODEL
    # forecast using first difference wright's law
    x_diff_val = np.diff(x_val)

    y_diff_val_sim = np.zeros((nsim, x_diff_val.shape[0]+1))
    for n in range(nsim):
        y_diff_val_pred = np.array([np.log10(y[round(y.shape[0]/2)-1])])
        # noise = results.resid[-1]
        noise = 0
        for i in range(x_diff_val.shape[0]):
            noise = 0.19 * noise + np.random.randn() * noise_std
            y_diff_val_pred = np.append(y_diff_val_pred,
                                    y_diff_val_pred[-1] + \
                                        lexp * x_diff_val[i] + \
                                            noise )
        y_diff_val_pred = 10**(y_diff_val_pred)
        y_diff_val_sim[n,:] = y_diff_val_pred

    # compute continuous rank probability score
    crps_ = np.zeros(y_diff_val_sim.shape[1])
    # iterate over observations in the validation period
    for i in range(y_diff_val_sim.shape[1]):
        # get the remaining observations
        obs = y[round(y.shape[0]/2)-1:]
        # get the prediction for the i-th element of the validation period
        preds = y_diff_val_sim[:,i]
        crps_[i] = compute_crps(preds, obs[i])
        # # build a cumulative distribution function
        # preds = np.sort(preds)
        # diff_preds = np.diff(preds)
        # diff_preds = np.append(diff_preds, 0.0)

        # for n in range(nsim):
        #     crps_[i] += diff_preds[n] * (n/nsim - 1.0*(preds[n] >= obs[i]))**2

    crps_wright = np.sum(crps_)


    # compute log scoring
    picp_ = np.zeros(y_diff_val_sim.shape[1])
    for i in range(y_diff_val_sim.shape[1]):
        obs = y[round(y.shape[0]/2)-1:]
        preds = y_diff_val_sim[:,i]
        preds_95 = np.percentile(preds, 95)
        preds_5 = np.percentile(preds, 5)
        
        picp_[i] = 1.0*(obs[i] >= preds_5 and obs[i] <= preds_95)

    picp_wright = np.sum(picp_)/y_diff_val_sim.shape[1]

    if plotFigTech:
        plt.plot(x[round(x.shape[0]/2)-1:], np.median(y_diff_val_sim, axis=0), color='red',
                 label='Lafond - median')
        plt.fill_between(x[round(x.shape[0]/2)-1:], np.percentile(y_diff_val_sim, 95, axis=0), 
                        np.percentile(y_diff_val_sim, 5, axis=0), alpha=0.2, color='red',
                        label='Lafond - 5-95%')


    # forecast using piecewise linear regression

    y_val_sim = np.zeros((nsim, x_val.shape[0]))
    for n in range(nsim):
        
        # get breakpoints
        breaks = np.array([sel['Breakpoint '+str(i)] for i in range(1,7)])
        breaks = breaks[~np.isnan(breaks)]
        # get learning rates
        lrs = np.array([sel['LR '+str(i)] for i in range(1,8)])
        lrs = lrs[~np.isnan(lrs)]
        lexps = np.log2(-(lrs/100 - 1))

        if breaks.shape[0] == 0:
            breaks = np.array([x_cal[-1]])

        ## extend breaks and lrs until the end of the dataset
        while breaks[-1] < x_val[-1]:
            breaks = np.append(breaks, max(x_cal[-1], breaks[-1] + min(np.log10(2), breaks_dist.rvs())))
            lexps = np.append(lexps, lexp_dist.rvs())

        # compute prediction for each x_val
        y_val_pred = []
        for i in range(x_val.shape[0]):
            if breaks.shape[0] == 0 or x_val[i] < breaks[0]:
                cost = sel['Intercept'] + (lexps[0]) * x_val[i]
            else:
                cost = sel['Intercept'] + \
                    lexps[0] * breaks[0]
                j = 1
                while j < breaks.shape[0] and breaks[j] < x_val[i]:
                    cost += lexps[j] * \
                            (breaks[j] - breaks[j-1])
                    j += 1
                else:
                    cost += lexps[j] * \
                                (x_val[i] - breaks[j-1])
            y_val_pred.append(cost + np.random.randn() * noise_pw)

        y_val_pred = np.array(y_val_pred)
        y_val_sim[n,:] = 10**y_val_pred
    
    # compute continuous rank probability score
    crps_ = np.zeros(y_val_sim.shape[1])
    for i in range(y_val_sim.shape[1]):
        obs = y[round(y.shape[0]/2)-1:]
        preds = y_val_sim[:,i]
        crps_[i] = compute_crps(preds, obs[i])
        # preds = np.sort(preds)
        # diff_preds = np.diff(preds)
        # diff_preds = np.append(diff_preds, 0.0)

        # for n in range(nsim):
        #     crps_[i] += diff_preds[n] * (n/nsim - 1.0*(preds[n] >= obs[i]))**2

    crps_piecewise = np.sum(crps_)

    # compute log scoring
    picp_ = np.zeros(y_val_sim.shape[1])
    for i in range(y_val_sim.shape[1]):
        obs = y[round(y.shape[0]/2)-1:]
        preds = y_val_sim[:,i]
        preds_95 = np.percentile(preds, 95)
        preds_5 = np.percentile(preds, 5)
        
        picp_[i] = 1.0*(obs[i] >= preds_5 and obs[i] <= preds_95)

    picp_piecewise = np.sum(picp_)/y_val_sim.shape[1]

    crps.append([crps_wright, crps_piecewise])
    picps.append([picp_wright, picp_piecewise])

    if plotFigTech:
        plt.plot(10**x_val, np.median(y_val_sim, axis=0), color='blue',
                 label='Piecewise linear regression - median')
        plt.fill_between(10**x_val, np.percentile(y_val_sim, 95, axis=0), 
                        np.percentile(y_val_sim, 5, axis=0), alpha=0.2, color='blue',
                        label='Piecewise linear regression - 5-95%') 
        plt.title(t)   
        plt.xlabel('Cumulative production')
        plt.ylabel('Unit cost')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/Users/angelocarlino/Desktop/Techs_comparison/'+t+'.png')
        plt.close('all')

plt.figure()
plt.bar([0,1], [np.sum(np.array(crps)[:,0] < np.array(crps)[:,1]), 
         np.sum(np.array(crps)[:,0] > np.array(crps)[:,1])])
plt.title('Continuous Ranked Probability Score')
plt.ylabel('Number of technologies scoring better')
plt.xticks([0,1],['Lafond', 'Piecewise'])

print('CRPS:')
print(crps)
print('\n\n\n\nPICP:')
print(picps)

plt.figure()
plt.bar([0,1], [np.sum(np.array(picps)[:,0] > np.array(picps)[:,1]), 
         np.sum(np.array(picps)[:,0] < np.array(picps)[:,1])])
plt.title('Prediction Interval Coverage Probability')
plt.ylabel('Number of technologies scoring better')
plt.xticks([0,1],['Lafond', 'Piecewise'])

picps = np.array(picps)
for i in range(picps.shape[0]):
    if picps[i,0] < picps[i,1]:
        picps[i,0] = 0
        picps[i,1] = 1
    elif picps[i,0] > picps[i,1]:
        picps[i,0] = 1
        picps[i,1] = 0
    else:
        picps[i,0] = 0.5
        picps[i,1] = 0.5


plt.figure(figsize=(16,8))
sns.heatmap(np.array(picps).transpose(), cmap=palette,
            cbar_kws={'label':'Prediction Interval Coverage Probability',
                      'shrink':0.5})
plt.xticks([x+0.5 for x in range(87)], [x.replace('_',' ') for x in df['Tech'].unique()], rotation=90)
plt.yticks([0.5,1.5],['Lafond', 'Piecewise'])
plt.tight_layout()

# crps_max = np.max(crps, axis=1)
# crps = np.array(crps)
# crps[:,0] = crps[:,0] / crps_max
# crps[:,1] = crps[:,1] / crps_max

crps = np.array(crps)
for i in range(crps.shape[0]):
    if crps[i,0] < crps[i,1]:
        crps[i,0] = 1
        crps[i,1] = 0
    elif crps[i,0] > crps[i,1]:
        crps[i,0] = 0
        crps[i,1] = 1
    else:
        crps[i,0] = 0.5
        crps[i,1] = 0.5


plt.figure()
sns.heatmap(np.array(crps).transpose(), cmap=palette, 
            # norm = matplotlib.colors.LogNorm(),
            cbar_kws={'label':'Continuous Ranked Probability Score'})
plt.xticks([x+0.5 for x in range(87)], df['Tech'].unique(), rotation=90)
plt.yticks([0.5,1.5],['Lafond', 'Piecewise'])


# plt.plot(np.array(mapes)[:,0], np.array(mapes)[:,1], 'o')
plt.show()











