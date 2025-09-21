import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
import scipy, matplotlib, utils, os
import cmcrameri as cm
import utils

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
breaks_dist = scipy.stats.expon(
    loc = params_breaks_lexp.loc[
        params_breaks_lexp['Variable'] == 'breaks', 'Loc'].values[0],
    scale = params_breaks_lexp.loc[
        params_breaks_lexp['Variable'] == 'breaks', 'Scale'].values[0])
lexp_dist = scipy.stats.norm(
    loc = params_breaks_lexp.loc[
        params_breaks_lexp['Variable'] == 'lexp', 'Loc'].values[0],
    scale = params_breaks_lexp.loc[
        params_breaks_lexp['Variable'] == 'lexp', 'Scale'].values[0])

# compare piecewise linear experience curves with 
# first difference wrights law using half of points for all techs
crpss, logs, ttests_crps, ttests_logs = [], [], [], []
plotFigTech = False
nsim = 1000

# iterate for each technology
for t in df['Tech'].unique():
    print(t)
    # remove prespecified sector if available
    if remove_sector is not None:
        if utils.sectorsinv[t] == remove_sector:
            continue

    # get cost and production data for technology
    x = df[df['Tech'] == t]['Cumulative production'].values
    y = df[df['Tech'] == t]['Unit cost'].values
    time = df[df['Tech'] == t]['Year'].values

    # plot data
    if plotFigTech:
        plt.figure(figsize=(10,6))
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(x,y, label='Data')

    # LINEAR REGRESSION MODEL - calibration period
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
    y_diff_cal_pred = 10**(np.log10(y[:round(x.shape[0]/2)-1]) 
                           + y_diff_cal_pred)
    y_diff_cal_pred = np.insert(y_diff_cal_pred,0,y[0])


    ## MOORE'S MODEL
    y_cal = np.log10(y[:round(y.shape[0]/2)])
    time_cal = time[:round(y.shape[0]/2)]

    # build linear regression model, fit it, and save progress exponent
    if all(np.diff(time_cal)) == 1:
        moore_lexp = np.mean(y_cal[1:] - y_cal[:-1])
        moore_noise = np.std( y_cal[1:] - y_cal[:-1]
                             - moore_lexp )
    else:
        time_diff = np.diff(time_cal)
        moore_lexp = np.mean((y_cal[1:] - y_cal[:-1]) / time_diff)
        moore_noise = np.std( (y_cal[1:] - y_cal[:-1]) / time_diff
                             - moore_lexp )

    # PIECEWISE REGRESSION MODEL - calibration period
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

    ## FORECAST COMPARISON

    # use the remaining points for prediction and error comparison
    x_val = np.log10(x[round(x.shape[0]/2)-1:])
    y_val = np.log10(y[round(y.shape[0]/2)-1:])
    time_val = time[round(y.shape[0]/2)-1:]

    # FIRST DIFFERENCE LINEAR REGRESSON MODEL FORECAST
    # forecast using first difference wright's law
    x_diff_val = np.diff(x_val)

    y_diff_val_sim = np.zeros((nsim, x_diff_val.shape[0] + 1))
    for n in range(nsim):
        y_diff_val_pred = np.array([np.log10(y[round(y.shape[0]/2)-1])])
        noise = 0
        for i in range(x_diff_val.shape[0]):
            noise = 0.19 * noise + np.random.randn() * noise_std
            y_diff_val_pred = np.append(y_diff_val_pred,
                                        y_diff_val_pred[-1] 
                                        + lexp * x_diff_val[i] 
                                        + noise)
        y_diff_val_pred = 10**(y_diff_val_pred)
        y_diff_val_sim[n,:] = y_diff_val_pred

    # compute continuous rank probability score and log scoring
    crps_wright_ = np.zeros(y_diff_val_sim.shape[1])
    logs_wright_ = np.zeros(y_diff_val_sim.shape[1])
    # iterate over observations in the validation period
    for i in range(y_diff_val_sim.shape[1]):
        # get the remaining observations
        obs = y[round(y.shape[0]/2)-1:]
        # get the prediction for the i-th element of the validation period
        preds = y_diff_val_sim[:,i]
        crps_wright_[i] = utils.compute_crps(preds, obs[i])
        logs_wright_[i] = utils.compute_logscore(preds, obs[i])



    if plotFigTech:
        plt.plot(x[round(x.shape[0]/2)-1:], 
                 np.median(y_diff_val_sim, axis=0), 
                 color='red',
                 label='Lafond - median')
        plt.fill_between(x[round(x.shape[0]/2)-1:], 
                         np.percentile(y_diff_val_sim, 95, axis=0), 
                         np.percentile(y_diff_val_sim, 5, axis=0), 
                         alpha=0.2, color='red',
                         label='Lafond - 5-95%')


    # MOORE'S MODEL FORECAST
    time_diff_val = np.diff(time_val)

    y_moore_val_sim = np.zeros((nsim, time_diff_val.shape[0] + 1))
    for n in range(nsim):
        y_moore_val_pred = np.array([np.log10(y[round(y.shape[0]/2)-1])])
        for i in range(time_diff_val.shape[0]):
            noise = np.random.randn() * moore_noise
            y_moore_val_pred = np.append(y_moore_val_pred,
                                        y_moore_val_pred[-1] 
                                        + moore_lexp
                                        + noise)
        y_moore_val_pred = 10**(y_moore_val_pred)
        y_moore_val_sim[n,:] = y_moore_val_pred    

    # compute continuous rank probability score and log score
    crps_moore_ = np.zeros(y_moore_val_sim.shape[1])
    logs_moore_ = np.zeros(y_moore_val_sim.shape[1])
    for i in range(y_moore_val_sim.shape[1]):
        obs = y[round(y.shape[0]/2)-1:]
        preds = y_moore_val_sim[:,i]
        crps_moore_[i] = utils.compute_crps(preds, obs[i])
        logs_moore_[i] = utils.compute_logscore(preds, obs[i])

    # PIECEWISE LINEAR REGRESSION FORECAST
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
            breaks = np.append(breaks, 
                               max(x_cal[-1], 
                                   breaks[-1] + min(np.log10(2), 
                                                    breaks_dist.rvs())))
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
    
    # compute continuous rank probability score and log score
    crps_piecewise_ = np.zeros(y_val_sim.shape[1])
    logs_piecewise_ = np.zeros(y_val_sim.shape[1])
    for i in range(y_val_sim.shape[1]):
        obs = y[round(y.shape[0]/2)-1:]
        preds = y_val_sim[:,i]
        crps_piecewise_[i] = utils.compute_crps(preds, obs[i])
        logs_piecewise_[i] = utils.compute_logscore(preds, obs[i])


    # perform t test to check significance of difference on average
    ttests_crps.append(scipy.stats.ttest_rel(crps_wright_, 
                                             crps_piecewise_)[1])
    ttests_logs.append(scipy.stats.ttest_rel(logs_wright_, 
                                             logs_piecewise_)[1])

    # store results
    crps_piecewise = np.mean(crps_piecewise_)
    crps_wright = np.mean(crps_wright_)
    crps_moore = np.mean(crps_moore_)
    crpss.append([crps_wright, crps_moore, crps_piecewise])

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
        os.makedirs('figs/SupplementaryFigures/Techs_comparison/', 
                    exist_ok=True)
        plt.savefig('figs/SupplementaryFigures/Techs_comparison/'+t+'.png')
        plt.close('all')

# save results to dataframe
result = pd.DataFrame(np.concatenate([crpss, 
                                      np.array(ttests_crps).reshape(-1,1), 
                                      ],
                                      axis=1),
                      columns=['CRPS - Constant',
                               'CRPS - Moore',
                               'CRPS - Variable',
                               'CRPS - p-value',
                               ])

print("Number of data series for which piecewise is better:")
print(result.loc[(result["CRPS - Variable"] < result['CRPS - Constant'])
                 & (result["CRPS - Variable"] < result['CRPS - Moore'])].shape)
print("Number of data series for which Wright's model is better:")
print(result.loc[(result["CRPS - Constant"] < result['CRPS - Variable'])
                 & (result["CRPS - Constant"] < result['CRPS - Moore'])].shape)
print("Number of data series for which Moore's model is better:")
print(result.loc[(result["CRPS - Moore"] < result['CRPS - Variable'])
                 & (result["CRPS - Moore"] < result['CRPS - Constant'])].shape)
print("Number of data series for which Piecewise is better than Moore's model:")
print(result.loc[(result["CRPS - Variable"] < result['CRPS - Moore'])].shape)

# print the results
print('Piecewise is better in CRPS:')
print(result.loc[(result['CRPS - p-value']<0.05) & \
                 (result['CRPS - Constant']>result['CRPS - Variable'])
                 ].count()[0])
print('No difference:')
print(result.loc[(result['CRPS - p-value']>0.05)].count()[0])

# report results when difference is significiant
result['Tech'] = df['Tech'].unique()
result['Sector'] = [utils.sectorsinv[s] 
                    for s in result['Tech'].values]
result['CRPS - Piecewise'] = [1.0*(x[0] < x[1])*(x[2] < 0.05) 
                              + 0.5*(x[2]>0.05) 
                              for x in result[
                                  ['CRPS - Variable',
                                   'CRPS - Constant',
                                   'CRPS - p-value']].values]
result['CRPS - Linear'] = [x[0] > x[1] 
                           for x in result[
                               ['CRPS - Variable',
                                'CRPS - Constant']].values]

## plotting the results
for sec in result['Sector'].unique():
    sel = result.loc[result['Sector']==sec]
    fig, ax = plt.subplots(figsize=(16,8))
    im = ax.imshow(np.transpose(sel[['CRPS - Piecewise']].values), 
              cmap=cm.cm.batlow, 
              norm=matplotlib.colors.Normalize(-0.1,1.4),
              alpha=.8,
              aspect=2
            ) 
    ax.scatter(range(sel.shape[0]), np.zeros(sel.shape[0]),
               s=50.0*(sel['CRPS - p-value']<0.05), color='k', marker='d')
    ax.axhline(.5, lw=.5, color='k')
    [ax.axvline(x+.5, lw=.5, color='k') for x in range(sel.shape[0])]
    ax.set_xticks([x for x in range(sel.shape[0])],
                    [s.replace('_',' ') for s in sel['Tech'].values], 
                    rotation=90) 
    ax.set_yticks([])
    ax.set_position([0.1, 0.7, sel.shape[0]/50*0.8, 0.3])
    plt.title(sec)
    plt.savefig('figs' + os.path.sep + sec + '_CRPS.pdf')

# prepare the legend for the mulitple panels
fig, ax = plt.subplots(figsize=(16,8))
im = ax.imshow([[0],[1],[0.5]], 
            cmap=cm.cm.batlow, 
            norm=matplotlib.colors.Normalize(-0.1,1.4),
            alpha=.8,
            aspect=2
        ) 
ax.scatter([0,0],[0,1], s=50.0, color='k', marker='d')
ax.axhline(.5, lw=.5, color='k')
ax.set_xticks([])
ax.yaxis.tick_right()
ax.set_yticks([0,1,2],
              ['First difference Wright\'s law has significantly lower error',
               'Piecewise experience curve has significantly lower error',
               'No significant difference in error between the forecasts'])
ax.set_position([0.1, 0.7, 1/50*0.8, 0.3])

### plot best, second best and worst
models = ['Constant', 'Moore', 'Variable']
modelmapping = {'Constant': "First difference Wright's Law",
                'Variable': 'Piecewise linear experience curve',
                'Moore': "Moore's law",
                }
best = pd.DataFrame()
for model in models:
    others = [m for m in models if not(m==model)]
    b = result.loc[
        (result['CRPS - ' + model] < result['CRPS - ' + others[0]])
        & (result['CRPS - ' + model] < result['CRPS - ' + others[1]])
    ].shape[0]
    w = result.loc[
        (result['CRPS - ' + model] > result['CRPS - ' + others[0]])
        & (result['CRPS - ' + model] > result['CRPS - ' + others[1]])
    ].shape[0]
    new_row = pd.DataFrame([[modelmapping[model], 
                                b, 
                                result['Tech'].nunique() - b - w,
                                w]],
                                columns = ['Model',
                                        'Best',
                                        'Second best',
                                        'Third best'])
    best = pd.concat([best, new_row])

df_melted = best.melt(id_vars='Model', 
                    value_vars=['Best', 'Second best', 'Third best'],
                    var_name='Rank', value_name='Number of technologies')


custom_palette = {
    model: color
    for model, color in zip(df_melted['Model'].unique(), [
        cm.cm.batlow(30),
        cm.cm.batlow(140),
        cm.cm.batlow(180),
        # Add more if needed depending on number of models
    ])
}

fig, ax = plt.subplots(figsize=(9,7))

sns.barplot(df_melted, 
            x='Rank',
            y='Number of technologies',
            hue = 'Model',
            palette=custom_palette)
ax.set_xlabel('')

plt.subplots_adjust(bottom=0.35, top=0.975)


ax.legend(title='Model',
          bbox_to_anchor=(0.5, -0.125),  # Centered below the plot
          loc='upper center',)


# plt.tight_layout()
plt.show()