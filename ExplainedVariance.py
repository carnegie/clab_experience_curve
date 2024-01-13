import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, cmcrameri, os, analysisFunctions
import seaborn as sns
import statsmodels.api as sm

matplotlib.rc('savefig', dpi=300)
sns.set_palette([cmcrameri.cm.batlowS(x) for x in range(10)])
sns.set_palette('colorblind')
sns.set_context('talk')
sns.set_style('whitegrid')
matplotlib.rc('font', family='Helvetica')

# read data
df = pd.read_csv('ExpCurves.csv')

# empty list to store learning rate data
lrs = []

# iterate over technologies
for t in df['Tech'].unique():

    # read technology cumulative production and unit cost
    x, y = np.log10(\
            df.loc[\
                df['Tech']==t,'Cumulative production'].values), \
            np.log10(\
                df.loc[df['Tech']==t,'Unit cost'].values)

    # iterate over all the points
    # consider first two points until the last two points
    for i in range(2, len(x)):

        # get data to be treated as observed
        x_cal, y_cal = x[:i], y[:i]

        # build regression model and fit it to data
        model_cal = sm.OLS(y_cal, sm.add_constant(x_cal))
        results_cal = model_cal.fit()

        # explore all combinations of future data
        for j in range(i+1, len(x)+1):

            # build validation dataset
            x_val, y_val = x[i-1:j], y[i-1:j]

            # build linear regression model and fit it to data
            model_val = sm.OLS(y_val, sm.add_constant(x_val))
            results_val = model_val.fit()

            # total flag 
            if j == len(x):
                tot = 1
            else:
                tot = 0

            # store data
            lrs.append([results_cal.params[1],
                        results_val.params[1], 
                        x_cal[-1] - x_cal[0], 
                        x_val[-1] - x_val[0], 
                        t, i/j, tot])

# learning rate analysis
lrs = pd.DataFrame(lrs, columns=['cal',
                                 'val',
                                 'calint',
                                 'valint',
                                 'tech',
                                 'pfrac',
                                 'tot'])

# define weights
lrs['weights'] = 1.0
for t in lrs['tech'].unique():
    lrs.loc[lrs['tech'] == t,'weights'] = 1 / \
        lrs.loc[lrs['tech'] == t].shape[0] / \
        lrs['tech'].nunique()

# build weighted linear regression model 
# fit it to data
# and print fraction of explained variance
m = sm.WLS(lrs['val'].values, \
        sm.add_constant(lrs['cal'].values),
        weights=lrs['weights'])
r = m.fit()
# print(100*r.rsquared)

# select only where observed + future covers total range
lrs__ = lrs.loc[lrs['tot']==1].copy()

#recompute weights
for t in lrs__['tech'].unique():
    lrs__.loc[lrs__['tech'] == t,'weights'] = 1 / \
        lrs__.loc[lrs__['tech'] == t].shape[0] / \
        lrs__['tech'].nunique()
    
# build weighted linear regression method
# fit it to data and print fraction of explained variance
m = sm.WLS(lrs__['val'].values, \
        sm.add_constant(lrs__['cal'].values),
        weights=lrs__['weights'])
r = m.fit()
# print(100*r.rsquared)

# create lists to store data
tr2xy_i, tr2xy_p, tr2xy_i_n, tr2xy_p_n = [], [], [], []
r2xy_ma, tr2xy_ma, r2xy_ma_n, tr2xy_ma_n =  [], [], [] ,[]

# define minimum, maximum and step 
# for fraction of data treated as observed
minv, maxv, stepv = 0.1, 0.9, 0.005

# create figure
fig, ax = plt.subplots(1,2, figsize=(12,5))

# iterate over all subintervals
for x in np.arange(minv, maxv, stepv):
    
    ### method based on range 
    ### of the logarithm of cumulative production 
    x = round(x, 3)
    # create lists to store data
    cals, vals, = [], []
    cals_n, vals_n = [], []

    # iterate over technologies
    for t in lrs['tech'].unique():
        
        # select only data from technologies
        # and covering full interval
        lrst = lrs.loc[\
                (lrs['tech']==t) & (lrs['tot']==1)
                    ].copy()
        
        # compute distance from the examined
        # fraction of data treated as observed
        lrst['distance'] = \
            (x - lrst['calint'] / 
                (lrst['calint']+lrst['valint'])
                    )**2
        
        # append data to lists where distance is minimum
        cals.append(\
            lrst.loc[lrst['distance'] == \
                lrst['distance'].min(),['cal']]\
                    .values[-1])
        vals.append(lrst.loc[\
            lrst['distance'] == \
                lrst['distance'].min(),['val']]\
                    .values[-1])
        
        # append data to lists where distance is minimum
        # for lists excluding nuclear
        if 'Nuclear' not in t:
            cals_n.append(\
                lrst.loc[lrst['distance'] == \
                            lrst['distance'].min(),['cal']]\
                            .values[-1])
            vals_n.append(\
                lrst.loc[lrst['distance'] == \
                            lrst['distance'].min(),['val']]\
                            .values[-1])
    
    # transform lists into arrays
    cals = np.array(cals)
    vals = np.array(vals)
    cals_n = np.array(cals_n)
    vals_n = np.array(vals_n)
    
    # build linear regression model and store r squared
    m = sm.OLS(vals, \
            sm.add_constant(cals)
            )
    r = m.fit()
    tr2xy_i.append([x, 100*r.rsquared])

    # build linear regression model and store r squared
    m = sm.OLS(vals_n, \
            sm.add_constant(cals_n)
            )
    r = m.fit()
    tr2xy_i_n.append([x, 100*r.rsquared])


    ### method based on number of points

    # create lists to store data
    cals, vals, = [], []
    cals_n, vals_n = [], []

    # iterate over technologies
    for t in lrs['tech'].unique():

        # select only data from technologies
        # and covering full interval
        lrst = lrs.loc[\
            (lrs['tech']==t) & (lrs['tot']==1)
                ].copy()
        
        # compute distance from the examined
        # fraction of data treated as observed
        lrst['distance'] = (x - lrst['pfrac'])**2

        # append data to lists where distance is minimum
        cals.append(\
            lrst.loc[lrst['distance']==\
                        lrst['distance'].min(),['cal']]\
                        .values[-1])

        vals.append(
            lrst.loc[lrst['distance']==\
                        lrst['distance'].min(),['val']]\
                        .values[-1])
        
        # append data to lists where distance is minimum
        # for lists excluding nuclear
        if 'Nuclear' not in t:
            cals_n.append(\
                lrst.loc[lrst['distance']==\
                            lrst['distance'].min(),['cal']]\
                            .values[-1])
            vals_n.append(\
                lrst.loc[lrst['distance']==\
                            lrst['distance'].min(),['val']]\
                            .values[-1])
    
    # transform lists into arrays
    cals = np.array(cals)
    vals = np.array(vals)
    cals_n = np.array(cals_n)
    vals_n = np.array(vals_n)

    # build linear regression model
    # fit it to data and append r squared data to list
    m = sm.OLS(vals, \
            sm.add_constant(cals)
            )
    r = m.fit()
    tr2xy_p.append([x, 100*r.rsquared])

    # build linear regression model for nuclear excluded
    # fit it to data and append r squared data to list
    m = sm.OLS(vals_n, \
            sm.add_constant(cals_n)
            )
    r = m.fit()
    tr2xy_p_n.append([x, 100*r.rsquared])

    # plot scatter when at least half of the points are observed
    if x == 0.5:
        ax[0].scatter(cals, vals)
        ax[0].axhline(0, color='silver', 
                        alpha=0.5, zorder=-1)
        ax[0].axvline(0, color='silver', 
                        alpha=0.5, zorder=-1)
        ax[0].set_xlabel('Observed learning exponent')
        ax[0].set_ylabel('Future learning exponent')
        ax[0].set_aspect('equal')
        # ax[0].annotate('A', xy=(-1, 0.75), 
        #                 xycoords='data',
        #                 ha='center',
        #                 va='center', fontsize=14)

    ### moving average method

    # select data in the interval
    lrs_ = lrs.loc[\
        (lrs['calint']/\
            (lrs['calint']+lrs['valint']) >= x-0.05) & \
        (lrs['calint']/\
            (lrs['calint']+lrs['valint']) < x+0.05)
                ].copy()
    
    # compute weights
    for t in lrs_['tech'].unique():
        lrs_.loc[lrs_['tech']==t,'weights'] = 1 / \
            lrs_.loc[lrs_['tech']==t].shape[0] / \
            lrs_['tech'].nunique()
        
    # build weighted linear regression model
    # fit it to data    
    # append data to list
    m = sm.WLS(lrs_['val'].values, \
            sm.add_constant(lrs_['cal'].values),
            weights=lrs_['weights'])
    r = m.fit()
    r2xy_ma.append([x, 100*r.rsquared])

    # remove nuclear data
    lrs__ = lrs_.loc[\
        ~(lrs_['tech'].str.contains('Nuclear'))].copy()

    # compute weights
    for t in lrs__['tech'].unique():
        lrs__.loc[lrs__['tech']==t,'weights'] = 1 / \
            lrs__.loc[lrs__['tech']==t].shape[0] / \
            lrs__['tech'].nunique()

    # build weighted linear regression model
    # fit it to data    
    # append data to list
    m = sm.WLS(lrs__['val'].values, \
            sm.add_constant(lrs__['cal'].values),
            weights=lrs__['weights'])
    r = m.fit()
    r2xy_ma_n.append([x, 100*r.rsquared])

    # select only data covering the full interval
    lrs_ = lrs_.loc[lrs_['tot']==1]

    # if empty skip
    if lrs_.empty:
        tr2xy_ma.append([x, np.nan])
        continue

    # compute weights
    for t in lrs_['tech'].unique():
        lrs_.loc[lrs_['tech']==t,'weights'] = 1 / \
            lrs_.loc[lrs_['tech']==t].shape[0] / \
            lrs_['tech'].nunique()
        
    # build weighted linear regression model
    # fit it to data    
    # append data to list
    m = sm.WLS(lrs_['val'].values, \
            sm.add_constant(lrs_['cal'].values),
            weights=lrs_['weights'])
    r = m.fit()
    tr2xy_ma.append([x, 100*r.rsquared])

    # remove nuclear data
    lrs__ = lrs_.loc[\
        ~(lrs_['tech'].str.contains('Nuclear'))].copy()
    
    # compute weights
    for t in lrs__['tech'].unique():
        lrs__.loc[lrs__['tech']==t,'weights'] = 1 / \
            lrs__.loc[lrs__['tech']==t].shape[0] / \
            lrs__['tech'].nunique()
    
    # build weighted linear regression model
    # fit it to data
    # append data to list
    m = sm.WLS(lrs__['val'].values, \
            sm.add_constant(lrs__['cal'].values),
            weights=lrs__['weights'])
    r = m.fit()
    tr2xy_ma_n.append([x, 100*r.rsquared])

# convert lists to arrays
tr2xy_i = np.array(tr2xy_i)
tr2xy_i_n = np.array(tr2xy_i_n)
tr2xy_p = np.array(tr2xy_p)
tr2xy_p_n = np.array(tr2xy_p_n)
r2xy_ma = np.array(r2xy_ma)
tr2xy_ma = np.array(tr2xy_ma)
r2xy_ma_n = np.array(r2xy_ma_n)
tr2xy_ma_n = np.array(tr2xy_ma_n)

# plot fraction of explained variance
# vs fraction of points treated as observed
ax[1].scatter(0.5, tr2xy_p\
              [np.where(
                  (tr2xy_p[:,0]>=0.5) * \
                    (tr2xy_p[:,0]<0.5+stepv)
                    )[0][0],1], 
              color='k', marker='o', s=50)
ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_p[:,1],
         label='All technologies')
ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_p_n[:,1],
         label='Nuclear excluded')

# set axes limits and labels, add legend
ax[1].set_ylim(0,100)
ax[1].set_xlabel('Fraction of data points treated as observed')
ax[1].set_ylabel('Percentage of explained variance [%]')
ax[1].legend(loc='upper right')

ax[0].annotate('a', xy=(0.05, 1.05),
                xycoords='axes fraction',
                ha='center', va='center')
ax[1].annotate('b', xy=(0.05, 1.05),
                xycoords='axes fraction',
                ha='center', va='center')
fig.subplots_adjust(wspace=0.3, hspace=0.3, 
                    left=0.1, right=0.95,
                    top=0.925, bottom=0.15)

if not(os.path.exists('figs' + os.path.sep + 'explainedVariance')):
    os.makedirs('figs' + os.path.sep + 'explainedVariance')

fig.savefig('figs' + os.path.sep + 'explainedVariance' + \
            os.path.sep + 'explainedVariance.png')

# plot explained variance vs fraction of data treated as observed
# using cumulative production range, points and moving average

# create figure
fig, ax = plt.subplots(1,1, figsize=(9,6))

# plot all the different methods
l = ax.plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_i[:,1],
         label='Logarithmic cumulative production range')
ax.plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_i_n[:,1], 
         color = l[0].get_color(), linestyle=':',
         label='Logarithmic cumulative' +\
             ' production range (no nuclear)')
l = ax.plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_p[:,1],
         label='Fraction of points')
ax.plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_p_n[:,1],
         color = l[0].get_color(), linestyle=':',
         label='Fraction of points ' + \
            ' (no nuclear)')

l = ax.plot([x for x in np.arange(minv, maxv, stepv)], 
         r2xy_ma[:,1],
         label='All combinations - moving average')
ax.plot([x for x in np.arange(minv, maxv, stepv)], 
         r2xy_ma_n[:,1],
         color = l[0].get_color(), linestyle=':',
         label='All combinations - moving average - nuclear excluded')
l = ax.plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_ma[:,1],
         label='Full range - moving average')
ax.plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_ma_n[:,1],
         color = l[0].get_color(), linestyle=':',
         label='Full range - moving average - nuclear excluded')
# set axes limits and labels, add legend
ax.set_ylim(0,100)
ax.set_xlabel('Fraction of data treated as observed')
ax.set_ylabel('Percentage of explained variance [%]')
ax.legend(loc='upper right')

fig.subplots_adjust(top=0.95, bottom=0.15,
                    left=0.15, right=0.9)

if not(os.path.exists('figs' + os.path.sep + 'SupplementaryFigures')):
    os.makedirs('figs' + os.path.sep + 'SupplementaryFigures')

fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
            os.path.sep + 'ExplainedVariance.png')

plt.show()
