import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, cmcrameri, analysisFunctions
import seaborn as sns
import statsmodels.api as sm

matplotlib.rc('savefig', dpi=300)

sns.set_palette([cmcrameri.cm.batlowS(x) for x in range(10)])
sns.set_palette('colorblind')
sns.set_context('talk')

matplotlib.rc('font', family='Helvetica')

df = pd.read_csv('ExpCurves.csv')

slopes = []
errors = []
lrs = []

calint = 0
valint = 0
ratio = 0

for t in df['Tech'].unique():

    # if not(analysisFunctions.sectorsinv[t] == 'Energy') or 'Nuclear' in t:
    #     continue

    x, y = np.log10(df.loc[df['Tech'] == t,'Cumulative production'].values), \
        np.log10(df.loc[df['Tech'] == t,'Unit cost'].values)

    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()

    slopes.append([result.params[1], analysisFunctions.sectorsinv[t], t])

    for i in range(len(x)-1):
        errors.append([y[i+1] - \
                    (y[i] + (-0.37274962196044487) * (x[i+1] - x[i])),t] ) 
        # errors.append([y[i+1] - \
        #             (y[i] + (-0.20723732127051694) * (x[i+1] - x[i])),t] ) 
        # errors.append([y[i+1] - \
        #             (y[i] + (-0.14238865714823595) * (x[i+1] - x[i])),t] ) 
    
    # consider first two points until the last two points
    for i in range(2, len(x)):
    # for i in range(round(len(x)/2)+1, round(len(x)/2)+2):

        x_cal, y_cal = x[:i], y[:i]

        model_cal = sm.OLS(y_cal, sm.add_constant(x_cal))
        results_cal = model_cal.fit()

        for j in range(i+1, len(x)+1):
        # for j in range(len(x), len(x)+1):
            x_val, y_val = x[i-1:j], y[i-1:j]
            model_val = sm.OLS(y_val, sm.add_constant(x_val))
            results_val = model_val.fit()

            if j == len(x):
                tot = 1
            else:
                tot = 0

            lrs.append([results_cal.params[1],
                        results_val.params[1], 
                        x_cal[-1] - x_cal[0], 
                        x_val[-1] - x_val[0], 
                        t, i/j, tot])

slopes = pd.DataFrame(slopes, columns=['Slope','Sector','Tech'])
print('Average mean slope and its standard error for the selected technologies: ')
print(slopes['Slope'].mean(), slopes['Slope'].std()/np.sqrt(slopes.shape[0]))
fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=True, sharex=True)
sns.kdeplot(slopes, hue='Sector', x='Slope',
             fill=True, alpha=0.3,
             ax=ax[0], legend=True)
# print(slopes.groupby(['Sector']).mean(numeric_only=True), 
#       slopes.groupby(['Sector']).median(numeric_only=True))
slopes = slopes.loc[~(slopes['Tech'].str.contains('Nuclear'))]
sns.kdeplot(slopes, hue='Sector', x='Slope',
             fill=True, alpha=0.3,
             ax=ax[1], legend=False)
sns.kdeplot(slopes, x='Slope',
             fill=True, alpha=0.3,
             ax=ax[1], legend=False)
sns.move_legend(ax[0], 'center left', ncol=1, bbox_to_anchor=(-1.5, 0.5))
# print(slopes.groupby(['Sector']).mean(numeric_only=True), 
#       slopes.groupby(['Sector']).median(numeric_only=True))
plt.subplots_adjust(left=0.4, bottom=0.2, right=0.95)

errors = pd.DataFrame(errors, columns=['Error','Tech'])
m, v = [], []
for t in errors['Tech'].unique():
    m.append(errors.loc[errors['Tech']==t,'Error'].mean())
    v.append(errors.loc[errors['Tech']==t,'Error'].var())
print('Average mean error and standard deviation of errors for the selected technologies using mean learning rate:' )
print(np.mean(m), np.sqrt(np.mean(v)))

# learning rate analysis

lrs = pd.DataFrame(lrs, columns=['cal','val','calint','valint','tech','pfrac','tot'])

lrs['weights'] = 1.0
for t in lrs['tech'].unique():
    lrs.loc[lrs['tech'] == t,'weights'] = 1 / \
        lrs.loc[lrs['tech'] == t].shape[0] / \
        lrs['tech'].nunique()
m = sm.WLS(lrs['val'].values, \
        sm.add_constant(lrs['cal'].values),
        weights=lrs['weights'])
r = m.fit()
# print(r.rsquared)

lrs__ = lrs.loc[lrs['tot']==1].copy()
for t in lrs__['tech'].unique():
    lrs__.loc[lrs__['tech'] == t,'weights'] = 1 / \
        lrs__.loc[lrs__['tech'] == t].shape[0] / \
        lrs__['tech'].nunique()
m = sm.WLS(lrs__['val'].values, \
        sm.add_constant(lrs__['cal'].values),
        weights=lrs__['weights'])
r = m.fit()
# print(r.rsquared)

r2xy, r2xy_p, r2xy_ma, tr2xy_n, tr2xy_p_n = [], [], [], [], []
tr2xy, tr2xy_p, tr2xy_ma = [], [], []
mu = []
minv, maxv, stepv = 0.2, 0.8, 0.005
minv, maxv, stepv = -1, 0, 0.01
minv, maxv, stepv = 0.1, 0.9, 0.005
fig, ax = plt.subplots(1,2, figsize=(10,4))
for x in np.arange(minv, maxv, stepv):

        ## interval fraction

        cals, vals, = [], []
        for t in lrs['tech'].unique():
            lrst = lrs.loc[(lrs['tech']==t)].copy()
            lrst = lrst.loc[lrst['tot']==1]
            lrst['distance'] = (lrst['calint']/(lrst['calint']+lrst['valint']) - x)**2
            # lrst['distance'] = (lrst['calint']/(lrst['calint']+lrst['valint']) - x)
            # lrst['distance'] = ([1e6*(x<0) for x in lrst['distance'].values]) + lrst['distance']
            # lrst['distance'] = ((10.0**(lrst['calint']-1))/ (10.0**(lrst['calint']+lrst['valint']-1)) - x)**2
            cals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['cal']].values[0])
            vals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['val']].values[0])
        vals = np.array(vals)
        cals = np.array(cals)
        m = sm.OLS(vals, \
                sm.add_constant(cals)
                )
        r = m.fit()
        if r.f_pvalue < 1.05:
            tr2xy.append([x, 100*r.rsquared, lrs['tech'].nunique()])
        else:
            tr2xy.append([x, np.nan, lrs['tech'].nunique()])

        if x >= 0.5 and x < 0.5+stepv:
            ax[0].scatter(cals, vals)
            ax[0].axhline(0, color='silver', alpha=0.5, zorder=-1)
            ax[0].axvline(0, color='silver', alpha=0.5, zorder=-1)
            ax[0].set_xlabel('Observed learning exponent')
            ax[0].set_ylabel('Future learning exponent')
            ax[0].set_aspect('equal')
            # ax[0].annotate('A', xy=(-1, 0.75), 
            #                xycoords='data',
            #                ha='center',
            #                va='center', fontsize=14)
            print(r.rsquared)

        cals, vals, = [], []
        for t in lrs['tech'].unique():
            if 'Nuclear' not in t:
                lrst = lrs.loc[(lrs['tech']==t)].copy()
                lrst = lrst.loc[lrst['tot']==1]
                lrst['distance'] = (lrst['calint']/(lrst['calint']+lrst['valint']) - x)**2
                # lrst['distance'] = (lrst['calint']/(lrst['calint']+lrst['valint']) - x)
                # lrst['distance'] = ([1e6*(x<0) for x in lrst['distance'].values]) + lrst['distance']
                cals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['cal']].values[0])
                vals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['val']].values[0])
        vals = np.array(vals)
        cals = np.array(cals)
        m = sm.OLS(vals, \
                sm.add_constant(cals)
                )
        r = m.fit()
        if r.f_pvalue < 1.05:
            tr2xy_n.append([x, 100*r.rsquared, lrs['tech'].nunique()])
        else:
            tr2xy_n.append([x, np.nan, lrs['tech'].nunique()])


        ## point fraction

        cals, vals, = [], []
        for t in lrs['tech'].unique():
            lrst = lrs.loc[(lrs['tech']==t)].copy()
            lrst = lrst.loc[lrst['tot']==1]
            lrst['distance'] = (lrst['pfrac'] - x)
            lrst['distance'] = ([1e6*(x<0) for x in lrst['distance'].values]) + lrst['distance']
            cals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['cal']].values[0])
            vals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['val']].values[0])
        vals = np.array(vals)
        cals = np.array(cals)
        m = sm.OLS(vals, \
                sm.add_constant(cals)
                )
        r = m.fit()
        if r.f_pvalue < 1.05:
            tr2xy_p.append([x, 100*r.rsquared, lrs['tech'].nunique()])
        else:
            tr2xy_p.append([x, np.nan, lrs['tech'].nunique()])

        cals, vals, = [], []
        for t in lrs['tech'].unique():
            if 'Nuclear' not in t:
                lrst = lrs.loc[(lrs['tech']==t)].copy()
                lrst = lrst.loc[lrst['tot']==1]
                lrst['distance'] = (lrst['pfrac'] - x)
                lrst['distance'] = ([1e6*(x<0) for x in lrst['distance'].values]) + lrst['distance']
                cals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['cal']].values[0])
                vals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['val']].values[0])
        vals = np.array(vals)
        cals = np.array(cals)
        m = sm.OLS(vals, \
                sm.add_constant(cals)
                )
        r = m.fit()
        if r.f_pvalue < 1.05:
            tr2xy_p_n.append([x, 100*r.rsquared, lrs['tech'].nunique()])
        else:
            tr2xy_p_n.append([x, np.nan, lrs['tech'].nunique()])

        # # moving average
        lrs_ = lrs.loc[(lrs['calint']/(lrs['calint']+lrs['valint']) >= x-0.05) & \
                   (lrs['calint']/(lrs['calint']+lrs['valint']) < x+0.05)
                   ].copy()
        for t in lrs_['tech'].unique():
            lrs_.loc[lrs_['tech']==t,'weights'] = 1 / \
                lrs_.loc[lrs_['tech']==t].count()[0] / \
                lrs_['tech'].nunique()
        m = sm.WLS(lrs_['val'].values, \
                sm.add_constant(lrs_['cal'].values),
                weights=lrs_['weights'])
        r = m.fit()
        if r.f_pvalue < 1.05:
            r2xy_ma.append([x, y, 100*r.rsquared, lrs_['tech'].nunique()])
        else:
            r2xy_ma.append([x, y, np.nan, lrs_['tech'].nunique()])

        lrs_ = lrs_.loc[lrs_['tot']==1]
        if lrs_.empty:
            tr2xy_ma.append([x, y, np.nan, np.nan])
            continue
        for t in lrs_['tech'].unique():
            lrs_.loc[lrs_['tech']==t,'weights'] = 1 / \
                lrs_.loc[lrs_['tech']==t].count()[0] / \
                lrs_['tech'].nunique()
        m = sm.WLS(lrs_['val'].values, \
                sm.add_constant(lrs_['cal'].values),
                weights=lrs_['weights'])
        r = m.fit()
        if r.f_pvalue < 1.05:
            tr2xy_ma.append([x, y, 100*r.rsquared, lrs_['tech'].nunique()])
        else:
            tr2xy_ma.append([x, y, np.nan, lrs_['tech'].nunique()])

r2xy = np.array(r2xy)
r2xy_ma = np.array(r2xy_ma)
r2xy_p = np.array(r2xy_p)
tr2xy = np.array(tr2xy)
tr2xy_ma = np.array(tr2xy_ma)
tr2xy_p = np.array(tr2xy_p)
tr2xy_p_n = np.array(tr2xy_p_n)
tr2xy_n = np.array(tr2xy_n)

# ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
#          r2xy[:,2],
#          label='All combinations - closest point for each technology')
# ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
#          r2xy_ma[:,2],
#          label='All combinations - moving average')
# ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
#          r2xy_p[:,2],
#          label='All combinations - points')
ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy[:,1],
         label='Logarithmic cumulative production range')
ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_n[:,1],
         label='Logarithmic cumulative production range (no nuclear)')
print(0.5, np.where((tr2xy[:,0]>=0.5)*(tr2xy[:,0]<0.5+stepv))[0][0])
ax[1].scatter(0.5,tr2xy[np.where((tr2xy[:,0]>=0.5)*(tr2xy[:,0]<0.5+stepv))[0][0],1], 
              color='k', marker='o', s=50)
# ax[1].annotate('A', xy=(0.5, 1.2*tr2xy[np.where((tr2xy[:,0]>=0.5)*(tr2xy[:,0]<0.5+stepv))[0][0],2]), 
#                ha='center', va='bottom',
#                xycoords='data', fontsize=14)
# ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
#          tr2xy_ma[:,2],
#          label='Full range - moving average')
ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_p[:,1],
         label='Number of data points')
ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_p_n[:,1],
         label='Number of data points (no nuclear)')
ax[1].set_ylim(0,100)
ax[1].set_xlabel('Fraction of data treated as observed')
ax[1].set_ylabel('Percentage of explained variance [%]')
ax[1].legend(loc='upper right')
# ax[1].set_xscale('log', base=10)
fig.subplots_adjust(wspace=0.3, hspace=0.3, 
                    left=0.075, right=0.925,
                    top=0.975, bottom=0.125)

plt.show()

fig, ax = plt.subplots()
lrs['valint/calint'] = lrs['valint'] / lrs['calint']
lrs['diff'] = 10**((lrs['val'] - lrs['cal']) * lrs['valint'])
lrs['diff2'] = 10**((lrs['val'] - np.mean(slopes)) * lrs['valint'])
ax.scatter(lrs['valint/calint'], lrs['diff'], alpha=0.3)
ax.scatter(lrs['valint/calint'], lrs['diff2'], alpha=0.3)
print(lrs['diff'].min())
# sns.kdeplot(lrs, x='valint/calint', y='diff', 
#             weights='weights', fill=False, levels=5,
#             ax=ax, alpha=0.3)
# sns.kdeplot(lrs, x='valint/calint', y='diff2', 
#             weights='weights', fill=False, levels=5,
#             ax=ax, alpha=0.3)
# ax.set_yscale('log', base=10)
# ax.set_ylim(1e-200,1e2)
plt.show()
