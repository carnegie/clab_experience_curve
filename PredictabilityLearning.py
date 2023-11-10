import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import statsmodels.api as sm
import cmcrameri, scipy

matplotlib.rc('savefig', dpi=300)

sns.set_palette([cmcrameri.cm.batlowS(x) for x in range(10)])
sns.set_palette('colorblind')
sns.set_context('paper')

matplotlib.rc('font', family='Helvetica')

df = pd.read_csv('ExpCurves.csv')

slopes = []
errors = []
lrs = []

# fig, ax = plt.subplots(1,2, figsize=(10,5), sharey=True, sharex=True)
calint = 0
valint = 0
ratio = 0
slp = []
for t in df['Tech'].unique():
    x, y = np.log10(df.loc[df['Tech'] == t,'Cumulative production'].values), \
        np.log10(df.loc[df['Tech'] == t,'Unit cost'].values)


    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()
    slopes.append(result.params[1])

    for i in range(len(x)-1):
        errors.append([y[i+1] - \
                      (y[i] + (-0.37274962196044487) * (x[i+1] - x[i])),t] ) 
    
    # print(10**(x[1] - x[0] - 1) / 10**(x[-1] - x[0] - 1), result.f_pvalue, t)
    
    # plt.figure()
    # plt.scatter(10**x,10**y)
    # plt.show()

    if result.f_pvalue > 1.5:
        continue

    slps = [0]
    dxs = []
    for i in range(1, len(x)):
        slps.append((y[i] - y[i-1]) / (x[i] - x[i-1]))
        dxs.append(x[i] - x[i-1])


    slps = slps[1:]

    # acf = []
    # mu = np.mean(slps)
    # var = np.var(slps)
    # bins = np.linspace(0, 5, 6)
    # for b in range(len(bins)-1):
    #     pairs = []
    #     for i in range(1,len(x)):
    #         for j in range(i+1, len(x)):
    #             if x[j] - x[i] > bins[b] and x[j]-x[i] < bins[b+1]:
    #                 # print(x[j]-x[i], bins[b], bins[b+1])
    #                 pairs.append([slps[i-1], slps[j-1]])
    #     acf.append((np.mean([(a[0] - mu) * (a[1] - mu) for a in pairs])/var))
    # for i in range(1,len(x)):
    #     for j in range(i+1, len(x)):
    #         acf.append([x[j] - x[i] ,((slps[i-1] - mu) * (slps[j-1] - mu))/var])
    # acf = np.array(acf)
    # figacf, axacf = plt.subplots()

    # axacf.scatter(acf[:,0], acf[:,1], alpha=0.3, zorder=-10)
    # plt.show()
    
   
    # acf2 = []
    # mu = np.mean(acf)
    # var = np.var(acf)



    # axacf.plot(bins[1:],acf, alpha=0.2)
    # plt.show()
    # for a,b in zip(slps, 1+x-x[0]):
    #     slp.append([a,b])
    # for a,b in zip(slps[1:], dxs):
    #     slp.append([a,b,t])

    for i in range(2, len(x)):
    # for i in range(round(len(x)/2)+1, round(len(x)/2)+2):

        x_cal, y_cal = x[:i], y[:i]

        model_cal = sm.OLS(y_cal, sm.add_constant(x_cal))
        results_cal = model_cal.fit()

        if results_cal.f_pvalue < 1.05:

            for j in range(i+1, len(x)+1):
            # for j in range(len(x), len(x)+1):
                x_val, y_val = x[i-1:j], y[i-1:j]
                model_val = sm.OLS(y_val, sm.add_constant(x_val))
                results_val = model_val.fit()

                # if (10**x_val[-1] - 10**x_val[0])/(10**x_cal[-1] - 10**x_cal[0]) < ratio :
                # if x_cal[-1] - x_cal[0] > calint and x_val[-1] - x_val[0] > valint:
                if j == len(x):
                    tot = 1
                else:
                    tot = 0

                if results_val.f_pvalue < 1.05:
                    lrs.append([results_cal.params[1],
                                results_val.params[1], 
                                x_cal[-1] - x_cal[0], 
                                x_val[-1] - x_val[0], 
                                t, i/j, tot])

print(scipy.stats.normaltest(slopes))
print(scipy.stats.shapiro(slopes))
print(scipy.stats.anderson(slopes, 'norm'))
sns.displot(slopes, kde=True)
print(np.mean(slopes), np.std(slopes))
errors = pd.DataFrame(errors, columns=['Error','Tech'])
m, v = [], []
for t in errors['Tech'].unique():
    m.append(errors.loc[errors['Tech']==t,'Error'].mean())
    v.append(errors.loc[errors['Tech']==t,'Error'].var())
print(np.mean(m), np.sqrt(np.mean(v)))
# print(np.mean(errors), np.std(errors))
plt.show()

lrs = pd.DataFrame(lrs, columns=['cal','val','calint','valint','tech','pfrac','tot'])

lrs['weights'] = 1
for t in lrs['tech'].unique():
    lrs.loc[lrs['tech'] == t,'weights'] = 1 / \
        lrs.loc[lrs['tech'] == t].count()[0] / \
        lrs['tech'].nunique()
m = sm.WLS(lrs['val'].values, \
        sm.add_constant(lrs['cal'].values),
        weights=lrs['weights'])
r = m.fit()
print(r.rsquared)

lrs__ = lrs.loc[lrs['tot']==1].copy()
for t in lrs__['tech'].unique():
    lrs__.loc[lrs__['tech'] == t,'weights'] = 1 / \
        lrs__.loc[lrs__['tech'] == t].count()[0] / \
        lrs__['tech'].nunique()
m = sm.WLS(lrs__['val'].values, \
        sm.add_constant(lrs__['cal'].values),
        weights=lrs__['weights'])
r = m.fit()
print(r.rsquared)

r2xy, r2xy_p, r2xy_ma, tr2xy_n, tr2xy_p_n = [], [], [], [], []
tr2xy, tr2xy_p, tr2xy_ma = [], [], []
mu = []
minv, maxv, stepv = 0.2, 0.8, 0.005
minv, maxv, stepv = -1, 0, 0.01
minv, maxv, stepv = 0.1, 0.9, 0.005
fig, ax = plt.subplots(1,2, figsize=(10,4))
for x in np.arange(minv, maxv, stepv):

        ## interval fraction
        # cals, vals, = [], []
        # for t in lrs['tech'].unique():
        #     lrst = lrs.loc[(lrs['tech']==t)].copy()
        #     # lrst['distance'] = (lrst['calint']/(lrst['calint']+lrst['valint']) - x)**2
        #     lrst['distance'] = (lrst['calint']/(lrst['calint']+lrst['valint']) - x)
        #     lrst['distance'] = ([1e6*(x<0) for x in lrst['distance'].values]) + lrst['distance']
        #     # lrst['distance'] = ((10.0**(lrst['calint']-1))/ (10.0**(lrst['calint']+lrst['valint']-1)) - x)**2
        #     cals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['cal']].values[0])
        #     vals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['val']].values[0])
        # vals = np.array(vals)
        # cals = np.array(cals)
        # m = sm.OLS(vals, \
        #         sm.add_constant(cals)
        #         )
        # r = m.fit()
        # if r.f_pvalue < 1.05:
        #     r2xy.append([x, y, 100*r.rsquared, lrs['tech'].nunique()])
        # else:
        #     r2xy.append([x, y, np.nan, lrs['tech'].nunique()])

        cals, vals, = [], []
        for t in lrs['tech'].unique():
            lrst = lrs.loc[(lrs['tech']==t)].copy()
            lrst = lrst.loc[lrst['tot']==1]
            # lrst['distance'] = (lrst['calint']/(lrst['calint']+lrst['valint']) - x)**2
            lrst['distance'] = (lrst['calint']/(lrst['calint']+lrst['valint']) - x)
            lrst['distance'] = ([1e6*(x<0) for x in lrst['distance'].values]) + lrst['distance']
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
            tr2xy.append([x, y, 100*r.rsquared, lrs['tech'].nunique()])
        else:
            tr2xy.append([x, y, np.nan, lrs['tech'].nunique()])

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
                # lrst['distance'] = (lrst['calint']/(lrst['calint']+lrst['valint']) - x)**2
                lrst['distance'] = (lrst['calint']/(lrst['calint']+lrst['valint']) - x)
                lrst['distance'] = ([1e6*(x<0) for x in lrst['distance'].values]) + lrst['distance']
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
            tr2xy_n.append([x, y, 100*r.rsquared, lrs['tech'].nunique()])
        else:
            tr2xy_n.append([x, y, np.nan, lrs['tech'].nunique()])


        ## point fraction
        # cals, vals, = [], []
        # for t in lrs['tech'].unique():
        #     lrst = lrs.loc[(lrs['tech']==t)].copy()
        #     lrst['distance'] = (lrst['pfrac'] - x)
        #     lrst['distance'] = ([1e6*(x<0) for x in lrst['distance'].values]) + lrst['distance']
        #     cals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['cal']].values[0])
        #     vals.append(lrst.loc[lrst['distance']==lrst['distance'].min(),['val']].values[0])
        # vals = np.array(vals)
        # cals = np.array(cals)
        # m = sm.OLS(vals, \
        #         sm.add_constant(cals)
        #         )
        # r = m.fit()
        # if r.f_pvalue < 1.05:
        #     r2xy_p.append([x, y, 100*r.rsquared, lrs['tech'].nunique()])
        # else:
        #     r2xy_p.append([x, y, np.nan, lrs['tech'].nunique()])

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
            tr2xy_p.append([x, y, 100*r.rsquared, lrs['tech'].nunique()])
        else:
            tr2xy_p.append([x, y, np.nan, lrs['tech'].nunique()])

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
            tr2xy_p_n.append([x, y, 100*r.rsquared, lrs['tech'].nunique()])
        else:
            tr2xy_p_n.append([x, y, np.nan, lrs['tech'].nunique()])
        # # moving average
        # lrs_ = lrs.loc[(lrs['calint']/(lrs['calint']+lrs['valint']) >= x-0.05) & \
        #            (lrs['calint']/(lrs['calint']+lrs['valint']) < x+0.05)
        #            ].copy()
        # # lrs_ = lrs.loc[((10.0**(lrs['calint'] - 1)) / (10.0**(lrs['calint']+lrs['valint']-1)) >= x - 0.4)  & \
        # #            ((10.0**(lrs['calint'] - 1) / ( 10.0** (lrs['calint']+lrs['valint'] - 1)) < x + 0.4))
        # #            ].copy()
        # for t in lrs_['tech'].unique():
        #     lrs_.loc[lrs_['tech']==t,'weights'] = 1 / \
        #         lrs_.loc[lrs_['tech']==t].count()[0] / \
        #         lrs_['tech'].nunique()
        # m = sm.WLS(lrs_['val'].values, \
        #         sm.add_constant(lrs_['cal'].values),
        #         weights=lrs_['weights'])
        # r = m.fit()
        # if r.f_pvalue < 1.05:
        #     r2xy_ma.append([x, y, 100*r.rsquared, lrs_['tech'].nunique()])
        # else:
        #     r2xy_ma.append([x, y, np.nan, lrs_['tech'].nunique()])

        # lrs_ = lrs_.loc[lrs_['tot']==1]
        # if lrs_.empty:
        #     tr2xy_ma.append([x, y, np.nan, np.nan])
        #     continue
        # for t in lrs_['tech'].unique():
        #     lrs_.loc[lrs_['tech']==t,'weights'] = 1 / \
        #         lrs_.loc[lrs_['tech']==t].count()[0] / \
        #         lrs_['tech'].nunique()
        # m = sm.WLS(lrs_['val'].values, \
        #         sm.add_constant(lrs_['cal'].values),
        #         weights=lrs_['weights'])
        # r = m.fit()
        # if r.f_pvalue < 1.05:
        #     tr2xy_ma.append([x, y, 100*r.rsquared, lrs_['tech'].nunique()])
        # else:
        #     tr2xy_ma.append([x, y, np.nan, lrs_['tech'].nunique()])

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
         tr2xy[:,2],
         label='Logarithmic cumulative production range')
ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_n[:,2],
         label='Logarithmic cumulative production range (no nuclear)')
print(0.5,np.where((tr2xy[:,0]>=0.5)*(tr2xy[:,0]<0.5+stepv))[0][0])
ax[1].scatter(0.5,tr2xy[np.where((tr2xy[:,0]>=0.5)*(tr2xy[:,0]<0.5+stepv))[0][0],2], 
              color='k', marker='o', s=50)
# ax[1].annotate('A', xy=(0.5, 1.2*tr2xy[np.where((tr2xy[:,0]>=0.5)*(tr2xy[:,0]<0.5+stepv))[0][0],2]), 
#                ha='center', va='bottom',
#                xycoords='data', fontsize=14)
# ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
#          tr2xy_ma[:,2],
#          label='Full range - moving average')
ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_p[:,2],
         label='Number of data points')
ax[1].plot([x for x in np.arange(minv, maxv, stepv)], 
         tr2xy_p_n[:,2],
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

range = 2
window = 1.0
# window = 0.5
r2xy = []
print(lrs['tech'].nunique())
# for x in np.linspace(0,range,int(range/window*2)+1):
#     for y in np.linspace(0,range,int(range/window*2)+1):
for x in np.linspace(0,range,int(range/window)+1):
    for y in np.linspace(0,range,int(range/window)+1):
        lrs_ = lrs.loc[(lrs['calint'] >= x) & \
                   (lrs['calint'] < x+window) & \
                     (lrs['valint'] >= y) & \
                     (lrs['valint'] < y+window)
                   ].copy()
        if not(lrs_.empty) and lrs_['tech'].nunique() >= 2:
            for t in lrs_['tech'].unique():

                lrs_.loc[lrs['tech']==t,'weights'] = 1 / \
                    lrs_.loc[lrs['tech']==t].count()[0] / \
                    lrs_['tech'].nunique()
            m = sm.WLS(lrs_['val'].values, \
                    sm.add_constant(lrs_['cal'].values),
                    weights=lrs_['weights'])
            r = m.fit()
            if r.f_pvalue < 1.05:
                r2xy.append([x,y,100*r.rsquared,lrs_['tech'].nunique()])
            else:
                r2xy.append([x,y,np.nan, lrs_['tech'].nunique()])
        else:
            r2xy.append([x,y,np.nan, np.nan])
r2xy = np.array(r2xy)

fig, ax = plt.subplots(1)

r2xydf = pd.DataFrame(r2xy, columns=['calint','valint','r2','count'])
r2xydf = r2xydf.loc[(r2xydf['calint'] <= 3.0) & \
    (r2xydf['valint'] <= 3.0)]

r2xydfp = r2xydf.copy().pivot(index='calint', columns='valint', values='count')

sns.heatmap(r2xydfp,
            cmap=cmcrameri.cm.batlow,
            annot=r2xydf.copy().pivot(index='calint', columns='valint', values='r2'), 
            fmt='.0f', 
            cbar_kws={'label': 'Number of technologies',
                      'ticks':[2,10,20,30,40,50,60,70],
                      },
            ax = ax)
xmin, xmax = ax.get_xlim()
ax.set_xticks(np.linspace(xmin, xmax, int((range+1)/window)+1))
ax.set_yticks(np.linspace(xmin, xmax, int((range+1)/window)+1))
ax.set_xticklabels(['10$^{'+str(x)+'}$' for x in np.linspace(0,range+window,int((range+window)/window)+1)])
ax.set_yticklabels(['10$^{'+str(x)+'}$' for x in np.linspace(0,range+window,int((range+window)/window)+1)])

ax.set_aspect('equal')


vals, wgts = np.ma.MaskedArray(r2xy[:,2], mask=np.isnan(r2xy[:,2])), \
                np.ma.MaskedArray(r2xy[:,3], mask=np.isnan(r2xy[:,2]))
print(np.average(vals, weights=wgts))

ax.invert_yaxis()
ax.set_xlabel('Ratio of future to current cumulative production')
ax.set_ylabel('Ratio of current to initial cumulative production')

fig.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.85)

plt.show()
exit()
lrs['diff'] = 10**((lrs['cal'] - lrs['val']) * (lrs['valint']**2) /2 )
lrs['diff2'] = 10**((np.mean(slopes) - lrs['val']) * (lrs['calint']**2) /2 )
lrs['weights'] = 1
for t in lrs['tech'].unique():
    lrs.loc[lrs['tech'] == t,'weights'] = 1 / \
        lrs.loc[lrs['tech'] == t].count()[0] / \
        lrs['tech'].nunique()

for t in lrs['tech'].unique():
    lrs_ = lrs.loc[lrs['tech'] == t]
    ax[0].scatter(lrs_['valint'], 
                    #  lrs['valint'], 
                    lrs_['diff'],
                # c=lrs['calint'],
                # norm=matplotlib.colors.Normalize(vmin=0, vmax=lrs['calint'].max()), 
                # cmap=cmcrameri.cm.batlow,
                alpha=0.3,)
    ax[1].scatter(lrs_['valint'], 
                    #  lrs['valint'], 
                    lrs_['diff2'],
                # c=lrs['calint'],
                # norm=matplotlib.colors.Normalize(vmin=0, vmax=lrs['calint'].max()), 
                # cmap=cmcrameri.cm.batlow,
                alpha=0.3,)
ax[0].set_yscale('log', base=10)
    # plt.pause(0.2)

max = 10
interval = 0.5
per95 = []
per5 = []
medavg = []
per95_2 = []
per5_2 = []
medavg_2 = []
for x in np.linspace(0,10,101):
    lrs_ = lrs.loc[(lrs['valint'] > x) & \
                   (lrs['valint'] < x+interval)
                   ]
    per95.append(lrs_['diff'].quantile(.95))
    per5.append(lrs_['diff'].quantile(.05))
    medavg.append(lrs_['diff'].median())
    per95_2.append(lrs_['diff2'].quantile(.95))
    per5_2.append(lrs_['diff2'].quantile(.05))
    medavg_2.append(lrs_['diff2'].median())
ax[0].plot(np.linspace(0,10,101), per95, color='r')
ax[0].plot(np.linspace(0,10,101), per5, color='r')
ax[0].plot(np.linspace(0,10,101), medavg, 'k--')
ax[1].plot(np.linspace(0,10,101), per95_2, color='r')
ax[1].plot(np.linspace(0,10,101), per5_2, color='r')
ax[1].plot(np.linspace(0,10,101), medavg_2, 'k--')
plt.show()
# plt.colorbar(ax)
# plt.ylim(0,10)

# sns.scatterplot(lrs, x='calint', y='valint',
#             hue='diff', alpha=0.2)
print(lrs.shape)

plt.show()

# compute weights
lrs['weights'] = 1
for t in lrs['tech'].unique():
    lrs.loc[lrs['tech'] == t,'weights'] = 1 / \
        lrs.loc[lrs['tech'] == t].count()[0] / \
        lrs['tech'].nunique()
    
sns.kdeplot(lrs, x='cal', y='val', levels=10,
            weights=lrs['weights'], cmap=cmcrameri.cm.batlow,
            ax = ax[0])

ax[0].scatter(lrs['cal'].values,lrs['val'].values,
                  alpha=0.1, zorder=-1, color='None',
                  edgecolor=cmcrameri.cm.batlow(1))

ax[0].set_xlim(-3,3)
ax[0].set_ylim(-3,3)
ax[0].set_xlabel('Past learning exponent')
ax[0].set_ylabel('Future learning exponent')
ax[0].set_aspect('equal')

# compute r squared
model = sm.OLS(lrs['val'].values, \
               sm.add_constant(lrs['cal'].values))
results = model.fit()
print(results.rsquared)

# compute weighted r squared
model = sm.WLS(lrs['val'].values, \
               sm.add_constant(lrs['cal'].values),
               weights=lrs['weights'])
results = model.fit()
print(results.rsquared)

# bootstrap weighted r squared
r2 = []
for iter in range(1000):
    lrs_ = lrs.sample(n=lrs.shape[0], 
                      replace=True, weights=lrs['weights'])
    
    lrs_['weights'] = 1
    for t in lrs_['tech'].unique():
        lrs_.loc[lrs_['tech'] == t,'weights'] = 1 / \
            lrs_.loc[lrs_['tech'] == t].count()[0] / \
            lrs_['tech'].nunique()

    model = sm.WLS(lrs_['val'].values, \
                sm.add_constant(lrs_['cal'].values),
                weights=lrs_['weights'])
    results = model.fit()
    r2.append(100*results.rsquared)

sns.boxplot(r2, ax=ax[1]) 
ax[1].set_xticks([])
ax[1].set_ylabel('Fraction of explained variance [%]')
fig.subplots_adjust(hspace=0.4, wspace=0.4,
                    right=0.9, left=0.1, 
                    top=0.95, bottom=0.1)
   
plt.show()