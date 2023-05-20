import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn, scipy
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

df = pd.read_csv('ExpCurves.csv')

# get slope for all technologies
x = np.log10(df['Cumulative production'].values)
y = np.log10(df['Unit cost'].values)
model = sm.OLS(y, sm.add_constant(x))
result = model.fit()
slopeall = result.params[1]

# get error using technology specific slope
count = 0
dferr = []
for tech in df['Tech'].unique():
    sel = df.loc[df['Tech']==tech]
    x = np.log10(sel['Cumulative production'].values)
    y = np.log10(sel['Unit cost'].values)
    H = len(x)
    # select N points before midpoint and compute slope
    for N in range(1, round(H/2)):
        slope = (y[round(H/2)-1] - y[round(H/2)-N-1]) /\
            (x[round(H/2)-1] - x[round(H/2)-N-1])
        # compute error associated using slope M points after midpoint
        for M in range(round(H/2),H):
            error = (10**(y[M] - y[round(H/2)-1] -\
                slopeall * (x[M] - x[round(H/2)-1])))**2
            dferr.append([x[round(H/2)-1] - x[round(H/2)-N-1],
                         x[M] - x[round(H/2)-1],
                         error])
            count += 1

dferr = pd.DataFrame(dferr, 
                     columns = ['Log of ratios for predictor',
                                'Log of ratios for prediction',
                                'Squared fractional error'])

# select ratios to be plotted
frac = np.log10([1, 2, 3, 5, 10, 1e2, 1e3, 1e9])

mean = np.empty((len(frac)-1,len(frac)-1))
count = np.empty((len(frac)-1,len(frac)-1))
for i in range(1, len(frac)):
    for j in range(1, len(frac)):
        select = dferr.loc[
            (dferr['Log of ratios for predictor']>frac[i-1]) &\
            (dferr['Log of ratios for predictor']<=frac[i]) & \
            (dferr['Log of ratios for prediction']>frac[j-1]) &\
            (dferr['Log of ratios for prediction']<=frac[j])]
        mean[i-1,j-1] = np.mean(select['Squared fractional error'].values)
        count[i-1,j-1] = (select['Squared fractional error'].count())
        # if count[i-1, j-1] < 10:
        #     count[i-1,j-1] = np.nan
        #     mean[i-1,j-1] = np.nan

print(frac)
im = plt.imshow(mean, aspect='auto')
plt.gca().set_xticks([x for x in range(7)], 
        [str(round(x,2))+' to '+str(round(y,2)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(7)], 
        [str(round(x,2))+' to '+str(round(y,2)) for x, y in zip(frac[:-1], frac[1:])])
plt.xlabel('Log of ratios for prediction')
plt.ylabel('Log of ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('Squared fractional error')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)
plt.suptitle('Average technological slope')

# plt.figure()
# im = plt.imshow(count)
# plt.gca().set_xticks([x for x in range(7)], 
#         [str(round(x,2))+' to '+str(round(y,2)) for x, y in zip(frac[:-1], frac[1:])],
#         rotation = 90)
# plt.gca().set_yticks([x for x in range(7)], 
#         [str(round(x,2))+' to '+str(round(y,2)) for x, y in zip(frac[:-1], frac[1:])])
# plt.colorbar(im)

plt.show()

exit()














count = 0
tot = 0
slopes = []
slopes_normalized = []
normal = []
normal_normalized = []
CProd = []
for tech in df['Tech'].unique():
    # read data for specific technology
    sel = df.loc[df['Tech'] == tech]
    try:
        x = np.log10(sel['Cumulative production'].values)
        y = np.log10(sel['Unit cost'].values)
    except KeyError:
        x = np.log10(sel['Normalized cumulative production'].values)
        y = np.log10(sel['Normalized unit cost'].values)

    slope = []
    slope_normalized = []
    for n in range(x.shape[0]-1):
        slope.append((y[n+1] - y[n]) / (x[n+1]-x[n]))
        slope_normalized.append((y[n+1] - y[n]) / (x[n+1]-x[n]) * \
                                (x[n+1]-x[n]) / (x[-1] - x[0]))
    slope = np.array(slope)
    slope_normalized = np.array(slope_normalized)
    slopes.append(slope)
    slopes_normalized.append(slope_normalized)
    CProd.append(10**x)
    tot += 1
    if scipy.stats.shapiro(slope)[1] < 0.05 :
        count += 1
        normal.append(tot)
    if scipy.stats.shapiro(slope_normalized)[1] < 0.05 :
        count += 1
        normal_normalized.append(tot)
print(len(normal), len(normal_normalized))

fig, ax = plt.subplots(figsize=(10,7))
ratios = []
for slope in slopes:
    dslope = np.diff(slope)
    dslopes = []
    for ds in dslope:
        # print(ds, np.abs(ds),np.abs(np.mean(slope)))
        dslopes.append(100*np.abs(ds)/np.abs(np.mean(slope)))
    # exit()
    while len(dslopes) < 30:
        dslopes.append(np.nan)
    ratios.append(np.array(dslopes[:30]))

ratios = np.vstack(ratios)
im = ax.imshow(ratios, aspect='auto')
im.set_clim(0,100)
cbar = plt.colorbar(im)
# ax.set_yticks([x for x in range(ratios.shape[0])], df['Tech'].unique(), fontsize=5)
ax.set_ylabel('Technologies')
cbar.set_label('Slope change over mean slope [-]')
plt.subplots_adjust(left=0.2)


cov = []
SNR = []
mean = []
for slope in slopes:
    cov.append(np.std(slope)/np.abs(np.mean(slope)))
    SNR.append(np.abs(np.mean(slope))/np.std(slope-0.15))
    mean.append(np.mean(slope))

plt.figure()
slopes_ = np.array([x for el in slopes for x in el])
print(1 - 2**np.mean(slopes_))
sums = []
for el in slopes_normalized:
    sums.append(sum(el))
mlrALL = np.mean(sums)
print(1 - 2**np.mean(sums))
sums = []
for el in slopes:
    sums.append(np.mean(el))
print(1 - 2**np.mean(sums))
## focus on slopes_normalized because it makes more sense than the other ones
print('For each fraction compute R2 for regression and for past mean value')
for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
    eLR, lLR = [], []
    print(frac)
    for slope in slopes_normalized:
        eLR.append(sum(slope[:round(len(slope)*frac)]))
        lLR.append(sum(slope[round(len(slope)*frac):]))
    model = sm.OLS(lLR, sm.add_constant(eLR))
    result = model.fit()
    print('\t'+str(result.rsquared))
    SStot = sum((lLR-np.mean(lLR))**2)
    SSmlr = sum((lLR-np.mean(eLR))**2)
    print('\t'+str(1 - SStot/SSmlr))
    if frac == 0.5:
        mlrALL = np.mean(eLR)
print(mlrALL)
dfij = []
for frac1 in [10, 50, 100, 500, 1000, 1e10]:
    for frac2 in [10, 50, 100, 500, 1000, 1e10]:
        eLR = []
        lLR = []
        print('Order of magnitude of calibration and validation: ', frac1, frac2)
        for idx, slope in enumerate(slopes_normalized):
            eLR_ = slope[:round(len(slope)*0.5)]
            lLR_ = slope[round(len(slope)*0.5):]
            cprod = CProd[idx]
            cprod = (cprod/cprod[0])[1:]
            eLR_new = []
            lLR_new = []
            count = 0
            for x in eLR_:
                if cprod[round(len(cprod)*1/2)] / cprod[count] < frac1:
                    eLR_new.append(eLR_[count])
                count += 1
            count = 0
            cprod = cprod[round(len(cprod)*1/2):]
            for x in lLR_:
                if cprod[count]/cprod[0] < frac2:
                    lLR_new.append(lLR_[count])
                count += 1
            eLR.append(sum(eLR_new))
            lLR.append(sum(lLR_new))
        model = sm.OLS(lLR, sm.add_constant(eLR))
        result = model.fit()
        print('\t'+str(result.rsquared))
        SStot = sum((lLR-mlrALL)**2)
        SSmlr = sum([(x-y)**2 for x,y in zip(lLR,eLR)])
        r2mlr = 1 - SSmlr/SStot
        SSmlrALL = sum((lLR-mlrALL)**2)
        r2mlrALL = 1 - SSmlrALL/SStot
        result.rsquared = 1 - sum((lLR - result.params[0] -\
                              result.params[1]*np.array(eLR))**2)/SStot
        print('\t'+str(r2mlr))
        dfij.append([frac1, frac2, result.rsquared, r2mlr, r2mlrALL])
dfij = pd.DataFrame(dfij, columns=['Frac1', 'Frac2','R2','R2m','R2mall'])
dfij['R2 improvement'] = dfij['R2'] - dfij['R2mall']
dfij['R2 improvement mtom'] = dfij['R2m'] - dfij['R2mall']
dfij['R2 improvement total'] = dfij['R2'] - dfij['R2mall']
R2diff = []
# for frac2 in dfij['Frac2'].unique()[::-1]:
#     r2diff = []
#     for frac1 in dfij['Frac1'].unique():
#         r2diff.append(dfij.loc[dfij['Frac1']==frac1]\
#                       .loc[dfij['Frac2']==frac2,'R2 improvement'].values[0])
#     R2diff.append(np.array(r2diff)) 
# R2diff = np.vstack(R2diff)
# im = plt.imshow( R2diff, aspect='auto', vmin=0)
# plt.xticks([0,1,2,3,4,5],[1, 1.5, 2, 2.5, 3, 'Half of data set'])
# plt.yticks([0,1,2,3,4,5],[1, 1.5, 2, 2.5, 3, 'Half of data set'][::-1])
# plt.xlabel('Orders of magnitude used for estimation of past learning rate')
# plt.ylabel('Orders of magnitude used for estimation of future learning rate')
# cbar = plt.colorbar(im)
# cbar.set_label('R2 improvement')
# plt.title('Improvement using regression over using mean past learning rate of all technologies')

plt.figure()
R2diff = []
for frac2 in dfij['Frac2'].unique()[::-1]:
    r2diff = []
    for frac1 in dfij['Frac1'].unique():
        r2diff.append(dfij.loc[dfij['Frac1']==frac1]\
                      .loc[dfij['Frac2']==frac2,'R2 improvement mtom'].values[0])
    R2diff.append(np.array(r2diff)) 
R2diff = np.vstack(R2diff)
im = plt.imshow( R2diff, aspect='auto')
plt.xticks([0,1,2,3,4,5],[1, 1.5, 2, 2.5, 3, 'Half of data set'])
plt.yticks([0,1,2,3,4,5],[1, 1.5, 2, 2.5, 3, 'Half of data set'][::-1])
plt.xlabel('Orders of magnitude used for estimation of past learning rate')
plt.ylabel('Orders of magnitude used for estimation of future learning rate')
cbar = plt.colorbar(im)
cbar.set_label('R2 improvement')
plt.title('Improvement using mean past learning rate of speficic technology over using mean past learning rate of all technologies')

# plt.figure()
# R2diff = []
# for frac2 in dfij['Frac2'].unique()[::-1]:
#     r2diff = []
#     for frac1 in dfij['Frac1'].unique():
#         r2diff.append(dfij.loc[dfij['Frac1']==frac1]\
#                       .loc[dfij['Frac2']==frac2,'R2 improvement total'].values[0])
#     R2diff.append(np.array(r2diff)) 
# R2diff = np.vstack(R2diff)
# im = plt.imshow( R2diff, aspect='auto', vmin=0)
# plt.xticks([0,1,2,3,4,5],[1, 1.5, 2, 2.5, 3, 'Half of data set'])
# plt.yticks([0,1,2,3,4,5],[1, 1.5, 2, 2.5, 3, 'Half of data set'][::-1])
# plt.xlabel('Orders of magnitude used for estimation of past learning rate')
# plt.ylabel('Orders of magnitude used for estimation of future learning rate')
# cbar = plt.colorbar(im)
# cbar.set_label('R2 improvement')
# plt.title('Improvement over prediction using mean past learning rate of all technologies')

plt.show()

exit()


plt.figure()
im = plt.imshow(slopes__, vmin=-1, vmax=1, aspect='auto', cmap='RdBu')
plt.colorbar(im)
plt.show()
seaborn.displot(slopes_.flatten(), rug=True, kind='kde')

plt.figure()
plt.scatter(np.arange(0,86,1), mean)
plt.plot([0,86],[np.mean(mean),np.mean(mean)])
# plt.xticks([x for x in range(len(slopes))], 
#               df['Tech'].unique(), 
#               fontsize=5, rotation=90)
# plt.plot([0,86],[np.mean(slopes),np.mean(slopes)])
plt.ylabel('Mean')
plt.xlabel('Technologies')


plt.figure()
plt.plot(sorted(SNR))
plt.ylabel('Signal-to-noise ratio')
plt.xlabel('Technologies')
plt.annotate('More noise than signal', (70,0.5), ha='center')
plt.annotate('More signal than noise', (20,1.5), ha='center')
plt.plot([0,86],[1,1])


plt.figure()
plt.plot(sorted(cov))
plt.ylabel('Coefficient of variation')
plt.xlabel('Technologies')

plt.show()
print(count, tot)