import pandas as pd
import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pwlf
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import mutual_info_score

df = pd.read_csv('ExpCurves.csv')

np.random.seed(1)

slopes_pwlf = []
slopes = []
for tech in df['Tech'].unique():
    sel = df.loc[df['Tech'] == tech]
    if sel.count()[0] < 15:
        continue
    x = np.log10(sel['Cumulative production'].values)
    y = np.log10(sel['Unit cost'].values)
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    # fit the data for four line segments
    # this performs 3 multi-start optimizations
    n_segments = 3
    bounds = np.zeros((n_segments-1, 2))
    bounds[0,0] = max(x)*0.33
    bounds[0,1] = max(x)*0.33
    bounds[1,0] = max(x)*0.66
    bounds[1,1] = max(x)*0.66
    res = my_pwlf.fit(n_segments
                      , bounds=bounds
                      )
    res = my_pwlf.calc_slopes()
    slopes_pwlf.append(res)

    # ken's approach - computing linear regression over segments defined by cumulative amount

    segments = [0, max(x)*0.33, max(x)*0.67, max(x)]

    slopes_ = []
    for seg in range(1,len(segments)):
        x_ = x[(x<=segments[seg])*(x>=segments[seg-1])]
        y_ = y[(x<=segments[seg])*(x>=segments[seg-1])]
        if x_.any(): 
            x_ = sm.add_constant(x_)
            model = sm.OLS(y_,x_)
            res = model.fit()
            [slopes_.append(res.params[1]) if len(res.params)>1 
                else slopes_.append(res.params[0])]
        else:
            slopes_.append(0)
    slopes.append(slopes_)
                     

    # fig, ax = plt.subplots()
    # ax.scatter(10**x, 10**y)
    # x_eval = np.linspace(0,max(x), 1000)
    # ax.plot(10**x_eval, 10**np.asarray(my_pwlf.predict(x_eval)))
    # ax.set_xscale('log', base = 10)
    # ax.set_yscale('log', base = 10)
    # ax.set_ylabel('Unit cost')
    # ax.set_xlabel('Cumulative production')
    # ax.plot()
    # plt.show()

## 
print('\n\n using the first slope to predict the third ')
slopes_pwlf = np.asarray(slopes_pwlf)
dfslopes = pd.DataFrame(slopes_pwlf, columns = ['I slope','II slope','III slope'])
dfslopes.to_csv('Slopes_Piecewise.csv')
plt.figure()
plt.scatter(slopes_pwlf.T[0], slopes_pwlf.T[2],
             edgecolor = 'tab:blue', 
             facecolor='none', alpha=0.5)
plt.ylabel('Third slope')
plt.xlabel('First slope')
X = slopes_pwlf.T[0]
X = sm.add_constant(X)
y = slopes_pwlf.T[2]
model = sm.OLS(y,X)
res = model.fit()
print(res.summary())
x_eval = np.arange(-10,10,0.1) 
x_eval = sm.add_constant(x_eval)
plt.plot([-10,10],[-10,10])
plt.plot(x_eval.T[1], res.predict(x_eval),'r--')
plt.xlim((-2,2))
plt.ylim((-2,2))

## 
print('\n\n using the second slope to predict the third ')
plt.figure()
plt.scatter(slopes_pwlf.T[1], slopes_pwlf.T[2],
             edgecolor = 'tab:blue', 
             facecolor='none', alpha=0.5)
plt.ylabel('Third slope')
plt.xlabel('Second slope')
X = slopes_pwlf.T[1]
X = sm.add_constant(X)
model = sm.OLS(y,X)
res = model.fit()
print(res.summary())
x_eval = np.arange(-10,10,0.1) 
x_eval = sm.add_constant(x_eval)
plt.plot([-10,10],[-10,10])
plt.plot(x_eval.T[1], res.predict(x_eval),'r--')
plt.xlim((-2,2))
plt.ylim((-2,2))

print('\n\n using first and second slope to predict the third ')
X = slopes_pwlf.T[[0,1]].reshape(-1,2)
y = slopes_pwlf.T[2]
X = sm.add_constant(X)
model = sm.OLS(y,X)
res = model.fit()
print(res.summary())
plt.figure()
X1, X2 = np.mgrid[-2:2.1:0.1, -2:2.1:0.1]
x_eval = sm.add_constant(np.mgrid[-2:2.1:0.1, -2:2.1:0.1].reshape(2,-1).T)
z = res.predict(x_eval).T.reshape(-1,41)
plt.contourf(X1, X2, z )
plt.xlim((-2,2))
plt.ylim((-2,2))

# plt.figure()
# plt.hist2d(slopes_pwlf.T[0], slopes_pwlf.T[2], bins=[100,100], range=[[-2,2],[-2,2]])
# plt.xlabel('First slope')
# plt.ylabel('Third slope')
# plt.plot([-10,10],[-10,10], 'r--', alpha = 0.3)
# plt.xlim((-2,2))
# plt.ylim((-2,2))

# plt.figure()

# plt.hist(slopes_pwlf.T[2] / slopes_pwlf.T[0], bins = 10, range=[-2,2], density=True)
# plt.xlabel('Ratio of third to first slope')



## 
print('\n\n using the first slope to predict the third ')
slopes = np.asarray(slopes)
dfslopes = pd.DataFrame(slopes, columns = ['I slope','II slope','III slope'])
dfslopes.to_csv('Slopes_Piecewise.csv')

print('mutual information of I and II slope with III slope: ')
print(mutual_info_regression(dfslopes[['I slope','II slope']].values, 
                             dfslopes['III slope'].values, n_neighbors=10))
print(mutual_info_regression(dfslopes[['I slope']].values, 
                             dfslopes['III slope'].values))
print(mutual_info_regression(dfslopes[['II slope']].values, 
                             dfslopes['III slope'].values))
for x in dfslopes[['I slope','II slope']].values.T:
    y = dfslopes['III slope'].values
    c_xy = np.histogram2d(x, y, 7)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    print(mi)

# compute MI
nbins = 7
# compute segments or bins to be used (5 bins because of few data using sturges law)
pi = np.linspace(min(dfslopes['I slope'].values),
                 max(dfslopes['I slope'].values)+1e-6,
                 nbins+1)
pii = np.linspace(min(dfslopes['II slope'].values),
                 max(dfslopes['II slope'].values)+1e-6,
                 nbins+1)
piii = np.linspace(min(dfslopes['III slope'].values),
                 max(dfslopes['III slope'].values)+1e-6,
                 nbins+1)
# compute marginal probs
pri = []
prii = []
priii = []
for n in range(1,nbins+1):
    pri.append(dfslopes.loc[(dfslopes['I slope'].values<pi[n]) & \
                        (dfslopes['I slope'].values>=pi[n-1])].count()[0] / \
                        dfslopes.count()[0])
    prii.append(dfslopes.loc[(dfslopes['II slope'].values<pii[n]) & \
                        (dfslopes['II slope'].values>=pii[n-1])].count()[0] / \
                        dfslopes.count()[0])
    priii.append(dfslopes.loc[(dfslopes['III slope'].values<piii[n]) & \
                        (dfslopes['III slope'].values>=piii[n-1])].count()[0] / \
                        dfslopes.count()[0])

#compute joint probs
pri_iii = np.zeros((nbins, nbins))
prii_iii = np.zeros((nbins, nbins))
for n in range(1,nbins+1):
    for nn in range(1,nbins+1):
        pri_iii[n-1,nn-1] = dfslopes.loc[(dfslopes['I slope'].values<pi[n]) & \
                            (dfslopes['I slope'].values>=pi[n-1]) &\
                            (dfslopes['III slope'].values<piii[nn]) &\
                            (dfslopes['III slope'].values>=piii[nn-1])].count()[0] / \
                            dfslopes.count()[0]
        prii_iii[n-1,nn-1] = dfslopes.loc[(dfslopes['II slope'].values<pii[n]) & \
                            (dfslopes['II slope'].values>=pii[n-1]) &\
                            (dfslopes['III slope'].values<piii[nn]) &\
                            (dfslopes['III slope'].values>=piii[nn-1])].count()[0] / \
                            dfslopes.count()[0]

#compute mi
MI_I_III = 0.0
MI_II_III = 0.0
for n in range(nbins):
    for nn in range(nbins):
        if pri_iii[n,nn] > 0 :
            MI_I_III +=  pri_iii[n,nn] * np.log(pri_iii[n,nn]/(pri[n]*priii[nn]))
        if prii_iii[n,nn] > 0:
            MI_II_III += prii_iii[n,nn] * np.log(prii_iii[n,nn]/(prii[n]*priii[nn]))
MI_I_III = MI_I_III
MI_II_III = MI_II_III
print('Mutual information check: ', MI_I_III, MI_II_III)

plt.figure()
plt.scatter(slopes.T[0], slopes.T[2],
             edgecolor = 'tab:blue', 
             facecolor='none', alpha=0.5)
plt.ylabel('Third slope')
plt.xlabel('First slope')
X = slopes.T[0]
X = sm.add_constant(X)
y = slopes.T[2]
model = sm.OLS(y,X)
res = model.fit()
print(res.summary())
x_eval = np.arange(-10,10,0.1) 
x_eval = sm.add_constant(x_eval)
plt.plot([-10,10],[-10,10])
plt.plot(x_eval.T[1], res.predict(x_eval),'r--')
plt.xlim((-2,2))
plt.ylim((-2,2))

## 
print('\n\n using the second slope to predict the third ')
plt.figure()
plt.scatter(slopes.T[1], slopes.T[2],
             edgecolor = 'tab:blue', 
             facecolor='none', alpha=0.5)
plt.ylabel('Third slope')
plt.xlabel('Second slope')
X = slopes.T[1]
X = sm.add_constant(X)
model = sm.OLS(y,X)
res = model.fit()
print(res.summary())
x_eval = np.arange(-10,10,0.1) 
x_eval = sm.add_constant(x_eval)
plt.plot([-10,10],[-10,10])
plt.plot(x_eval.T[1], res.predict(x_eval),'r--')
plt.xlim((-2,2))
plt.ylim((-2,2))

print('\n\n using first and second slope to predict the third ')

plt.figure()
CS = plt.scatter(slopes.T[0], 
                 slopes.T[1], 
                 c = slopes.T[2],
                 vmin=-1, vmax=1
)
cbar = plt.colorbar(CS)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('First slope')
plt.ylabel('Second slope')
cbar.set_label('Third slope')
X = slopes.T[[0,1]].reshape(-1,2)
y = slopes.T[2]
X = sm.add_constant(X)
model = sm.OLS(y,X)
res = model.fit()
print(res.summary())
plt.figure()
X1, X2 = np.mgrid[-2:2.1:0.1, -2:2.1:0.1]
x_eval = sm.add_constant(np.mgrid[-2:2.1:0.1, -2:2.1:0.1].reshape(2,-1).T)
z = res.predict(x_eval).T.reshape(-1,41)
CS = plt.contourf(X1, X2, z , levels=10,)
plt.xlabel('First slope')
plt.xlabel('Second slope')
plt.xlim((-2,2))
plt.ylim((-2,2))
cbar = plt.colorbar(CS)
cbar.set_label('Third slope')
plt.show()
