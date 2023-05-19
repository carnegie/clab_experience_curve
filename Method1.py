import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

df = pd.read_csv('ExpCurves.csv')
# df = pd.read_csv('NormalizedExpCurves.csv')

fig, ax = plt.subplots(10,9)
count=0
for tech in df['Tech'].unique():
    if count in [81]:
        for x in range(2):
            ax[int(count/9)][count%9].axis('off')
            count += 1
    sel = df.loc[df['Tech'] == tech]
    uc = sel['Unit cost'].values /sel['Unit cost'].values[0]
    cp = sel['Cumulative production'].values/sel['Cumulative production'].values[0]
    ax[int(count/9)][count%9].plot(cp,uc)
    ax[int(count/9)][count%9].set_yscale('log', base=10)
    ax[int(count/9)][count%9].set_xscale('log', base=10)
    ax[int(count/9)][count%9].axis('off')
    count += 1
ax[-1][-2].axis('off')
ax[-1][-1].axis('off')



fraction = 1/2

LR_cal = []
LR_val = []
figscatter , axscatter = plt.subplots()
fig2 , ax2 = plt.subplots(10,9, figsize=(13,7))
fig3 , ax3 = plt.subplots(10,9, figsize=(13,7))
count = 0
better = 0
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
    x_cal = x[:round(x.shape[0]*fraction)]
    x_val = x[round(x.shape[0]*fraction):]
    y_cal = y[:round(y.shape[0]*fraction)]
    y_val = y[round(y.shape[0]*fraction):]
    # # alternative - separate calibration and validation datasets based on fraction of cumulative production
    # x_cal = x[x<(x[-1]-x[0])*fraction + x[0]]
    # x_val = x[x>=(x[-1]-x[0])*fraction + x[0]]
    # y_cal = y[x<(x[-1]-x[0])*fraction + x[0]]
    # y_val = y[x>=(x[-1]-x[0])*fraction + x[0]]
    # perform regression on both datasets separately
    model_cal = sm.OLS(y_cal, sm.add_constant(x_cal))
    result_cal = model_cal.fit()
    model_val = sm.OLS(y_val, sm.add_constant(x_val))
    result_val = model_val.fit()
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()
    LRall = result.params[1]
    axscatter.scatter(result_cal.params[1], result_val.params[1], 
                color='tab:red', facecolor='None', alpha = 0.6)
    LR_cal.append(result_cal.params[1])
    LR_val.append(result_val.params[1])
    if count in [81]:
        for x in range(2):
            ax2[int(count/9)][count%9].axis('off')
            ax3[int(count/9)][count%9].axis('off')
            count += 1
    uc = sel['Unit cost'].values /sel['Unit cost'].values[0]
    cp = sel['Cumulative production'].values/sel['Cumulative production'].values[0]
    ax2[int(count/9)][count%9].scatter(cp, uc, 
                                       marker='o', 
                                       color='firebrick',
                                       lw=0.5, 
                                       facecolor='None', s=2)
    ax2[int(count/9)][count%9].set_yscale('log', base=10)
    ax2[int(count/9)][count%9].set_xscale('log', base=10)
    ucpred = 10**(result_cal.params[0] + result_cal.params[1] * \
        np.concatenate([x_cal, x_val]))/sel['Unit cost'].values[0]
    errpred = 10**(result_cal.params[0] + result_cal.params[1] * \
        x_val) - sel['Unit cost'].values[len(x_cal):]
    errpred = result_cal.params[0] + result_cal.params[1] * \
        x_val - np.log10(sel['Unit cost'].values[len(x_cal):])
    ax2[int(count/9)][count%9].plot(cp, ucpred, 
                                       color='k',
                                       alpha = 0.6)
    ucpred2 = 10**(y_cal[-1] +  (-0.2912729419630219)* \
        (np.concatenate([np.array([x_cal[-1]]), x_val]) - x_cal[-1]))/sel['Unit cost'].values[0]
    errpred2 = 10**(y_cal[-1] +  (-0.2912729419630219)* \
        (x_val - x_cal[-1])) - sel['Unit cost'].values[len(x_cal):]
    errpred2 = y_cal[-1] +  (-0.2912729419630219)* \
        (x_val - x_cal[-1]) - np.log10(sel['Unit cost'].values[len(x_cal):])
    ax2[int(count/9)][count%9].plot(cp[len(x_cal)-1:], ucpred2, 
                                       color='g',
                                       alpha = 0.6)
    xlim = ax2[int(count/9)][count%9].get_xlim()
    ylim = ax2[int(count/9)][count%9].get_ylim()
    ax2[int(count/9)][count%9].plot(
        [cp[len(x_cal)-1],cp[len(x_cal)-1]],
        [0,10],'k', alpha=0.2, zorder=-30)
    ax2[int(count/9)][count%9].set_xlim(xlim)
    ax2[int(count/9)][count%9].set_ylim(ylim)
    ax2[int(count/9)][count%9].set_xticks([])
    ax2[int(count/9)][count%9].set_yticks([])
    ax2[int(count/9)][count%9].minorticks_off()
    # plt.pause(0.01)
    for axis in ['top','bottom','left','right']:
        ax2[int(count/9)][count%9].spines[axis].set_linewidth(0.1)
        ax3[int(count/9)][count%9].spines[axis].set_linewidth(0.1)
        ax2[int(count/9)][count%9].spines[axis].set_alpha(0.5)
        ax3[int(count/9)][count%9].spines[axis].set_alpha(0.5)
    sns.kdeplot(errpred, color='k', 
                ax=ax3[int(count/9)][count%9])
    sns.kdeplot(errpred2, color='g', 
                ax=ax3[int(count/9)][count%9])
    xlim = ax3[int(count/9)][count%9].get_xlim()
    ax3[int(count/9)][count%9].set_xlim((-max(np.abs(xlim)),max(np.abs(xlim))))
    ax3[int(count/9)][count%9].set_xticks([0],[])
    ax3[int(count/9)][count%9].set_yticks([])
    ax3[int(count/9)][count%9].set_ylabel('')
    if sum(errpred**2) > sum(errpred2**2):
        for axis in ['top','bottom','left','right']:
            ax3[int(count/9)][count%9].spines[axis].set_color('green')
            ax3[int(count/9)][count%9].spines[axis].set_linewidth(1.0)
        better += 1
    count += 1
ax3[int(count/9)][count%9].annotate(
    'The error of average technological learning rate is lower than'+\
    '\n'+'each technology learning rate for '+\
    str(better)+ ' ('+str(round(better/86,2))+'%) technologies', 
    (0.5,0.075), ha='center',
    xycoords='figure fraction')

ax2[-1][-2].axis('off')
ax2[-1][-1].axis('off')
ax3[-1][-2].axis('off')
ax3[-1][-1].axis('off')
fig2.suptitle('Predictions and observations')
fig3.suptitle('Unit cost error distributions')
ax2[0][0].annotate('Unit cost', (0.05,0.55), 
                   ha='center', rotation=90, 
                    xycoords='figure fraction')
ax2[0][0].annotate('Cumulative production', 
                   (0.5,0.1), ha='center',
                   xycoords='figure fraction')
axscatter.set_xlabel('Early learning rate')
axscatter.set_ylabel('Late learning rate')
legend_elements = [
    matplotlib.lines.Line2D([0],[0],color='k',
                            label='Technology-specific learning rate'),
    matplotlib.lines.Line2D([0],[0],color='g',
                            label='Mean technological learning rate')
]
fig3.legend(handles=legend_elements, ncol=2, loc='lower center')
legend_elements = [
    matplotlib.lines.Line2D([0],[0],color='firebrick', lw=0,
                            marker='o', markerfacecolor='None',
                            label='Observations'),
    matplotlib.lines.Line2D([0],[0],color='k',
                            label='Technology-specific learning rate'),
    matplotlib.lines.Line2D([0],[0],color='g',
                            label='Mean technological learning rate')
]
fig2.legend(handles=legend_elements, ncol=3, loc='lower center')
fig2.subplots_adjust(bottom=0.15)
fig3.subplots_adjust(bottom=0.15)

model = sm.OLS(LR_val, sm.add_constant(LR_cal))
result = model.fit()
print(result.summary())
axscatter.plot(np.linspace(-2,2,100), 
         result.params[0] + result.params[1] * np.linspace(-1,1,100),
         zorder = -1, label='Regression')
axscatter.plot([-2,2],[np.mean(LR_cal),np.mean(LR_cal)], label='Mean past learning rate')
axscatter.plot([-2,2],[np.mean(LR_val),np.mean(LR_val)], label='Mean future learning rate')
axscatter.axis('equal')
axscatter.set_xlim((-2,2))
axscatter.set_ylim((-2,2))
axscatter.annotate('R2 = ' + str(round(result.rsquared,2)) + \
             '\n N = ' + str(len(df['Tech'].unique())),
             (-1.5,1))
axscatter.annotate('late LR = ' + str(round(result.params[0],2)) + \
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
figscatter.legend()
plt.show()