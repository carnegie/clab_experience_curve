import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import cmcrameri
import matplotlib
from matplotlib import rc
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cmcrameri.cm.batlowS(np.linspace(0,1,100)))
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('ExpCurves.csv')

# df = df.loc[~(df['Tech'].str.contains('Nuclear'))]
print(df['Tech'].unique())
sectors = {'Energy':['Wind_Turbine_2_(Germany)', 'Fotovoltaica',
                    'Crude_Oil', 'Photovoltaics_2', 
                    'Onshore_Gas_Pipeline', 'Wind_Electricity',
                    'Photovoltaics_4','Geothermal_Electricity',
                    'Solar_Thermal','Nuclear_Electricity',
                    'Solar_Thermal_Electricity', 'Electric_Power',
                    'SCGT', 'Photovoltaics', 
                    'Solar_Water_Heaters', 'Wind_Turbine_(Denmark)',
                    'CCGT_Power','Nuclear_Power_(OECD)',
                    'CCGT_Electricity', 'Offshore_Gas_Pipeline',
                    'Wind_Power'], 
        'Chemicals':['Titanium_Sponge','CarbonDisulfide',
                    'Primary_Aluminum', 'Acrylonitrile',
                    'HydrofluoricAcid','PolyesterFiber',
                    'SodiumHydrosulfite', 'EthylAlcohol',
                    'Ethanol_2', 'Cyclohexane',
                    'Polyvinylchloride','PolyethyleneLD',
                    'Trichloroethane', 'Polypropylene',
                    'Pentaerythritol','Ethylene_2',
                    'VinylAcetate', 'CarbonBlack',
                    'Aniline', 'PhthalicAnhydride',
                    'Magnesium', 'MaleicAnhydride',
                    'TitaniumDioxide', 'Paraxylene',
                    'Ammonia', 'VinylChloride',
                    'Sorbitol', 'Styrene',
                    'Aluminum', 'Polystyrene',
                    'Phenol', 'BisphenolA',
                    'EthyleneGlycol', 'Methanol',
                    'PolyethyleneHD', 'Low_Density_Polyethylene',
                    'Urea', 'Sodium',
                    'Ethanolamine', 'SodiumChlorate',
                    'Primary_Magnesium', 'NeopreneRubber',
                    'Ethylene', 'AcrylicFiber',
                    'Formaldehyde', 'Benzene',
                    'Ethanol_(Brazil)', 'IsopropylAlcohol',
                    'Motor_Gasoline', 'Caprolactam'],
        'Hardware': ['Transistor', 'DRAM', 
                     'Hard_Disk_Drive', 'Laser_Diode'],
        'Consumer goods': ['Monochrome_Television', 'Automotive_(US)' ,
                           'Electric_Range', 'Free_Standing_Gas_Range'],
        'Food': ['Milk_(US)', 'Refined_Cane_Sugar',
                 'Wheat_(US)', 'Beer_(Japan)',
                 'Corn_(US)'],
        'Genomics':['Shotgun_Sanger_DNA_Sequencing', 'Capillary_DNA_Sequencing']
}
sectorsColor = {'Energy':'royalblue', 'Chemicals':'black',
               'Hardware':'red', 'Consumer goods':'forestgreen',
               'Food':'cyan', 'Genomics':'darkmagenta'}
sectorsinv = {v:k for k, vlist in sectors.items() for v in vlist}
df['Sector'] = [sectorsinv[tech] for tech in df['Tech']]

fraction = 1/2
points = True

LR_cal = []
LR_val = []
figscatter , axscatter = plt.subplots(figsize=(8,6))
fig2 , ax2 = plt.subplots(10,9, figsize=(13,7))
fig3 , ax3 = plt.subplots(10,9, figsize=(13,7))
fig4, ax4 = plt.subplots(10,9, figsize=(13,7))
count = 0
better = 0
df['LowerErrorMethod'] = ''
for tech in df['Tech'].unique():
    # read data for specific technology
    sel = df.loc[df['Tech'] == tech]
    x = np.log10(sel['Cumulative production'].values)
    y = np.log10(sel['Unit cost'].values)
    if points == True:
        # separate calibration and validation datasets based on number of points
        x_cal = x[:round(x.shape[0]*fraction)]
        x_val = x[round(x.shape[0]*fraction):]
        y_cal = y[:round(y.shape[0]*fraction)]
        y_val = y[round(y.shape[0]*fraction):]
    else:
        # separate calibration and validation datasets based on cumulative production range
        idxcal = np.where(np.array(x)<=x[0]+(x[-1]-x[0])*fraction)
        idxval = np.where(np.array(x)>x[0]+(x[-1]-x[0])*fraction)
        x_cal = x[idxcal]
        x_val = x[idxval]
        y_cal = y[idxcal]
        y_val = y[idxval]

    sel2 = df.loc[~(df['Tech'] == tech)]
    for tech2 in sel2['Tech'].unique():
        sel2_ = sel2.loc[sel2['Tech'] == tech2]
        x_ = np.log10(sel2_['Cumulative production'].values)
        y_ = np.log10(sel2_['Unit cost'].values)
        model = sm.OLS(y_, sm.add_constant(x_))
        result = model.fit()
        slopeall = result.params[1]

    # perform regression on both datasets separately
    model_cal = sm.OLS(y_cal, sm.add_constant(x_cal))
    result_cal = model_cal.fit()
    model_val = sm.OLS(y_val, sm.add_constant(x_val))
    result_val = model_val.fit()
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()
    LRall = result.params[1]
    axscatter.scatter(result_cal.params[1], result_val.params[1], 
                color = sectorsColor[sel['Sector'].values[0]],
                # color='tab:red', 
                # facecolor='None', 
                alpha = 0.4)
    LR_cal.append(result_cal.params[1])
    LR_val.append(result_val.params[1])
    if count in [81]:
        for x in range(2):
            ax2[int(count/9)][count%9].axis('off')
            ax3[int(count/9)][count%9].axis('off')
            ax4[int(count/9)][count%9].axis('off')
            count += 1
    uc = sel['Unit cost'].values #/sel['Unit cost'].values[0]
    cp = sel['Cumulative production'].values #/sel['Cumulative production'].values[0]
    ax2[int(count/9)][count%9].scatter(cp, uc, 
                                       marker='o', 
                                       color='firebrick',
                                       lw=0.5, 
                                       facecolor='None', s=2)
    ax2[int(count/9)][count%9].set_yscale('log', base=10)
    ax2[int(count/9)][count%9].set_xscale('log', base=10)
    ax4[int(count/9)][count%9].scatter(cp, uc, 
                                       marker='o', 
                                       color='firebrick',
                                       lw=0.5, 
                                       facecolor='None', s=2)
    ax4[int(count/9)][count%9].set_yscale('log', base=10)
    ax4[int(count/9)][count%9].set_xscale('log', base=10)
    ucpred = 10**(result_cal.params[0] + result_cal.params[1] * \
        np.concatenate([x_cal, x_val])) #/sel['Unit cost'].values[0]
    errpred = 10**(result_cal.params[0] + result_cal.params[1] * \
        x_val) - sel['Unit cost'].values[len(x_cal):]
    errpred = result_cal.params[0] + result_cal.params[1] * \
        x_val - np.log10(sel['Unit cost'].values[len(x_cal):])
    ucalpred = 10**(result_cal.params[0] + result_cal.params[1] * \
        (x_cal)) #/sel['Unit cost'].values[0]
    uvalpred = 10**(result_val.params[0] + result_val.params[1] * \
        (x_val)) #/sel['Unit cost'].values[0]
    ax2[int(count/9)][count%9].plot(cp, ucpred, 
                                       color='k',
                                       alpha = 0.6)
    ax4[int(count/9)][count%9].plot(10**x_cal, ucalpred, 
                                       color='k',
                                       alpha = 0.6)
    ax4[int(count/9)][count%9].plot(10**x_val, uvalpred, 
                                       color='b',
                                       alpha = 0.6)
    ucpred2 = 10**(y_cal[-1] +  slopeall* \
        (np.concatenate([np.array([x_cal[-1]]), x_val]) - x_cal[-1])) #/sel['Unit cost'].values[0]
    errpred2 = 10**(y_cal[-1] +  slopeall* \
        (x_val - x_cal[-1])) - sel['Unit cost'].values[len(x_cal):]
    errpred2 = y_cal[-1] +  slopeall* \
        (x_val - x_cal[-1]) - np.log10(sel['Unit cost'].values[len(x_cal):])
    ax2[int(count/9)][count%9].plot(cp[len(x_cal)-1:], ucpred2, 
                                       color='g',
                                       alpha = 0.6)
    xlim = ax2[int(count/9)][count%9].get_xlim()
    ylim = ax2[int(count/9)][count%9].get_ylim()
    ax2[int(count/9)][count%9].plot(
        [cp[len(x_cal)-1],cp[len(x_cal)-1]],
        [0,max(10**y)*1.2],'k', alpha=0.2, zorder=-30)
    ax4[int(count/9)][count%9].plot(
        [cp[len(x_cal)-1],cp[len(x_cal)-1]],
        [0,max(10**y)*1.2],'k', alpha=0.2, zorder=-30)
    ax2[int(count/9)][count%9].set_xlim(xlim)
    ax2[int(count/9)][count%9].set_ylim(ylim)
    ax2[int(count/9)][count%9].set_xticks([])
    ax2[int(count/9)][count%9].set_yticks([])
    ax2[int(count/9)][count%9].minorticks_off()
    ax4[int(count/9)][count%9].set_xlim(xlim)
    ax4[int(count/9)][count%9].set_ylim(ylim)
    ax4[int(count/9)][count%9].set_xticks([])
    ax4[int(count/9)][count%9].set_yticks([])
    ax4[int(count/9)][count%9].minorticks_off()
    # plt.pause(0.01)
    for axis in ['top','bottom','left','right']:
        ax2[int(count/9)][count%9].spines[axis].set_linewidth(0.1)
        ax3[int(count/9)][count%9].spines[axis].set_linewidth(0.1)
        ax4[int(count/9)][count%9].spines[axis].set_linewidth(0.1)
        ax2[int(count/9)][count%9].spines[axis].set_alpha(0.5)
        ax3[int(count/9)][count%9].spines[axis].set_alpha(0.5)
        ax4[int(count/9)][count%9].spines[axis].set_alpha(0.5)
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
        df.loc[df['Tech']==tech,'LowerErrorMethod']='Average slope'
    else:
        df.loc[df['Tech']==tech,'LowerErrorMethod']='Technology-specific'        
    ax2[int(count/9)][count%9].axis('off')
    count += 1

axscatter.plot([0,0],[-3,3], color='k', alpha=0.8, lw=0.2, zorder=-30)
axscatter.plot([-3,3],[0,0], color='k', alpha=0.8, lw=0.2, zorder=-30)
axscatter.set_xlim((-2,2))
axscatter.set_ylim((-2,2))
# plt.pause(15)
ax3[int(count/9)][count%9].annotate(
    'The sum of squared errors is lower \n using the average technological slope'+\
    '\n'+' for '+\
    str(better)+ ' ('+str(round(100*better/86))+'%) technologies', 
    (0.5,0.065), ha='center', fontsize=10,
    xycoords='figure fraction')

ax2[-1][-2].axis('off')
ax2[-1][-1].axis('off')
ax3[-1][-2].axis('off')
ax3[-1][-1].axis('off')
ax4[-1][-2].axis('off')
ax4[-1][-1].axis('off')
fig2.suptitle('Predictions and observations')
fig3.suptitle('Unit cost error distributions')
fig4.suptitle('First and second half regression lines')
ax2[0][0].annotate('Unit cost', (0.05,0.55), 
                   ha='center', rotation=90, 
                    xycoords='figure fraction')
ax2[0][0].annotate('Cumulative production', 
                   (0.5,0.1), ha='center',
                   xycoords='figure fraction')
ax4[0][0].annotate('Unit cost', (0.05,0.55), 
                   ha='center', rotation=90, 
                    xycoords='figure fraction')
ax4[0][0].annotate('Cumulative production', 
                   (0.5,0.1), ha='center',
                   xycoords='figure fraction')
axscatter.set_xlabel('First half slope')
axscatter.set_ylabel('Second half slope')
legend_elements = [
    matplotlib.lines.Line2D([0],[0],color='k',
                            label='Technology-specific slope'),
    matplotlib.lines.Line2D([0],[0],color='g',
                            label='Mean technological slope')
]
fig3.legend(handles=legend_elements, ncol=2, loc='lower center')
legend_elements = [
    matplotlib.lines.Line2D([0],[0],color='firebrick', lw=0,
                            marker='o', markerfacecolor='None',
                            label='Observations'),
    matplotlib.lines.Line2D([0],[0],color='k',
                            label='Technology-specific slope'),
    matplotlib.lines.Line2D([0],[0],color='g',
                            label='Mean technological slope')
]
fig2.legend(handles=legend_elements, ncol=3, loc='lower center')
legend_elements = [
    matplotlib.lines.Line2D([0],[0],color='firebrick', lw=0,
                            marker='o', markerfacecolor='None',
                            label='Observations'),
    matplotlib.lines.Line2D([0],[0],color='k',
                            label='First half regression line'),
    matplotlib.lines.Line2D([0],[0],color='b',
                            label='Second half regression line')
]
fig4.legend(handles=legend_elements, ncol=3, loc='lower center')
fig2.subplots_adjust(bottom=0.15)
fig3.subplots_adjust(bottom=0.15)
fig4.subplots_adjust(bottom=0.15)
axscatter.set_aspect('equal','box')

model = sm.OLS(LR_val, sm.add_constant(LR_cal))
result = model.fit()
print(result.summary())
axscatter.set_xlim((-2,2))
axscatter.set_ylim((-2,2))
axscatter.annotate('R2 = ' + str(round(result.rsquared,2)) + \
             '\n N = ' + str(len(df['Tech'].unique())),
             (-1.5,1))
# axscatter.annotate('LR2 = ' + str(round(result.params[0],2)) + \
#              '+ ' + str(round(result.params[1],2)) +'* LR1'+\
#                 '\n F-test p-value: '+str(round(result.f_pvalue,3)),
#                 (1.2,-1.25), ha='center')
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
print('Mean past and future slope: ')
print(np.mean(LR_cal), np.mean(LR_val))
print('R2 for regression and for prediction using mean past slope for all technologies: ')
print(1 - ssereg/ssetot)
print(1 - ssecal/ssetot)
print(1 - ssecal2/ssetot)
print(1 - sseval/ssetot)
print(ssetot, ssereg, (ssetot-ssereg), (ssereg) / (((ssetot-ssereg))/84))
# figscatter.legend(ncol=3, loc='lower center')
figscatter.subplots_adjust(bottom=0.1, top=0.9, left=0.0)
axscatter.set_xlim((-2,2))
axscatter.set_ylim((-2,2))
axscatter.set_aspect('equal', 'box')
legend_elements = [
    matplotlib.lines.Line2D([0],[0],lw=0,color='royalblue',marker='o', label='Energy'),
    matplotlib.lines.Line2D([0],[0],lw=0,color='black',marker='o', label='Chemicals'),
    matplotlib.lines.Line2D([0],[0],lw=0,color='red',marker='o', label='Hardware'),  
    matplotlib.lines.Line2D([0],[0],lw=0,color='forestgreen',marker='o', label='Consumer goods'),
    matplotlib.lines.Line2D([0],[0],lw=0,color='cyan',marker='o', label='Food'),
    matplotlib.lines.Line2D([0],[0],lw=0,color='darkmagenta',marker='o', label='Genomics')
]
figscatter.legend(handles=legend_elements, loc='center right', title='Sector')
# plt.show()

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9,6))
count = 0
for sector in df['Sector'].unique():
    avg, tech = 0, 0
    sel = df.loc[df['Sector']==sector]
    for t in sel['Tech'].unique():
        if sel.loc[sel['Tech']==t,'LowerErrorMethod'].values[0] == 'Technology-specific':
            tech += 1
        else:
            avg += 1
    ax[count//3][count%3].set_title(sector)
    ax[count//3][count%3].bar([0,1],
                              [tech, avg], color=sectorsColor[sector])
    count += 1
ax[1][0].set_xticks([0,1],['Technology-specific','Average slope'])
ax[1][1].set_xticks([0,1],['Technology-specific','Average slope'])
ax[1][2].set_xticks([0,1],['Technology-specific','Average slope'])
ax[0][0].set_ylabel('Number of technologies with lower MSE')
ax[1][0].set_ylabel('Number of technologies with lower MSE')
# plt.show()

# Monte Carlo sampling with replacement
np.random.seed(0)
TechStats = [df['Tech'].unique(), LR_cal, LR_val]
TechStats = pd.DataFrame(TechStats).transpose()
TechStats.columns = ['Tech', 'LR_cal', 'LR_val']

fig, ax = plt.subplots(1,1, figsize=(11,6))
R2 = {}
label = 'All Technologies (N = '+str(df['Tech'].nunique())+')'
R2[label] = []
for MN_N in range(1000):
    techs = []
    for MN_T in range(df['Tech'].nunique()):
        idx = np.random.randint(0, df['Tech'].nunique())
        techs.append(TechStats.iloc[idx].values.tolist())
    techs = pd.DataFrame(techs, columns=['Tech', 'LR_cal', 'LR_val'])
    model = sm.OLS(techs['LR_val'].values, sm.add_constant(techs['LR_cal'].values))
    result = model.fit()
    R2[label].append(result.rsquared)
for sector in sectors:
    sel = df.loc[df['Sector']==sector]
    label = sector+' (N = '+str(sel['Tech'].nunique())+')'
    R2[label] = []
    for MN_N in range(1000):
        techs = []
        for MN_T in range(sel['Tech'].nunique()):
            idx = np.random.randint(0, sel['Tech'].nunique())
            t = sel['Tech'].unique()[idx]
            techs.append(TechStats.loc[TechStats['Tech']==t].values.tolist()[0])
        techs = pd.DataFrame(techs, columns=['Tech', 'LR_cal', 'LR_val'])
        model = sm.OLS(techs['LR_val'].values, sm.add_constant(techs['LR_cal'].values))
        result = model.fit()
        R2[label].append(result.rsquared)
R2 = pd.DataFrame(R2)
R2.boxplot(ax=ax, whis=99)
ax.grid(False)
ax.set_ylabel('R2')
plt.subplots_adjust(right=0.95, left=0.05, bottom=0.1, top=0.95)
plt.show()

for sector in df['Sector'].unique():
    LR_cal = []
    LR_val = []
    figscatter , axscatter = plt.subplots(figsize=(8,6))
    count = 0
    better = 0
    for tech in df['Tech'].unique():
        if tech not in sectors[sector]:
            continue
        # read data for specific technology
        sel = df.loc[df['Tech'] == tech]
        x = np.log10(sel['Cumulative production'].values)
        y = np.log10(sel['Unit cost'].values)
        if points == True:
            # separate calibration and validation datasets based on number of points
            x_cal = x[:round(x.shape[0]*fraction)]
            x_val = x[round(x.shape[0]*fraction):]
            y_cal = y[:round(y.shape[0]*fraction)]
            y_val = y[round(y.shape[0]*fraction):]
        else:
            # separate calibration and validation datasets based on cumulative production range
            idxcal = np.where(np.array(x)<=x[0]+(x[-1]-x[0])*fraction)
            idxval = np.where(np.array(x)>x[0]+(x[-1]-x[0])*fraction)
            x_cal = x[idxcal]
            x_val = x[idxval]
            y_cal = y[idxcal]
            y_val = y[idxval]

        sel2 = df.loc[~(df['Tech'] == tech)]
        for tech2 in sel2['Tech'].unique():
            sel2_ = sel2.loc[sel2['Tech'] == tech2]
            x_ = np.log10(sel2_['Cumulative production'].values)
            y_ = np.log10(sel2_['Unit cost'].values)
            model = sm.OLS(y_, sm.add_constant(x_))
            result = model.fit()
            slopeall = result.params[1]

        # perform regression on both datasets separately
        model_cal = sm.OLS(y_cal, sm.add_constant(x_cal))
        result_cal = model_cal.fit()
        model_val = sm.OLS(y_val, sm.add_constant(x_val))
        result_val = model_val.fit()
        model = sm.OLS(y, sm.add_constant(x))
        result = model.fit()
        LRall = result.params[1]
        axscatter.scatter(result_cal.params[1], result_val.params[1], 
                    color = sectorsColor[sel['Sector'].values[0]],
                    # color='tab:red', 
                    # facecolor='None', 
                    alpha = 0.4)
        LR_cal.append(result_cal.params[1])
        LR_val.append(result_val.params[1])

    axscatter.plot([0,0],[-3,3], color='k', alpha=0.8, lw=0.2, zorder=-30)
    axscatter.plot([-3,3],[0,0], color='k', alpha=0.8, lw=0.2, zorder=-30)
    axscatter.set_xlim((-2,2))
    axscatter.set_ylim((-2,2))
    axscatter.set_xlabel('First half slope')
    axscatter.set_ylabel('Second half slope')
    axscatter.set_aspect('equal','box')
    print(LR_cal, LR_val)
    model = sm.OLS(LR_val, sm.add_constant(LR_cal))
    result = model.fit()
    print(result.summary())
    axscatter.set_xlim((-2,2))
    axscatter.set_ylim((-2,2))
    axscatter.annotate('R2 = ' + str(round(result.rsquared,2)) + \
                '\n N = ' + str(len(df.loc[df['Sector']==sector,'Tech'].unique())),
                (-1.5,1))
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
    print('Mean past and future slopes: ')
    print(np.mean(LR_cal), np.mean(LR_val))
    print('R2 for regression and for prediction using mean past slope for all technologies: ')
    print(1 - ssereg/ssetot)
    print(1 - ssecal/ssetot)
    print(1 - ssecal2/ssetot)
    print(1 - sseval/ssetot)
    print(ssetot, ssereg, (ssetot-ssereg), (ssereg) / (((ssetot-ssereg))/84))
    # figscatter.legend(ncol=3, loc='lower center')
    figscatter.subplots_adjust(bottom=0.1, top=0.9, left=0.0)
    axscatter.set_xlim((-2,2))
    axscatter.set_ylim((-2,2))
    axscatter.set_aspect('equal', 'box')
    legend_elements = [
        matplotlib.lines.Line2D([0],[0],lw=0,color='royalblue',marker='o', label='Energy'),
        matplotlib.lines.Line2D([0],[0],lw=0,color='black',marker='o', label='Chemicals'),
        matplotlib.lines.Line2D([0],[0],lw=0,color='red',marker='o', label='Hardware'),  
        matplotlib.lines.Line2D([0],[0],lw=0,color='forestgreen',marker='o', label='Consumer goods'),
        matplotlib.lines.Line2D([0],[0],lw=0,color='cyan',marker='o', label='Food'),
        matplotlib.lines.Line2D([0],[0],lw=0,color='darkmagenta',marker='o', label='Genomics')
    ]
    figscatter.legend(handles=legend_elements, loc='center right', title='Sector')
    plt.show()