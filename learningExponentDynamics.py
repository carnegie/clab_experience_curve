import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib, cmcrameri, os
import statsmodels.api as sm

# set figure parameters
matplotlib.rc('savefig', dpi=300)
sns.set_palette([cmcrameri.cm.batlowS(x) for x in range(10)])
sns.set_palette('colorblind')
sns.set_context('talk')
sns.set_style('whitegrid')
matplotlib.rc('font', family='Helvetica')


# set colormap
cmap = cmcrameri.cm.hawaii

# select techs to be plotted
selTechs = ['Photovoltaics_2']

# set column names 
cols = ['Unit cost', 'Year',
        'Production','Cumulative production']

# iterate over selected techs
for tech in selTechs:

    # read data and drop nan values
    df = pd.read_csv('expCurveData/' + tech + '.csv').dropna()

    # rename columns
    df.columns = [ a + ' ('+b.split('(')[1]
                   for a,b in zip(cols,df.columns)]

    # convert data to float
    for col in df.columns:
        df[col] = [float(x) for x in df[col].values]
        
    # some techs have multiple data per year
    # and the year is not an integer:
    # assume all data collected in the same year 
    # are avaialable for prediction at the end of the year
    for col in [df.columns[1]]:
        df[col] = [int(x) for x in df[col].values]
    
    # create figure
    fig, ax = plt.subplots(1,2, figsize=(14,6))
    # scatter unit cost vs cumulative production
    ax[0].scatter(df[df.columns[3]].values, 
                  df[df.columns[0]].values, color='k')
    # set log-log scale and label axes
    ax[0].set_xscale('log', base=10)
    ax[0].set_yscale('log', base=10)
    ax[0].set_xlabel(df.columns[3])
    ax[0].set_ylabel(df.columns[0])

    # add star representing technology-specific learning exponent
    model = sm.OLS(np.log10(df[df.columns[0]].values),
                     sm.add_constant(np.log10(df[df.columns[3]].values)))
    res = model.fit()
    ax[1].scatter(res.params[1], res.params[1], 
                  color='gold', marker='*', 
                  s=200, edgecolor='k')
    ax[1].set_xlabel('Learning exponent - Calibration')
    ax[1].set_ylabel('Learning exponent - Test')

    # diffc = np.diff(np.log10(df[df.columns[0]].values))
    # diffp = np.diff(np.log10(df[df.columns[3]].values))
    # lexp = sum([a*b for a,b in zip(diffc,diffp)])/\
    #             sum([a**2 for a in diffp])
    
    # ax[1].scatter(lexp, lexp, color='gold', marker='x',
    #                 s=200, edgecolor='k')


    # set norm for colormap
    norm = matplotlib.colors.Normalize(
            vmin=df[df.columns[1]].unique()[1],
            vmax=df[df.columns[1]].unique()[-1])

    # iterate over all years from 2nd to second to last
    for i in range(df[df.columns[1]].unique()[1],
                   df[df.columns[1]].unique()[-1]):
        
        # split data into calibration and validation sets
        cal = df[df[df.columns[1]]<=i]
        val = df[df[df.columns[1]]>=i]

        # fit regression model to calibration  and validation data
        modelcal = sm.OLS(np.log10(cal[cal.columns[0]].values),
                            sm.add_constant(
                                np.log10(cal[cal.columns[3]].values)))
        modelval = sm.OLS(np.log10(val[val.columns[0]].values),
                            sm.add_constant(
                                np.log10(val[val.columns[3]].values)))
        rescal = modelcal.fit()
        resval = modelval.fit()

        # # alternative estimate
        # diffcalc = np.diff(np.log10(cal[cal.columns[0]].values))
        # diffcalp = np.diff(np.log10(cal[cal.columns[3]].values))
        # diffvalc = np.diff(np.log10(val[val.columns[0]].values))
        # diffvalp = np.diff(np.log10(val[val.columns[3]].values))

        # lexpcal = sum([a*b for a,b in zip(diffcalc,diffcalp)])/\
        #             sum([a**2 for a in diffcalp])
        # lexpval = sum([a*b for a,b in zip(diffvalc,diffvalp)])/\
        #             sum([a**2 for a in diffvalp])

        # add lines to scatter plot
        ax[0].plot(cal[cal.columns[3]].values,
                     10**rescal.predict(
                          sm.add_constant(
                            np.log10(cal[cal.columns[3]].values))),
                     color=cmap(norm(i)),
                     alpha=0.5, zorder=1, lw=1)
        ax[0].plot(val[val.columns[3]].values,
                        10**resval.predict(
                            sm.add_constant(
                                np.log10(val[val.columns[3]].values))),
                        color=cmap(norm(i)),
                        alpha=0.5, zorder=1, lw=1)
        
        # add points to learning exponent dynamics
        ax[1].scatter(rescal.params[1], resval.params[1],
                        color=cmap(norm(i)), zorder=1)
        # ax[1].scatter(lexpcal, lexpval,
        #                 color=cmap(norm(i)), zorder=1, marker='x')
        
    
    # add colorbar
    smap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    smap.set_array([])
    fig.colorbar(smap, ax=ax[1], label='Year')

    # add identity line to learning exponent dynamics
    lims = [ax[1].get_xlim(), ax[1].get_ylim()]
    lims = [min(lims[0][0],lims[1][0]), max(lims[0][1],lims[1][1])]
    ax[1].plot([-2,2],[-2,2], color='k', ls='--', lw=1, zorder=-10)
    ax[1].set_xlim(lims)
    ax[1].set_ylim(lims)
    ticks = [round(x,3) for x in 
             np.arange(int(lims[0]*10-1.2)*0.1,
                       int(lims[1]*10+1.2)*0.1,
                       0.1)]
    ax[1].set_xticks(ticks)
    ax[1].set_yticks(ticks)
    ax[1].set_aspect('equal')
    
    # add title and adjust spacing
    fig.suptitle(tech.split('_')[0] + ' (' + 
                 df[df.columns[1]].values[0].astype(str) +
                    '-' +
                df[df.columns[1]].values[-1].astype(str) + ')')
    fig.subplots_adjust(wspace=0.25, 
                        left=0.075, right=0.95,
                        bottom=0.15, top=0.85)
    # # annotate panels
    ax[0].annotate('a', xy=(0.05, 1.05),
                    xycoords='axes fraction',
                    ha='center', va='center')
    ax[1].annotate('b', xy=(0.05, 1.05),
                    xycoords='axes fraction',
                    ha='center', va='center')
    

    # save figure
    if not(os.path.exists('figs' + 
                          os.path.sep + 
                          'learningExponentDynamics')):
        os.makedirs('figs' + 
                    os.path.sep +
                      'learningExponentDynamics')

    fig.savefig('figs' + os.path.sep + 
                'learningExponentDynamics' +
                 os.path.sep + tech + '.png')
    fig.savefig('figs' + os.path.sep + 
                'learningExponentDynamics' +
                 os.path.sep + tech + '.eps')



plt.show()
        

