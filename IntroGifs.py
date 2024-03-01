import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, cmcrameri, os
import statsmodels.api as sm
import matplotlib.animation as animation
import seaborn as sns

matplotlib.rc('savefig', dpi=300)
sns.set_style('ticks')
matplotlib.rc('font',
                **{'family':'sans-serif',
                   'sans-serif':'Helvetica'})

# read data
DF = pd.read_csv('ExpCurves.csv')

# some techs have multiple data per year
# and the year is not an integer:
# assume all data collected in the same year 
# are avaialable for prediction at the end of the year
DF['Year'] = [int(x) for x in DF['Year'].values]

cmap = cmcrameri.cm.hawaii
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

### new func to iterate over all techs
def animate(yy, *fargs):
    ## read dataframe containing technology data
    df = fargs[0]

    # unpack information in dataframe
    x, y, year, = \
        np.log10(df[['Cumulative production']].values), \
        np.log10(df[['Unit cost']].values), \
        df[['Year']].values
    
    # save position of axes
    pos = []
    for axis in ax:
        pos.append(axis.get_position())

    # in the first year (set axes limits and labels)
    if yy == df['Year'].values[0]:
        
        # compute regression model over all data
        m = sm.OLS(y, sm.add_constant(x))
        r = m.fit()

        # plot regression line and all data points
        ax[1].plot(10**x, 10**r.predict(sm.add_constant(x)), 
                    color = 'firebrick', alpha=1.0,
                    zorder=1, lw=2)
        ax[1].scatter(10**x, 10**y, color='k')
        # add star representing learning exponent
        # computed over all data points
        ax[0].scatter(r.params[1], r.params[1], 
                    color='gold', edgecolor='k',
                    marker='*', s=200)

        # add title to figure
        title = t + ' (' + \
                    str(int(df['Year'].values[0])) + '-' +\
                    str(int(df['Year'].values[-1])) + ')'
        if df['Tech'].values[0] == 'Fotovoltaica':
            title = 'Solar PV CAPEX [$/W]'
            title = title + ' (' + \
                        str(int(df['Year'].values[0])) + '-' +\
                        str(int(df['Year'].values[-1])) + ')'
        fig.suptitle(title)

        # add axes and identity line
        ax[0].axhline(0, color='silver', zorder=-1, lw=.5)
        ax[0].axvline(0, color='silver', zorder=-1, lw=.5)
        ax[0].plot([-3,3],[-3,3], color='k', 
                zorder=-1, lw=1.5, linestyle='--')
        # set axes limits and aspect ratio 
        # for observed and future learning axes
        ax[0].set_xlim(r.params[1]-0.75,r.params[1]+0.75)
        ax[0].set_ylim(r.params[1]-0.75,r.params[1]+0.75)
        ax[0].set_aspect('equal')
        # set log scale and axes limits
        # for data points and regression line
        ax[1].set_xscale('log', base=10)
        ax[1].set_yscale('log', base=10)
        ax[1].set_ylim(min(10**y)*0.3, max(10**y)*3)
        ax[1].set_xlim(min(10**x)*0.3, max(10**x)*3)
        # set axes labels
        ax[0].set_xlabel('Observed learning exponent')
        ax[1].set_xlabel('Cumulative production')
        ax[0].set_ylabel('Future learning exponent')
        ax[1].set_ylabel('Unit cost')
          
    # during the second year available
    # remove regression lines and data points
    # from the axes
    elif yy == df['Year'].values[1]:
        for l in ax[1].lines:
            l.remove()
        for c in ax[1].collections:
            c.remove()

    # in any year after the first
    # plot available and future data points
    # plot current and future regression line
    # add scatter of observed and future learning exponents
    if yy > df['Year'].values[0]:

        # make previous lines transparent
        for ln in ax[1].get_lines():
            ln.set_alpha(0.05)

        # select data available, build regression 
        # model and fit it to available data
        x_cal, y_cal = x[year<=yy], y[year<=yy]
        model = sm.OLS(y_cal, sm.add_constant(x_cal))
        result_cal = model.fit()

        # select future data, build regression 
        # model and fit it to future data
        x_val, y_val = x[year>=yy], y[year>=yy]
        model = sm.OLS(y_val, sm.add_constant(x_val))
        result_val = model.fit()

        # plot available and future regression lines
        ax[1].plot(10**x_cal, 
                   10**result_cal.predict(sm.add_constant(x_cal)),
                    color = cmap(len(y_cal)/len(y)), alpha=1.0,
                    zorder=1, lw=2)
        ax[1].plot(10**x_val, 
                    10**result_val.predict(sm.add_constant(x_val)),
                    color = cmap(len(y_cal)/len(y)), alpha=1.0,
                    zorder=1, lw=2)

        # plot scatter of available data points
        ax[1].scatter(10**x_cal, 10**y_cal,
                    zorder=-1, color='k',
                    edgecolor='k')
        # plot scatter of data points from current year
        ax[1].scatter(10**x_cal[-1], 10**y_cal[-1],
                    zorder=-1, 
                    color=cmap(len(y_cal)/len(y)),
                    edgecolor='r')
        # plot scatter of future data points
        ax[1].scatter(10**x_val, 10**y_val,
                    zorder=-1, color='silver',
                    alpha=0.3,
                    edgecolor='k')

        # plot scatter of observed and future learning exponents
        ax[0].scatter(result_cal.params[1], 
                      result_val.params[1], 
                        color=cmap(len(y_cal)/len(y))) 

        # keep position of axes fixed        
        for axis in enumerate(ax):
            axis[1].set_position(pos[axis[0]])

### loop over all technologies
for t in DF['Tech'].unique():
    print(t)

    # select only data from technology of interest
    df = DF.loc[DF['Tech']==t]

    # close previous figures
    plt.close('all')

    # create figure and colorbar
    fig, ax = plt.subplots(1,2,figsize=(9,5), 
                        layout='constrained'
                        )
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ticks = [0,1], 
                            ax = ax,
                            location='bottom',
                            shrink=0.5)
    cbar.set_ticklabels([int(df['Year'].values[1]),
                        int(df['Year'].values[-1]-2)])
    
    # create animation
    ani = animation.FuncAnimation(fig, animate, 
            fargs=[df], repeat=True,
            frames=df['Year'].values[:-1], 
            interval=200)

    # save animation as a gif
    writer = animation.PillowWriter(fps=5,
                metadata=dict(artist='Me'),)
    if not os.path.exists('figs' + os.path.sep + 'gifs'):
        os.makedirs('figs' + os.path.sep + 'gifs')
    ani.save('figs' + os.path.sep + 'gifs'+ 
             os.path.sep + t +'.gif', writer=writer)

plt.close('all')