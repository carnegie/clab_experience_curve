import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri, os, utils

# set figure parameters
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.sans-serif'] = 'Helvetica'
sns.set_context('talk')
sns.set_style('ticks')

# set colormap
cmap = cmcrameri.cm.hawaii

# select techs to be plotted
df = pd.read_csv('ExpCurves.csv')

# iterate over selected techs
for tech in df['Tech'].unique():

    fig, ax = plt.subplots(1,2, figsize=(12,6))

    # read data and drop nan values
    df = pd.read_csv('expCurveData' 
                     + os.path.sep
                     + tech + '.csv').dropna()

    utils.plot_cost_prod_learning_dynamics(df=df, tech=tech, 
                                           fig=fig, ax=ax,
                                           cmap=cmap,)

# plot learning rate dynamics for solar pv, wind, and li-ion battery

# create figure
fig, ax = plt.subplots(3, 2, figsize=(12,12))

# define technology
tech = 'Solar PV'
# read data and drop nan values
df = pd.read_csv('solar_pv_PCDB_IEA_IRENA.csv').dropna()
# plot data
utils.plot_cost_prod_learning_dynamics(df=df, tech=tech, 
                                 fig=fig, ax=ax[0,:],
                                 cmap=cmap,
                                 cbar_kws = {'loc':[0.9, 0.25, 0.01, 0.5],
                                             'orientation': 'vertical'},
                                 savefig=False)

# define technology
tech = 'Wind power'
# read data and drop nan values
df = pd.read_csv('wind_Bolinger_IEA_IRENA.csv').dropna()
# plot data
utils.plot_cost_prod_learning_dynamics(df=df, tech=tech,
                                 fig=fig, ax=ax[1,:],
                                 cmap=cmap,
                                 time_range=[1977,2022],
                                 savefig=False)

# define technology
tech = 'Li-ion battery'
# read original li-ion battery data 
filename = ('..' + os.path.sep
            + 'MicahTrancik' + os.path.sep
            + 'LiIonDataSeries_represonly_withcover.xlsx')
try: 
    price = pd.read_excel(filename,
        sheet_name='RepreSeries_Price_All_Cells')
except FileNotFoundError:
    print('File not found. Please download the Lihium-ion battery data from '
          'the Harvard Dataverse repository at '
          'https://doi.org/10.7910/DVN/9FEJ7C')
    price = None
if price is not None:
    price = price[['IndependentAxisData','DependentAxisData']]
    price.columns = ['Time (Year)','Unit cost (2018 USD/kWh)']
    prod = pd.read_excel(filename, 
                        sheet_name='RepreSeries_MarketSize_All_MWh')
    prod = prod[['IndependentAxisData','DependentAxisData']]
    prod.columns = ['Time (Year)','Cumulative production (MWh)']
    prod = prod[['Time (Year)','Cumulative production (MWh)']]

    df = pd.merge(price, prod, on='Time (Year)').reset_index(drop=True)
    df['Production (MWh)'] = df['Cumulative production (MWh)'].diff()
    df = df[['Unit cost (2018 USD/kWh)', 'Time (Year)', 
            'Production (MWh)','Cumulative production (MWh)']]

    # plot data
    utils.plot_cost_prod_learning_dynamics(df=df, tech=tech, fig=fig, 
                                        ax=ax[2,:],
                                        time_range=[1977,2022], 
                                        cmap=cmap,
                                        savefig=False)

    # adjust figure spacing
    fig.subplots_adjust(left=0.1, right=0.95,
                        top=0.925, bottom=0.1,
                        hspace=0.65, wspace=0.1)

    # adjust ticks colorbar
    fig.axes[-1]._colorbar.set_ticks([1977,1980,1990,2000,
                                      2010,2020,2022])
    fig.axes[-1]._colorbar.set_ticklabels([1977,1980,1990,2000,
                                           2010,2020,2022])

    # set equal aspect maintaining box size
    ax[0][0].set_aspect('equal', adjustable='datalim')
    ax[1][0].set_aspect('equal', adjustable='datalim')
    ax[2][0].set_aspect('equal', adjustable='datalim')

    # save figure
    if not os.path.exists('figs'):
        os.makedirs('figs')
    plt.savefig('figs' + os.path.sep + 'learningRateDynamics.png')
    plt.savefig('figs' + os.path.sep + 'learningRateDynamics.pdf')

plt.show()

plt.close()