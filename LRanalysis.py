import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import scipy, utils, os
import statsmodels.api as sm

plt.rcParams['savefig.dpi'] = 300
sns.set_style('ticks')
sns.set_context('talk')
plt.rcParams['font.sans-serif'] = 'Helvetica'

# Load the data
df = pd.read_csv('ExpCurves.csv')

lexps = []
for tech in df['Tech'].unique():

    # select only the technology of interest
    sel = df.loc[df['Tech']==tech]

    # compute past and future learning exponents
    for i in range(1, sel.shape[0] - 1):

        lexp_past = utils.computeSlopeLafond(sel.iloc[:i+1])
        lexp_future = utils.computeSlopeLafond(sel.iloc[i:])

        # append data to list
        lexps.append([tech, lexp_past, lexp_future, 
                    (i+1)/sel.shape[0], sel.shape[0], 
                    np.log10(sel['Cumulative production'].values[i]),
                    np.log10(sel['Cumulative production'].values[0]),
                    np.log10(sel['Cumulative production'].values[-1]),
                    utils.sectorsinv[tech]])

# convert list to dataframe
lexps = pd.DataFrame(lexps, columns=['Tech', 'Past LEXP', 'Future LEXP', 
                                 'Fraction of points', 'Number of points',
                                 'Log10 Cumulative Production', 
                                 'Log10 Cumulative Production Min', 
                                 'Log10 Cumulative Production Max',
                                 'Sector'])

# create a dataframe selecting only the index for the half point
halfpoint = []
for tech in lexps['Tech'].unique():

    # select only the technology of interest
    sel = lexps.loc[lexps['Tech']==tech].copy()

    # compute the distance from the mid point
    sel['Dist'] = (sel['Fraction of points'] - 0.5)**2

    # append data to list
    halfpoint.append(sel.loc[sel['Dist']==sel['Dist'].min(), 
                             ['Tech', 'Number of points', 
                              'Past LEXP', 'Future LEXP','Sector']].values[0])

# convert list to dataframe
halfpoint = pd.DataFrame(halfpoint, columns=['Tech', 
                                             'Number of points', 
                                             'Past LEXP', 
                                             'Future LEXP','Sector'])

halfpoint['Color'] = [utils.sectors_colors[x] \
                        for x in halfpoint['Sector']]

# create a scatter plot
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(100*(1-2**halfpoint['Past LEXP']), 
            100*(1-2**halfpoint['Future LEXP']),
            facecolor="None", alpha=0.75,
            edgecolor=halfpoint['Color'])

# add text annotations
for tech in halfpoint['Tech'].unique():

    sel = halfpoint.loc[halfpoint['Tech']==tech].copy()

    if 100*(1-2**sel['Past LEXP'].values[0])<-40 or\
        100*(1-2**sel['Future LEXP'].values[0])>60:
        print(sel)

    if tech in ['Wind_Electricity', 'Fotovoltaica', 
             'DRAM','Hard_Disk_Drive','Transistor','Laser_Diode',
             'Solar_Water_Heaters','Nuclear_Electricity'
             ]:
        if tech == 'Fotovoltaica':
            tech = 'Solar PV Electricity'
            xytext = (40, -20)
        elif tech == 'DRAM':
            xytext = (30, 40)
        elif tech == 'Hard_Disk_Drive':
            xytext = (100*(1 - 2**sel['Past LEXP'].values[0]),0)
        elif tech == 'Transistor':
            xytext = (40,55)
        elif tech == 'Laser_Diode':
            xytext = (100*(1 - 2**sel['Past LEXP'].values[0]),55)
        elif tech == 'Solar_Water_Heaters':
            xytext = (100*(1 - 2**sel['Past LEXP'].values[0]),-30)
        elif tech == 'Wind_Electricity':
            xytext = (-20,50)
        elif tech == 'Nuclear_Electricity':
            xytext = (-10, -35)
            
        ax.annotate(tech.replace('_',' '),
						xy=(100*(1-2**sel['Past LEXP'].values[0]),
						100*(1-2**sel['Future LEXP'].values[0])),
                        xytext = xytext,
						textcoords='data',
						ha='center',
						va='center',
						fontsize=14,
                        arrowprops=dict(arrowstyle='->',
                                        color='k',
                                        shrinkA=5,
                                        shrinkB=5,
                                        lw=0.5))
        


ax.set_xlabel('Past learning rate (%)')
ax.set_ylabel('Future learning rate (%)')
ax.plot([-200,200], [-200,200], 'k--', zorder=-10, lw=.75)
ax.set_xlim(-40,60)
ax.set_yticks(ax.get_xticks())
ax.set_ylim(-40,60)
ax.set_aspect('equal')
plt.tight_layout()
plt.subplots_adjust(top=0.95, right=0.95, bottom=0.1)

fig.savefig('figs' + os.path.sep + 'learning_past_future.png')

# print R2
print("Pearson's correlation coefficient: ",
      scipy.stats.pearsonr(halfpoint['Future LEXP'], halfpoint['Past LEXP']),
        scipy.stats.pearsonr(halfpoint['Future LEXP'].values, 
                             halfpoint['Past LEXP'].values).confidence_interval(0.95))

# order dataset by number of points
halfpoint = halfpoint.sort_values('Number of points')

print("Pearson's correlation coefficient for the longest 44 data series: ",
        scipy.stats.pearsonr(halfpoint['Future LEXP'].values[-44:], 
                             halfpoint['Past LEXP'].values[-44:]),
        scipy.stats.pearsonr(halfpoint['Future LEXP'].values[-44:], 
                             halfpoint['Past LEXP'].values[-44:]).confidence_interval(0.95))  
print("Pearson's correlation coefficient for the shortest 43 data series: ",
        scipy.stats.pearsonr(halfpoint['Future LEXP'].values[:43], 
                             halfpoint['Past LEXP'].values[:43]),
        scipy.stats.pearsonr(halfpoint['Future LEXP'].values[:43], 
                             halfpoint['Past LEXP'].values[:43]).confidence_interval(0.95))

# by sector
print('\n')
for s in halfpoint['Sector'].unique():
    sel = halfpoint.loc[halfpoint['Sector']==s]
    print("Pearson's correlation coefficient for the ", s," sector: ",
        scipy.stats.pearsonr(sel['Future LEXP'].values, 
                             sel['Past LEXP'].values),
        scipy.stats.pearsonr(sel['Future LEXP'].values, 
                             sel['Past LEXP'].values).confidence_interval(0.95))  

