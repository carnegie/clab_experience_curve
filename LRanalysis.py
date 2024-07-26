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
    sel = lexps.loc[lexps['Tech']==tech]

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
ax.set_xlabel('Past learning rate (%)')
ax.set_ylabel('Future learning rate (%)')
ax.plot([-200,200], [-200,200], 'k--', zorder=-10, lw=.75)
ax.set_xlim(-120,60)
ax.set_yticks(ax.get_xticks())
ax.set_ylim(-120,60)
ax.set_aspect('equal')
plt.tight_layout()

fig.savefig('figs' + os.path.sep + 'learning_past_future.png')

# print R2
print( 1 - (sum( (halfpoint['Future LEXP'].values - halfpoint['Past LEXP'].values)**2 )) / 
            (sum( (halfpoint['Future LEXP'].values - halfpoint['Future LEXP'].mean() )**2)))
print(sklearn.metrics.r2_score(halfpoint['Future LEXP'], halfpoint['Past LEXP']))
print(sklearn.metrics.r2_score(100 * (1 - 2**halfpoint['Future LEXP']), 
                               100 * (1 - 2**halfpoint['Past LEXP'])))
print(scipy.stats.pearsonr(halfpoint['Future LEXP'], halfpoint['Past LEXP']))

plt.show()

