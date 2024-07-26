import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils, distfit, os, scipy

sns.set_style("ticks")
sns.set_context("talk")
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300

first_diff = False

# read dataset from piecewise regression
df = pd.read_csv('IC' + first_diff*'_first_diff' + '.csv')

# select only the best model for each technology 
# using Bayesian Information Criterion
BIC = df.loc[df.groupby('Tech')['BIC'].idxmin()].reset_index()   

# add sectors and sector colors
BIC['Sector'] = [utils.sectorsinv[x] for x in BIC['Tech'].values]

BIC['Color'] = BIC['Sector'].map(utils.sectors_colors)

for i in range(1, 8):
    BIC['LEXP '+str(i)] = np.log2(1 - BIC['LR '+str(i)]/100)

# compute production increase between breakpoints
for i in range(1, 7):
    if i == 1:
        BIC['Prod diff '+str(i)] = BIC['Breakpoint '+str(i)] - \
                                        BIC['Initial production']
    else:
        BIC['Prod diff '+str(i)] = BIC['Breakpoint '+str(i)] - \
                                        BIC['Breakpoint '+str(i-1)]

# analyze breakpoints, learning rate, and learning exponent change

# define name of the columns
bcols = ['Initial production']
bcols.extend(['Breakpoint '+str(i) for i in range(1, 7)])
bcols.append('Final production')
lexpcols = ['LEXP '+str(i) for i in range(1, 8)]

# extract breaks and learning exponents
breaks = BIC[bcols]
lexps = BIC[lexpcols]

# get all the learning exponents and remove nan values
lexps_fl = lexps.values.flatten()
lexps_fl = lexps_fl[~np.isnan(lexps_fl)]
# perform normality test
print(scipy.stats.normaltest(lexps_fl))
# print mean and standard deviation of normal distribution
print(np.mean(lexps_fl), np.std(lexps_fl, ddof=1))

# empty lists to store results
blist = []
lexplist = []
acflist = []
sigmalist = []
lexpchangelist = []

break_dist = []
lexpchanges = []
colors = []
sectors = []
# iterate over breaks and learning exponents
for i in range(breaks.shape[0]):

    # store breakpoint distance
    [blist.append(x) \
         for x in breaks.loc[i,:].dropna().diff().values[1:]]

    #store learning exponent change
    [lexplist.append(x) \
        for x in lexps.loc[i,:].dropna().values]

    [break_dist.append(x) \
        for x in breaks.loc[i,:].dropna().diff().values[1:]]
    [lexpchanges.append(x) \
        for x in lexps.loc[i,:].dropna().diff().values[1:]]
    [colors.append(BIC.loc[i,'Color']) \
        for x in lexps.loc[i,:].dropna().values[1:]]
    [sectors.append(BIC.loc[i,'Sector']) \
        for x in lexps.loc[i,:].dropna().values]

    # store learning exponent change standard deviation
    if lexps.loc[i,:].dropna().diff().values[1:].shape[0]>1:
        lexpchangelist.append(\
                    np.std(lexps.loc[i,:].dropna().diff().values[1:]))

# create dataframe with learning exponent changes
lexplistdf = pd.DataFrame(lexplist, columns=['Learning exponent change'])

# extract all the learning exponent changes and remove nan values
lexps_fl = lexplistdf.values.flatten()
lexps_fl = lexps_fl[~np.isnan(lexps_fl)]

# perform normality test
print(scipy.stats.normaltest(lexps_fl))


breaks_lexp = pd.DataFrame({'Distance between breakpoints': 
                                        [10**x for x in break_dist], 
                                   'Learning exponent': 
                                        lexplist,
                                   'Sector': sectors})

sns.scatterplot(data=breaks_lexp, 
                x='Distance between breakpoints',
                y='Learning exponent',
                hue='Sector', 
                hue_order=utils.sectors_colors.keys(),
                palette=utils.sectors_colors.values(), 
                alpha=0.5)
plt.xscale('log')
plt.tight_layout()

sns.jointplot(data=breaks_lexp, 
                x='Distance between breakpoints',
                y='Learning exponent',
                hue='Sector', 
                hue_order=utils.sectors_colors.keys(),
                palette=utils.sectors_colors.values(), 
                alpha=0.5,
                height=8.1,
                marginal_kws=dict(log_scale=(True,False),
                                  multiple='stack',
                                  lw=.5,),
)
plt.gcf().axes[-1].set_xscale('linear')
# plt.gca().set_xlim(10**0, 10**5)
# plt.gca().set_ylim(-2, 2)
plt.tight_layout()
plt.subplots_adjust(top=1, bottom=0.1, right=1)
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                os.path.sep + 'Breakpoints_vs_LEXPchanges' + \
                    first_diff * '_first_diff' + '.png')


plt.figure()
sns.kdeplot(data=breaks_lexp,
            x='Distance between breakpoints',
            hue='Sector', 
            hue_order=utils.sectors_colors.keys(),
            palette=utils.sectors_colors.values(), 
            log_scale=True,
            multiple='stack',
            fill=True)
plt.tight_layout()


plt.figure()
sns.kdeplot(data=breaks_lexp,
            x='Learning exponent',
            hue='Sector', 
            hue_order=utils.sectors_colors.keys(),
            palette=utils.sectors_colors.values(), 
            multiple='stack',
            fill=True)
plt.tight_layout()

print('Standard deviation of learning exponent change:')
print('Considering all learning exponent changes equally: ', 
        np.std(lexplist))
print('Considering learning exponent changes for each technology' + 
        ' and then averaging or taking the median: ', 
        np.mean(lexpchangelist), np.median(lexpchangelist))

# calibrate distribution of distance between breakpoints
dfit = distfit.distfit()
print('Calibrating model for distance between breakpoints:')
dfit.fit_transform(np.array(blist).reshape(-1,1))
print(dfit.summary)
print('Best model for distance between breakpoints:')
print(dfit.summary.name[0], dfit.summary.params[0])
print(dfit.summary.name[1], dfit.summary.params[1])
print(dfit.summary.params[0])

print('Calibrating model for learning rate changes:')
dfit.fit_transform(np.array(lexplist).reshape(-1,1))
print(dfit.summary)

plt.show()