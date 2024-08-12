import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import utils, distfit, os, scipy
import statsmodels.api as sm

sns.set_style("ticks")
sns.set_context("talk")
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300

# read dataset from piecewise regression
df = pd.read_csv('IC.csv')

# select only the best model for each technology 
# using Bayesian Information Criterion
BIC = df.loc[df.groupby('Tech')['BIC'].idxmin()].reset_index()   

# add sectors and sector colors
BIC['Sector'] = [utils.sectorsinv[x] for x in BIC['Tech'].values]
BIC['Color'] = BIC['Sector'].map(utils.sectors_colors)

# check if some sectors are more prone to breakpoints using the chi square test
# create a contingency table
print('\n\n\n')
print("Checking if some sectors are more prone to breakpoints ...")
BICC = BIC.copy()
contingency = pd.crosstab(BICC['Sector'], BICC['n_breaks'])
print(scipy.stats.chi2_contingency(contingency))
print("Checking if some sectors are more prone to breakpoints " + 
      "(removing hardware)...")
BICC = BIC.loc[~BIC['Sector'].isin(['Hardware'])].copy()
contingency = pd.crosstab(BICC['Sector'], BICC['n_breaks'])
print(scipy.stats.chi2_contingency(contingency))

plt.figure(figsize=(9,6))
sns.scatterplot(data=BIC, x='n_breaks', y='Number of observations',
                hue='Sector', palette=utils.sectors_colors.values(),
                alpha=0.5)
print('\n\n\n')
print('Correlation between number of breakpoints and number of observations')
print(scipy.stats.pearsonr(BIC['n_breaks'], BIC['Number of observations']))
plt.xlabel('Number of breakpoints')
plt.tight_layout()


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

# is there correlation between successive learning exponents? No
print('\n\n\n')
print('Examining correlation between successive learning exponents..')
plt.figure()
lr_corr = []
for x in range(len(lexps.columns)-1):
    s = lexps.loc[:,[lexps.columns[i] for i in range(x,x+2)]].dropna()
    plt.scatter(s.iloc[:,0], s.iloc[:,1], 
                color='None', edgecolor='k', alpha=0.5)
    if s.shape[0] > 2:
        print(scipy.stats.pearsonr(s.iloc[:,0], s.iloc[:,1]),
            scipy.stats.pearsonr(s.iloc[:,0], s.iloc[:,1]).confidence_interval(0.95))

    [lr_corr.append([x,y]) for x,y in zip(s.iloc[:,0], s.iloc[:,1])]
print('\n\nExamining correlation between successive ' + 
      'learning exponents using linear regression.. ')
print(sm.OLS([x[1] for x in lr_corr], \
             sm.add_constant([x[0] for x in lr_corr]))\
                .fit().summary())
print('Correlation between successive learning exponents:')
print(scipy.stats.pearsonr([x[1] for x in lr_corr], 
                           [x[0] for x in lr_corr]),
      scipy.stats.pearsonr([x[1] for x in lr_corr], 
                           [x[0] for x in lr_corr]).confidence_interval(0.95))
plt.xlabel('Learning exponent at time t')
plt.ylabel('Learning exponent at time t+1')
plt.tight_layout()

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
    [break_dist.append(x) \
        for x in breaks.loc[i,:].dropna().diff().values[1:]]

    #store learning exponents
    [lexplist.append(x) \
        for x in lexps.loc[i,:].dropna().values]

    # store learning exponent changes
    [lexpchanges.append(x) \
        for x in lexps.loc[i,:].dropna().diff().values[1:]]
    
    # store colors and sectors
    [colors.append(BIC.loc[i,'Color']) \
        for x in lexps.loc[i,:].dropna().values[1:]]
    [sectors.append(BIC.loc[i,'Sector']) \
        for x in lexps.loc[i,:].dropna().values]

# examine learning exponent and distance between breakpoints

# create dataframe with learning exponent and distance between breakpoints
breaks_lexp = pd.DataFrame({'Distance between breakpoints': 
                                        [10**x for x in break_dist], 
                                   'Learning exponent': 
                                        lexplist,
                                   'Sector': sectors})

# create a scatter plot
sns.jointplot(data=breaks_lexp, 
                x='Distance between breakpoints',
                y='Learning exponent',
                hue='Sector', 
                hue_order=utils.sectors_colors.keys(),
                palette=utils.sectors_colors.values(), 
                alpha=0.5,
                height=7.25,
                marginal_kws=dict(log_scale=(True,False),
                                  multiple='stack',
                                  lw=.5,),
)
plt.gcf().axes[-1].set_xscale('linear')
plt.gca().set_ylim(-2, 2)
plt.tight_layout()
plt.subplots_adjust(top=1, bottom=0.1, right=1)
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                os.path.sep + 'Breakpoints_vs_LEXPchanges' + '.png')
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                os.path.sep + 'Breakpoints_vs_LEXPchanges' + '.pdf')


## fit probability distribution for distance between breakpoints

plt.figure()
plt.ecdf(break_dist, label='Breakpoint distance')
plt.xlabel('Distance between breakpoints (log10)')
plt.ylabel('ECDF')
plt.tight_layout()

print('\n\n\n')
print('Calibrating model for distance between breakpoints:')
summary_breaks = \
    utils.fit_probability_dist(np.array(break_dist).reshape(-1,1))

print('Calibrating model for distance between breakpoints' + \
        ' including min distance constraint:')
summary_breaks_constr = \
    utils.fit_probability_dist(np.array(break_dist).reshape(-1,1), 
                                floc=np.log10(2))
print(summary_breaks)
print(summary_breaks_constr)

# get exponential distribution parameters
params_expon = summary_breaks_constr\
                    .loc[summary_breaks_constr['Distribution']=='expon',
                         ['Param1', 'Param2']].values[0]\

# create figure
plt.figure(figsize=(9,6))
sns.kdeplot(data=breaks_lexp,
            x='Distance between breakpoints',
            hue='Sector', 
            hue_order=utils.sectors_colors.keys(),
            palette=utils.sectors_colors.values(), 
            log_scale=True,
            multiple='stack',
            fill=True,
            )     
plt.plot(10**np.linspace(-1, 5, 1000),
             scipy.stats.expon.pdf(np.linspace(-1, 5, 1000),
                                  *params_expon, ),
                color='k',
                lw=2)

# add legend and labels
legend = plt.gca().get_legend()
handles = legend.legendHandles
handles.append(plt.Line2D([0], [0], color='k', lw=2))
labels = [x.get_text() for x in legend.get_texts()]
labels.append('Exponential distribution fit')
plt.legend(handles=handles, labels=labels, loc='best')       
plt.tight_layout()
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                os.path.sep + 'Breakpoints_fitting' + '.png')
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                os.path.sep + 'Breakpoints_fitting' + '.pdf')


## fit probability distribution for learning exponent changes

# create dataframe with learning exponent changes
lexplistdf = pd.DataFrame(lexpchanges, columns=['Learning exponent change'])

# extract all the learning exponent changes and remove nan values
lexpchanges_fl = np.array(lexpchanges).flatten()
lexpchanges_fl = lexpchanges_fl[~np.isnan(lexpchanges_fl)]

# perform normality test
print('\n\n\n')
print('Normality test for learning exponent changes:')
print(scipy.stats.normaltest(lexpchanges_fl))

# get all the learning exponents and remove nan values
lexps_fl = lexps.values.flatten()
lexps_fl = lexps_fl[~np.isnan(lexps_fl)]
# perform normality test
print('Normality test for learning exponent:')
print(scipy.stats.normaltest(lexps_fl))

summary_lexp = utils.fit_probability_dist(np.array(lexplist).reshape(-1,1))
print(summary_lexp)
params_norm = summary_lexp.loc[summary_lexp['Distribution']=='norm',
                                ['Param1', 'Param2']].values[0]

plt.figure(figsize=(9,6))
sns.kdeplot(data=breaks_lexp,
            x='Learning exponent',
            hue='Sector', 
            hue_order=utils.sectors_colors.keys(),
            palette=utils.sectors_colors.values(), 
            multiple='stack',
            fill=True,
            legend=True)
plt.plot(np.linspace(-2, 2, 1000),
            scipy.stats.norm.pdf(np.linspace(-2, 2, 1000),
                                *params_norm),
            color='k',
            lw=2)

legend = plt.gca().get_legend()
handles = legend.legendHandles
handles.append(plt.Line2D([0], [0], color='k', lw=2))
labels = [x.get_text() for x in legend.get_texts()]
labels.append('Normal distribution fit')
plt.legend(handles=handles, labels=labels, loc='best')
plt.tight_layout()
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                os.path.sep + 'LEXP_fitting' + '.png')
plt.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
                os.path.sep + 'LEXP_fitting' + '.pdf')

params_breaks_lexp = [['breaks','expon',*params_expon],
                      ['lexp','norm',*params_norm]]
params_breaks_lexp = pd.DataFrame(params_breaks_lexp,
                                    columns=['Variable', 'Distribution',
                                            'Loc', 'Scale',])
params_breaks_lexp.to_csv('params_breaks_lexp.csv', index=False)

plt.show()