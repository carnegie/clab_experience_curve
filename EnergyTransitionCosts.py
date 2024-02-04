import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib, os
import energySim.EnergySim as EnergySim
import energySim.EnergySimParams as EnergySimParams
import numpy as np

matplotlib.rc('savefig', dpi=300)
sns.set_style('whitegrid')
sns.set_context('talk')
matplotlib.rc('font',
                **{'family':'sans-serif',
                   'sans-serif':'Helvetica'})

## set to True to run new simulations
simulate = False
# select the number of cost projection simulations
# needed to explore parameters' uncertainty
# used only if new simulations are run
nsim = 1000


# create labels for different cost assumptions
labels = ['Technology-specific - Way et al. (2022)',
          'Technology-mean - PCDB',
          'Technology-mean - PCDB-Energy, no nuclear',
          'Technology-mean - PCDB-Energy',
          'Technology-mean - Way et al. (2022)']

# define colors for technologies
techcolors = ['black','saddlebrown','darkgray',
                  'saddlebrown','darkgray',
                  'magenta','royalblue',
                  'forestgreen','deepskyblue',
                  'orange','pink','plum','lawngreen', 'burlywood'] 

# resimulate only if required
if simulate:
    np.random.seed(0)

    # create dictionary to store total costs
    tcosts = {}

    # for each label, create an empty dictionary to store costs
    for l in labels:
        tcosts[l] = {}

    # create empty list to store technology expansion
    techExp = []

    # iterate over scenarios
    for scenario in EnergySimParams.scenarios.keys():

        # create empty list to store total costs
        for l in labels:
            tcosts[l][scenario] = []

        # pass input data to model
        model = EnergySim.EnergyModel(\
                    EFgp = EnergySimParams.scenarios[scenario][0],
                    slack = EnergySimParams.scenarios[scenario][1])

        # simulate model
        model.simulate()

        # run multiple iterations to explore cost parameters' uncertainty
        for n in range(nsim):
            
            # for each cost assumption, compute total costs
            # and append it to the dictionary
            # 1e-12 is used to convert from USD to trillion USD
            for l in labels:
                tcosts[l][scenario].append( 1e-12 * \
                    model.computeCost(\
                        EnergySimParams.costsAssumptions[l],
                        EnergySimParams.learningRateTechs)[1])

        # append technology expansion to list
        for t in model.technology[5:13]:
            techExp.append([t, scenario, model.z[t][0], model.z[t][-1],
                            model.c[t][0], EnergySimParams.costparams['omega'][t]])

    # create dataframe from dictionary, update columns,
    #  and focus on relevant scenarios
    df = pd.DataFrame(tcosts).stack().explode().reset_index()
    df.columns = ['Scenario',
                'Learning rate assumptions', 
                'Net Present Cost [trillion USD]']
    df = df.loc[~df['Scenario'].str.contains('nuclear|historical') ]

    # save dataframe to csv
    df.to_csv('energySim' + os.path.sep + 'Costs_all.csv')

    # convert tech expansion list to dataframe
    df = pd.DataFrame(techExp, 
                    columns=['Technology', 
                            'Scenario', 
                            'Reference production [EJ]',
                                'Final production [EJ]',
                                'Reference cost [USD/GJ]',
                                'Learning exponent'])
    df.to_csv('energySim' + os.path.sep + 'TechnologyExpansion.csv')

# read data
df = pd.read_csv('energySim' + os.path.sep + 'Costs_all.csv')

# convert scenario name to Sentence case formatting
df['Scenario'] = df['Scenario'].str.title()

# df = df.loc[df['Learning rate assumptions'].isin(labels) ]
# df['Learning rate assumptions'] = df['Learning rate assumptions'].str.replace('.','.,')

# create figure
fig = plt.figure(figsize=(15,6))

# add boxplots
ax = sns.boxplot(data=df, 
                    hue='Scenario', 
                    y='Net Present Cost [trillion USD]', 
                    x='Learning rate assumptions', 
                    hue_order=['No Transition',
                                'Slow Transition', 
                                'Fast Transition'],
                    width=0.5, 
                    whis=(5,95),
                    linewidth=1.75,
                    palette='colorblind', 
                    gap = 0.2,
                    **{'showfliers':False})

plt.gca().set_xlabel('Experience curve assumptions')

# set x-axis labels
plt.gca()\
    .set_xticks(plt.gca().get_xticks(),
        [label.get_text().replace(' - ', '\n') \
            for label in ax.get_xticklabels()])

# move legend on the bottom
sns.move_legend(ax, "lower center", 
                ncol=3, bbox_to_anchor=(0.5, -0.6))

# adjust figure
plt.subplots_adjust(bottom=0.375, top=0.95, 
                    left=0.075, right=0.95)


axes = fig.add_axes([0.8, -0.05, 0.2, 0.35])

axes.grid(False)
axes.set_axis_off()

axes.plot([0,.5], [1,1], color='black')
axes.plot([0,.5,.5,0,0], [0.5,0.5,1.5,1.5,0.5], color='black')
# axes.fill_between([0,.5], [0,0], [2,2], color='black', alpha=.2)
axes.fill_between([0,.5], [.5,.5], [1.5,1.5], color='black', alpha=.2)
axes.plot([0,.5], [0,0], color='black')
axes.plot([0,.5], [2,2], color='black')
axes.plot([0.25,0.25], [0,.5], color='black')
axes.plot([0.25,0.25], [1.5,2], color='black')
axes.set_ylim(-1,3)
axes.set_xlim(-1.8,3)
fontsize = 14
axes.annotate('50%', xy=(-.5,1),
                    ha='center', va='center',
                    xycoords='data', 
                    fontsize=fontsize)
axes.annotate('90%', xy=(-1.5,1),
                    ha='center', va='center',
                    xycoords='data',
                    fontsize=fontsize)
axes.annotate('Median', xy=(.6,1),
                ha='left', va='center',
                    xycoords='data',
                    fontsize=fontsize)
axes.plot([-.1,-.5,-.5], [1.5,1.5,1.25], color='black')
axes.plot([-.1,-.5,-.5], [.5,.5,.75], color='black')
axes.plot([-.1,-1.5,-1.5], [2,2,1.25], color='silver')
axes.plot([-.1,-1.5,-1.5], [0,0,.75], color='silver')

if not os.path.exists('figs' + os.path.sep + 'SupplementaryFigures'):
    os.makedirs('figs' + os.path.sep + 'SupplementaryFigures')

fig.savefig('figs' + os.path.sep + 'SupplementaryFigures' + \
            os.path.sep + 'TotalDiscountedCostsExtended.png')


df = df.loc[df['Learning rate assumptions'].str.contains('Way') ]
df['Learning rate assumptions'] = df['Learning rate assumptions'].str.replace('.','.,')

# create figure
fig = plt.figure(figsize=(15,6))

# add boxplots
ax = sns.boxplot(data=df, 
                    hue='Scenario', 
                    y='Net Present Cost [trillion USD]', 
                    x='Learning rate assumptions', 
                    hue_order=['No Transition',
                                'Slow Transition', 
                                'Fast Transition'],
                    width=0.5, 
                    whis=(5,95),
                    linewidth=1.75,
                    palette='colorblind', 
                    gap = 0.2,
                    **{'showfliers':False})

plt.gca().set_xlabel('Experience curve assumptions')

# set x-axis labels
plt.gca()\
    .set_xticks(plt.gca().get_xticks(),
        [label.get_text().replace(' - ', '\n') \
            for label in ax.get_xticklabels()])

# move legend on the bottom
sns.move_legend(ax, "lower center", 
                ncol=3, bbox_to_anchor=(0.5, -0.6))

# adjust figure
plt.subplots_adjust(bottom=0.375, top=0.95, 
                    left=0.15, right=0.85)


axes = fig.add_axes([0.8, -0.05, 0.2, 0.35])

axes.grid(False)
axes.set_axis_off()

axes.plot([0,.5], [1,1], color='black')
axes.plot([0,.5,.5,0,0], [0.5,0.5,1.5,1.5,0.5], color='black')
# axes.fill_between([0,.5], [0,0], [2,2], color='black', alpha=.2)
axes.fill_between([0,.5], [.5,.5], [1.5,1.5], color='black', alpha=.2)
axes.plot([0,.5], [0,0], color='black')
axes.plot([0,.5], [2,2], color='black')
axes.plot([0.25,0.25], [0,.5], color='black')
axes.plot([0.25,0.25], [1.5,2], color='black')
axes.set_ylim(-1,3)
axes.set_xlim(-1.8,3)
fontsize = 14
axes.annotate('50%', xy=(-.5,1),
                    ha='center', va='center',
                    xycoords='data', 
                    fontsize=fontsize)
axes.annotate('90%', xy=(-1.5,1),
                    ha='center', va='center',
                    xycoords='data',
                    fontsize=fontsize)
axes.annotate('Median', xy=(.6,1),
                ha='left', va='center',
                    xycoords='data',
                    fontsize=fontsize)
axes.plot([-.1,-.5,-.5], [1.5,1.5,1.25], color='black')
axes.plot([-.1,-.5,-.5], [.5,.5,.75], color='black')
axes.plot([-.1,-1.5,-1.5], [2,2,1.25], color='silver')
axes.plot([-.1,-1.5,-1.5], [0,0,.75], color='silver')

if not os.path.exists('figs' + os.path.sep + 'energyTransitionCost'):
    os.makedirs('figs' + os.path.sep + 'energyTransitionCost')
fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'CostLearningAssumptions.png')
fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'CostLearningAssumptions.eps')

# read data
df = pd.read_csv('energySim' + os.path.sep + 'TechnologyExpansion.csv')

# remove less relevant scenarios
df = df.loc[~df['Scenario'].str.contains('nuclear|historical') ]

# initial production level from Supplementary Material of Way et al., (2022)
df.loc[df['Technology']=='electrolyzers', 
       'Initial production [EJ]'] = 1e-10
df.loc[df['Technology']=='multi-day storage', 
       'Initial production [EJ]'] = 3.6e-8
df.loc[df['Technology']=='daily batteries', 
       'Initial production [EJ]'] = 3.6e-6
df.loc[df['Technology']=='solar pv electricity', 
       'Initial production [EJ]'] = 3.6e-5
df.loc[df['Technology']=='wind electricity', 
       'Initial production [EJ]'] = 3.6e-3
df.loc[df['Technology']=='biopower electricity', 
       'Initial production [EJ]'] = 5
df.loc[df['Technology']=='hydroelectricity', 
       'Initial production [EJ]'] = 378
df.loc[df['Technology']=='nuclear electricity', 
       'Initial production [EJ]'] = 1.8

# create figure
fig, ax = plt.subplots(3,1, 
                       sharex=True, sharey=True, 
                       figsize=(12,8))

# counter for scenarios
count = 0

# iterate over scenarios
for s in ['no transition', 'slow transition', 'fast transition']:

    # counter for technologies
    countl = 0

    # iterate over technologies
    for t in df['Technology'].unique():

        # select data for specific technology and scenario
        sel = df.loc[df['Technology']==t]\
                .loc[df['Scenario']==s]
        
        # plot line from Initial to reference cumulative production
        ax[count].plot(
            [sel['Initial production [EJ]'].values[0], 
            sel['Reference production [EJ]'].values[0]], 
            [10**(np.log10(sel['Reference cost [USD/GJ]'].values[0]) + \
                sel['Learning exponent'].values[0] * \
                    (np.log10(sel['Reference production [EJ]'].values[0]) - \
                     np.log10(sel['Initial production [EJ]'].values[0]))), 
            sel['Reference cost [USD/GJ]'].values[0]],
            color=techcolors[countl+5], 
            label=t.capitalize().replace('pv','PV'),
            zorder=-1)
        
        # plot line from Reference to final cumulative production
        ax[count].plot(
            [sel['Reference production [EJ]'].values[0], 
            sel['Final production [EJ]'].values[0]], 
            [sel['Reference cost [USD/GJ]'].values[0],
             10**(np.log10(sel['Reference cost [USD/GJ]'].values[0]) - \
                sel['Learning exponent'].values[0] * \
                    (np.log10(sel['Final production [EJ]'].values[0]) - \
                     np.log10(sel['Reference production [EJ]'].values[0])))
            ],
            color=techcolors[countl+5],
            linestyle='--',
            zorder=-1)
        
        # add Initial point
        ax[count].scatter(
            sel['Initial production [EJ]'].values[0], 
            10**(np.log10(sel['Reference cost [USD/GJ]'].values[0]) + \
                sel['Learning exponent'].values[0] * \
                    (np.log10(sel['Reference production [EJ]'].values[0]) - \
                     np.log10(sel['Initial production [EJ]'].values[0]))),
            color=techcolors[countl+5], 
            edgecolor='k',
            marker='s',
            zorder=2)
        
        # add Reference point
        ax[count].scatter(
            sel['Reference production [EJ]'].values[0], 
            sel['Reference cost [USD/GJ]'].values[0],
            color=techcolors[countl+5],  
            edgecolor='k',
            marker='o',
            zorder=2)
        
        # add triangle for final point to obtain arrow
        ax[count].scatter(
            sel['Final production [EJ]'].values[0], 
            10**(np.log10(sel['Reference cost [USD/GJ]'].values[0]) - \
                sel['Learning exponent'].values[0] * \
                    (np.log10(sel['Final production [EJ]'].values[0]) - \
                     np.log10(sel['Reference production [EJ]'].values[0]))),
            color=techcolors[countl+5],  
            edgecolor='k',
            marker='>',
            zorder=2)
        countl += 1

    # remove y axis ticks
    ax[count].set_yticks([])

    # add scenario as title
    ax[count].set_title(s.title())
    count += 1

# set axes scale, labels and limits 
ax[0].set_xscale('log', base=10)
ax[0].set_yscale('log', base=10)
ax[-1].set_xlabel('Cumulative Production [EJ]')
# ax[0].set_ylim(-1,countl+1)
ax[1].set_ylabel('Cost [USD/GJ]')

# adjust figure
fig.subplots_adjust(hspace=0.3, wspace=0.1, 
                    top=0.95, bottom=0.1, 
                    left=0.1, right=0.725)

# add legend
fig.legend(handles = ax[0].get_legend_handles_labels()[0][::-1],
            labels = ax[0].get_legend_handles_labels()[1][::-1],
            loc='center right', 
            bbox_to_anchor=(1, 0.5),
            title='Energy technology'
            )

fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'TechnologiesProductionRange.png')
fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'TechnologiesProductionRange.eps')

plt.show()
