import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib, os
import energySim.EnergySim as EnergySim
import energySim.EnergySimParams as EnergySimParams
import numpy as np

matplotlib.rc('savefig', dpi=300)
sns.set_style('ticks')
sns.set_context('talk')
matplotlib.rc('font',
                **{'family':'sans-serif',
                   'sans-serif':'Helvetica'})

## set to True to run new simulations
simulate = True
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

for t in df['Technology'].unique():
    df.loc[df['Technology']==t,
       'Initial cost [USD/GJ]'] = \
         10**(np.log10(df.loc[df['Technology']==t,
                              'Reference cost [USD/GJ]'].values) + \
            df.loc[df['Technology']==t, 'Learning exponent'].values * \
                ( np.log10(df.loc[df['Technology']==t, 
                                    'Reference production [EJ]'].values) - \
                    np.log10(df.loc[df['Technology']==t, 
                                    'Initial production [EJ]'].values)))
    df.loc[df['Technology']==t,
         'Final cost [USD/GJ]'] = \
            10**(np.log10(df.loc[df['Technology']==t,
                    'Reference cost [USD/GJ]'].values) + \
                df.loc[df['Technology']==t, 'Learning exponent'].values * \
                (np.log10(df.loc[df['Technology']==t,
                    'Reference production [EJ]'].values) - \
                np.log10(df.loc[df['Technology']==t,
                    'Final production [EJ]'].values)) ) 

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

handles, labels = ax[0].get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]
handles.append(matplotlib.lines.Line2D([0],[0], color='none', marker='none', 
                                        markerfacecolor='none', markersize=10))
labels.append('')
syms = [['Initial','s'],['Present','o'],['Future','>']]
for sym in syms:
    handles.append(matplotlib.lines.Line2D([0],[0],
                                            marker=sym[1], 
                                            markeredgecolor='k', 
                                            markerfacecolor='w',
                                            markersize=8.5,
                                            markeredgewidth=1.5,
                                            color='w',))
    labels.append(sym[0])
handles.append(matplotlib.lines.Line2D([0],[0], color='k'))
labels.append('Observed')
handles.append(matplotlib.lines.Line2D([0],[0], color='k', linestyle='--'))
labels.append('Projected')

# add legend
fig.legend(handles = handles,
            labels = labels,
            loc='center right', 
            bbox_to_anchor=(1, 0.5),
            title='Energy technology'
            )

fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'TechnologiesProductionRange.png')
fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'TechnologiesProductionRange.eps')

# new figure for the same concept above

fig, ax = plt.subplots(1,3,
                       sharey=True, sharex=True,
                          figsize=(14,9))


df['Final production [EJ]'] = df['Final production [EJ]']/df['Reference production [EJ]']
df['Initial production [EJ]'] = df['Initial production [EJ]']/df['Reference production [EJ]']
df['Reference production [EJ]'] = df['Reference production [EJ]']/df['Reference production [EJ]']

df['Final cost [USD/GJ]'] = (df['Final cost [USD/GJ]']/df['Reference cost [USD/GJ]'])
df['Initial cost [USD/GJ]'] = (df['Initial cost [USD/GJ]']/df['Reference cost [USD/GJ]'])
df['Reference cost [USD/GJ]'] = (df['Reference cost [USD/GJ]']/df['Reference cost [USD/GJ]'])

count = 0
# iterate over scenarios
for s in ['no transition', 'slow transition', 'fast transition']:

    # counter for technologies
    countl = 10

    # iterate over technologies
    for t in df['Technology'].unique():

        if t in ['hydroelectricity', 'nuclear electricity']:
            continue

        # select data for specific technology and scenario
        sel = df.loc[df['Technology']==t]\
                .loc[df['Scenario']==s]

        ax[count].plot(sel[['Initial production [EJ]', 'Reference production [EJ]']].values[0],
                       10 ** (3 * (countl-9)) * 
                       sel[['Initial cost [USD/GJ]', 'Reference cost [USD/GJ]']].values[0],
                       marker = 'o', markeredgecolor='k', color=techcolors[countl-3], linestyle='-')
        ax[count].plot(sel[['Reference production [EJ]', 'Final production [EJ]']].values[0],
                       10 ** (3 * (countl-9)) *  
                          sel[['Reference cost [USD/GJ]', 'Final cost [USD/GJ]']].values[0],
                          marker = 'o', markeredgecolor='k', linestyle='--', color=techcolors[countl-3])
        ax[count].annotate(t.capitalize().replace('pv','PV'),
                            xy=(sel['Reference production [EJ]'].values[0] * 1.5, 
                                 10 ** (3 * (0.1+countl-9)) * 
                                 sel['Reference cost [USD/GJ]'].values[0]),
                            xycoords='data',
                            ha='left',
                            va='bottom',
                            style='italic',
                            fontsize=14,
                            color=techcolors[countl-3])
        # ax[count].annotate('- '+str(round(100 * (1 - sel['Reference cost [USD/GJ]'].values[0])))+' %',
        #                     xy=(sel['Reference production [EJ]'].values[0] * 0.5,
        #                         10 ** (-0.2+countl) * 
        #                         sel['Reference cost [USD/GJ]'].values[0]**0.25),
        #                     xycoords='data',
        #                     ha='center',
        #                     va='top',
        #                     fontsize=12,
        #                     color=techcolors[countl-3])
        ax[count].annotate('- '+str(round(100 * (1 - sel['Final cost [USD/GJ]'].values[0]/\
                                                 sel['Reference cost [USD/GJ]'].values[0])))+' %',
                            xy=(sel['Final production [EJ]'].values[0] * 5,
                                10 ** (3*(countl-9)) * 
                                sel['Final cost [USD/GJ]'].values[0]),
                            xycoords='data',
                            ha='left',
                            va='center',
                            fontsize=14,
                            color=techcolors[countl-3])

        countl += 1

        ax[count].set_xscale('log', base=10)
        ax[count].set_yscale('log', base=10)
    ax[count].set_yticklabels([])
    [ax[count].axhline(10**x, color='k', linestyle='-', 
                       lw=.25, alpha=0.75, zorder=-5)
        for x in range(3,19,3)]
    ax[count].set_yticks([10**x for x in range(1,19)])

    minorticks = []
    for val in range(2,19):
        # [ax[count].axhline(10*(x-1), color='k', linestyle='-', 
        #                    lw=.05, alpha=0.5, zorder=-10) 
        #  for x in np.arange(10**(val),10**(val+1),10**(val))]
        [minorticks.append(10*(x-1)) for x in np.arange(10**(val),10**(val+1),10**(val))]
    ax[count].set_yticks(minorticks, minor=True)


    [ax[count].axvline(10**x, color='k', linestyle='-',
                      lw=.25, alpha=0.75, zorder=-5)
        for x in range(-9,12,3)]
    

    count += 1
ax[0].set_xlim(1e-8,10**12)
ax[0].set_ylim(500,6*10**18)
ax[0].set_xticks([1e-6,1e-3,1,1e3,1e6,1e9,1e12],)
labels = ax[0].get_xticklabels()
labels[2] = '1'
ax[0].set_xticklabels(labels)
ax[0].set_title('No transition')
ax[1].set_title('Slow transition')
ax[2].set_title('Fast transition')
ax[0].set_ylabel('Change in cost relative to present')
ax[1].set_xlabel('Change in cumulative production relative to present')

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.975)

fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'TechnologiesProductionRangeRelative.png')
fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'TechnologiesProductionRangeRelative.eps')

# # combine figures

# fig = plt.figure(figsize=(15,13))

# gs0 = matplotlib.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2,1])

# gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0])
# gs01 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1])

# ax = gs00.subplots(sharex=True, sharey=True)

# count = 0
# # iterate over scenarios
# for s in ['no transition', 'slow transition', 'fast transition']:

#     # counter for technologies
#     countl = 10

#     # iterate over technologies
#     for t in df['Technology'].unique():

#         if t in ['hydroelectricity', 'nuclear electricity']:
#             continue

#         # select data for specific technology and scenario
#         sel = df.loc[df['Technology']==t]\
#                 .loc[df['Scenario']==s]

#         ax[count].plot(sel[['Initial production [EJ]', 'Reference production [EJ]']].values[0],
#                        10 ** (3 * (countl-9)) * 
#                        sel[['Initial cost [USD/GJ]', 'Reference cost [USD/GJ]']].values[0],
#                        marker = 'o', markeredgecolor='k', color=techcolors[countl-3], linestyle='-')
#         ax[count].plot(sel[['Reference production [EJ]', 'Final production [EJ]']].values[0],
#                        10 ** (3 * (countl-9)) *  
#                           sel[['Reference cost [USD/GJ]', 'Final cost [USD/GJ]']].values[0],
#                           marker = 'o', markeredgecolor='k', linestyle='--', color=techcolors[countl-3])
#         ax[count].annotate(t.capitalize().replace('pv','PV'),
#                             xy=(sel['Reference production [EJ]'].values[0] * 1.5, 
#                                  10 ** (3 * (0.05+countl-9)) * 
#                                  sel['Reference cost [USD/GJ]'].values[0]),
#                             xycoords='data',
#                             ha='left',
#                             va='bottom',
#                             style='italic',
#                             fontsize=16,
#                             color=techcolors[countl-3])
#         # ax[count].annotate('- '+str(round(100 * (1 - sel['Reference cost [USD/GJ]'].values[0])))+' %',
#         #                     xy=(sel['Reference production [EJ]'].values[0] * 0.5,
#         #                         10 ** (-0.2+countl) * 
#         #                         sel['Reference cost [USD/GJ]'].values[0]**0.25),
#         #                     xycoords='data',
#         #                     ha='center',
#         #                     va='top',
#         #                     fontsize=12,
#         #                     color=techcolors[countl-3])
#         ax[count].annotate('- '+str(round(100 * (1 - sel['Final cost [USD/GJ]'].values[0]/\
#                                                  sel['Reference cost [USD/GJ]'].values[0])))+' %',
#                             xy=(sel['Final production [EJ]'].values[0] * 5,
#                                 10 ** (3*(countl-9)) * 
#                                 sel['Final cost [USD/GJ]'].values[0]),
#                             xycoords='data',
#                             ha='left',
#                             va='center',
#                             fontsize=16,
#                             color=techcolors[countl-3])

#         countl += 1

#         ax[count].set_xscale('log', base=10)
#         ax[count].set_yscale('log', base=10)
#     ax[count].set_yticklabels([])
#     [ax[count].axhline(10**x, color='k', linestyle='-', 
#                        lw=.25, alpha=0.75, zorder=-5)
#         for x in range(3,19,3)]
#     ax[count].set_yticks([10**x for x in range(1,19)])

#     minorticks = []
#     for val in range(2,19):
#         # [ax[count].axhline(10*(x-1), color='k', linestyle='-', 
#         #                    lw=.05, alpha=0.5, zorder=-10) 
#         #  for x in np.arange(10**(val),10**(val+1),10**(val))]
#         [minorticks.append(10*(x-1)) for x in np.arange(10**(val),10**(val+1),10**(val))]
#     ax[count].set_yticks(minorticks, minor=True)


#     [ax[count].axvline(10**x, color='k', linestyle='-',
#                       lw=.25, alpha=0.75, zorder=-5)
#         for x in range(-9,12,3)]
    

#     count += 1
# ax[0].set_xlim(1e-8,10**12)
# ax[0].set_ylim(500,6*10**18)
# ax[0].set_xticks([1e-6,1e-3,1,1e3,1e6,1e9,1e12],)
# labels = ax[0].get_xticklabels()
# labels[2] = '1'
# ax[0].set_xticklabels(labels)
# ax[0].set_title('No transition')
# ax[1].set_title('Slow transition')
# ax[2].set_title('Fast transition')
# ax[0].set_ylabel('Change in cost relative to present')
# ax[1].set_xlabel('Change in cumulative production relative to present')
# ax[0].annotate('a',
#             xy=(0.05, 0.97),
#             xycoords='axes fraction',
#             ha='center',
#             va='center',)
# ax[1].annotate('b',
#             xy=(0.05, 0.97),
#             xycoords='axes fraction',
#             ha='center',
#             va='center',)
# ax[2].annotate('c',
#             xy=(0.05, 0.97),
#             xycoords='axes fraction',
#             ha='center',
#             va='center',)

# # read data
# df = pd.read_csv('energySim' + os.path.sep + 'Costs_all.csv')


# # convert scenario name to Sentence case formatting
# df['Scenario'] = df['Scenario'].str.title()

# ax = gs01.subplots(sharex=True, sharey=True)

# count = 0
# for hyp in ['Technology-specific - Way et al. (2022)',
#             'Technology-mean - PCDB']:
#     counts = 0
#     for s in ['No Transition', 'Slow Transition', 'Fast Transition']:

#         sel = df.loc[df['Learning rate assumptions']==hyp]\
#                 .loc[df['Scenario']==s]
        
#         ax[count].fill_between([-.75+3*counts,.75+3*counts],
#                         sel['Net Present Cost [trillion USD]'].quantile([.05,.05]),
#                         sel['Net Present Cost [trillion USD]'].quantile([.95,.95]),
#                         color=sns.color_palette('colorblind')[counts], lw=0, alpha=.3)
#         ax[count].fill_between([-.75+3*counts,.75+3*counts],
#                         sel['Net Present Cost [trillion USD]'].quantile([.25,.25]),
#                         sel['Net Present Cost [trillion USD]'].quantile([.75,.75]),
#                         color=sns.color_palette('colorblind')[counts], lw=0, alpha=.6)
#         ax[count].plot([-.75+3*counts,.75+3*counts],
#                        [sel['Net Present Cost [trillion USD]'].median(),
#                         sel['Net Present Cost [trillion USD]'].median()],
#                        color='w', linestyle='--', lw=2)
#         counts += 1
#     count += 1

# ax[0].set_title('Technology-specific')
# ax[1].set_title('Technology-mean')
# ax[0].set_ylabel('Net Present Cost (T $)')
# ax[0].set_xticks([])
# ax[1].set_xticks([])
# ax[0].annotate('d',
#                 xy=(.05,.9),
#                 xycoords='axes fraction',
#                 weight='bold')
# ax[1].annotate('e',
#                 xy=(.05,.9),
#                 xycoords='axes fraction',
#                 weight='bold')
# ax[0].annotate('No transition',
#                 xy=(0,220),
#                 xycoords='data',
#                 ha='center',
#                 va='center',
#                 color=sns.color_palette('colorblind')[0])
# ax[0].annotate('Slow transition',
#                 xy=(3,235),
#                 xycoords='data',
#                 ha='center',
#                 va='center',
#                 color=sns.color_palette('colorblind')[1])
# ax[0].annotate('Fast transition',
#                 xy=(6,250),
#                 xycoords='data',
#                 ha='center',
#                 va='center',
#                 color=sns.color_palette('colorblind')[2])
# ax[1].annotate('No transition',
#                 xy=(0,220),
#                 xycoords='data',
#                 ha='center',
#                 va='center',
#                 color=sns.color_palette('colorblind')[0])
# ax[1].annotate('Slow transition',
#                 xy=(3,200),
#                 xycoords='data',
#                 ha='center',
#                 va='center',
#                 color=sns.color_palette('colorblind')[1])
# ax[1].annotate('Fast transition',
#                 xy=(6,180),
#                 xycoords='data',
#                 ha='center',
#                 va='center',
#                 color=sns.color_palette('colorblind')[2])

# for axx in [ax[0], ax[1]]:
#     axx.fill_between([0.2,.6], [112,112], [150,150], color='gray', alpha=.3)
#     axx.fill_between([0.2,.6], [122,122], [140,140], color='gray', alpha=.6)
#     axx.plot([0.2,.6], [132,132], color='w', linestyle='--', lw=2)
#     axx.annotate('90%', xy=(-.1,145), xycoords='data', 
#                 ha='center', va='center', color='k',
#                 fontsize=12
#                 )
#     axx.annotate('50%', xy=(-.1,135), xycoords='data', 
#                 ha='center', va='center', color='k',
#                 fontsize=12
#                 )
#     axx.annotate('Median', xy=(1.1,132), xycoords='data', 
#                 ha='center', va='center', color='k',
#                 fontsize=12
#                 )
#     axx.plot([.1,-.45,-.45,.1], [140,140,122,122], color='k', lw=.2)
#     axx.plot([.1,-.7,-.7,.1], [150,150,112,112], color='k', lw=.2)

# ax[0].set_ylim(110,260)
# ax[0].set_xlim(-1.5,7.5)


# fig.subplots_adjust(top=0.95, bottom=0.025, 
#                     left=0.1, right=0.95,
#                     hspace=0.25, wspace=0.1)


# fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
#             os.path.sep + 'EnergySystemCost.png')
# fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
#             os.path.sep + 'EnergySystemCost.eps')     


# separate the above into two figures
# aggregate the three panels into one

fig, ax = plt.subplots(1,1, figsize=(7,8))

# counter for technologies
countl = 10

# iterate over technologies
for t in df['Technology'].unique():

    if t in ['hydroelectricity', 'nuclear electricity']:
        continue

    # select data for specific technology and scenario
    sel = df.loc[df['Technology']==t]

    ax.plot(sel[['Initial production [EJ]', 'Reference production [EJ]']].values[0],
                    10 ** (3 * (countl-9)) * 
                    sel[['Initial cost [USD/GJ]', 'Reference cost [USD/GJ]']].values[0],
                    marker = 'o', markeredgecolor='k', color=techcolors[countl-3], linestyle='-')
    markers = ['s','>','d']
    counts = 0
    for s in ['no transition', 'slow transition', 'fast transition']:

        sel_ = sel.loc[sel['Scenario']==s]

        ax.plot(sel_[['Reference production [EJ]', 'Final production [EJ]']].values[0],
                        10 ** (3 * (countl-9)) *  
                            sel_[['Reference cost [USD/GJ]', 'Final cost [USD/GJ]']].values[0],
                            linestyle='--', color=techcolors[countl-3])
        ax.scatter(sel_['Final production [EJ]'].values[0],
                    10 ** (3*(countl-9)) * 
                    sel_['Final cost [USD/GJ]'].values[0],
                    color=techcolors[countl-3], 
                    edgecolor='k',
                    marker=markers[counts],
                    zorder=2, alpha=.7)
        counts += 1
    ax.annotate(t.capitalize().replace('pv','PV'),
                        xy=(sel['Reference production [EJ]'].values[0] * 1.5, 
                                10 ** (3 * (0.05+countl-9)) * 
                                sel['Reference cost [USD/GJ]'].values[0]),
                        xycoords='data',
                        ha='left',
                        va='bottom',
                        style='italic',
                        fontsize=16,
                        color=techcolors[countl-3])
    # ax.annotate('- '+str(round(100 * (1 - sel['Final cost [USD/GJ]'].values[0]/\
    #                                             sel['Reference cost [USD/GJ]'].values[0])))+' %',
    #                     xy=(sel['Final production [EJ]'].values[0] * 5,
    #                         10 ** (3*(countl-9)) * 
    #                         sel['Final cost [USD/GJ]'].values[0]),
    #                     xycoords='data',
    #                     ha='left',
    #                     va='center',
    #                     fontsize=16,
    #                     color=techcolors[countl-3])

    countl += 1

ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
ax.set_yticklabels([])
[ax.axhline(10**x, color='k', linestyle='-', 
                    lw=.25, alpha=0.75, zorder=-5)
    for x in range(3,19,3)]
ax.set_yticks([10**x for x in range(1,19)])

minorticks = []
for val in range(2,19):
    # [ax[count].axhline(10*(x-1), color='k', linestyle='-', 
    #                    lw=.05, alpha=0.5, zorder=-10) 
    #  for x in np.arange(10**(val),10**(val+1),10**(val))]
    [minorticks.append(10*(x-1)) for x in np.arange(10**(val),10**(val+1),10**(val))]
ax.set_yticks(minorticks, minor=True)


[ax.axvline(10**x, color='k', linestyle='-',
                    lw=.25, alpha=0.75, zorder=-5)
    for x in range(-9,12,3)]
    


ax.set_xlim(1e-8,10**12)
ax.set_ylim(500,6*10**18)
ax.set_xticks([1e-6,1e-3,1,1e3,1e6,1e9,1e12],)
labels = ax.get_xticklabels()
labels[2] = '1'
ax.set_xticklabels(labels)
# ax[0].set_title('No transition')
# ax[1].set_title('Slow transition')
# ax[2].set_title('Fast transition')
ax.set_ylabel('Change in cost relative to present')
ax.set_xlabel('Change in cumulative production relative to present')
# ax[0].annotate('a',
#             xy=(0.05, 0.97),
#             xycoords='axes fraction',
#             ha='center',
#             va='center',)
# ax[1].annotate('b',
#             xy=(0.05, 0.97),
#             xycoords='axes fraction',
#             ha='center',
#             va='center',)
# ax[2].annotate('c',
#             xy=(0.05, 0.97),
#             xycoords='axes fraction',
#             ha='center',
#             va='center',)

fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.725)

axes = fig.add_axes([0.725, 0.4, 0.25, 0.2])
axes.axis('off')
axes.scatter([-.5], [1], color='none',
           marker='s', edgecolor='k', lw=1, s=100)
axes.scatter([-.5], [0], color='none',
           marker='>', edgecolor='k', lw=1, s=100)
axes.scatter([-.5], [-1], color='none',
           marker='d', edgecolor='k', lw=1, s=100)
axes.annotate('No Transition', xy=(1,1),
                ha='center', va='center',
                xycoords='data',
                fontsize=14)
axes.annotate('Slow Transition', xy=(1,0),
                ha='center', va='center',
                xycoords='data',
                fontsize=14)
axes.annotate('Fast Transition', xy=(1,-1),
                ha='center', va='center',
                xycoords='data',
                fontsize=14)

axes.set_xlim(-1,2)
axes.set_ylim(-2,2)

fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'ProductionProjections.png')
fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'ProductionProjections.eps')

# figure for the cost projections
fig, ax = plt.subplots(1,2, figsize=(12,7), sharex=True, sharey=True)

# read data
df = pd.read_csv('energySim' + os.path.sep + 'Costs_all.csv')


# convert scenario name to Sentence case formatting
df['Scenario'] = df['Scenario'].str.title()

count = 0
for hyp in ['Technology-specific - Way et al. (2022)',
            'Technology-mean - PCDB']:
    counts = 0
    for s in ['No Transition', 'Slow Transition', 'Fast Transition']:

        sel = df.loc[df['Learning rate assumptions']==hyp]\
                .loc[df['Scenario']==s]
        
        ax[count].fill_between([-.75+3*counts,.75+3*counts],
                        sel['Net Present Cost [trillion USD]'].quantile([.05,.05]),
                        sel['Net Present Cost [trillion USD]'].quantile([.95,.95]),
                        color=sns.color_palette('colorblind')[counts], lw=0, alpha=.3)
        ax[count].fill_between([-.75+3*counts,.75+3*counts],
                        sel['Net Present Cost [trillion USD]'].quantile([.25,.25]),
                        sel['Net Present Cost [trillion USD]'].quantile([.75,.75]),
                        color=sns.color_palette('colorblind')[counts], lw=0, alpha=.6)
        ax[count].plot([-.75+3*counts,.75+3*counts],
                       [sel['Net Present Cost [trillion USD]'].median(),
                        sel['Net Present Cost [trillion USD]'].median()],
                       color='w', linestyle='--', lw=2)
        counts += 1
    count += 1

ax[0].set_title('Technology-specific')
ax[1].set_title('Technology-mean')
ax[0].set_ylabel('Net Present Cost (T $)')
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[0].annotate('a',
                xy=(.05,.9),
                xycoords='axes fraction',
                weight='bold')
ax[1].annotate('b',
                xy=(.05,.9),
                xycoords='axes fraction',
                weight='bold')
ax[0].annotate('No transition',
                xy=(0,220),
                xycoords='data',
                ha='center',
                va='center',
                color=sns.color_palette('colorblind')[0])
ax[0].annotate('Slow transition',
                xy=(3,235),
                xycoords='data',
                ha='center',
                va='center',
                color=sns.color_palette('colorblind')[1])
ax[0].annotate('Fast transition',
                xy=(6,250),
                xycoords='data',
                ha='center',
                va='center',
                color=sns.color_palette('colorblind')[2])
ax[1].annotate('No transition',
                xy=(0,220),
                xycoords='data',
                ha='center',
                va='center',
                color=sns.color_palette('colorblind')[0])
ax[1].annotate('Slow transition',
                xy=(3,200),
                xycoords='data',
                ha='center',
                va='center',
                color=sns.color_palette('colorblind')[1])
ax[1].annotate('Fast transition',
                xy=(6,180),
                xycoords='data',
                ha='center',
                va='center',
                color=sns.color_palette('colorblind')[2])

for axx in [ax[0], ax[1]]:
    axx.fill_between([0.2,.6], [112,112], [150,150], color='gray', alpha=.3)
    axx.fill_between([0.2,.6], [122,122], [140,140], color='gray', alpha=.6)
    axx.plot([0.2,.6], [131,131], color='w', linestyle='--', lw=2)
    axx.annotate('90%', xy=(-.1,145), xycoords='data', 
                ha='center', va='center', color='k',
                fontsize=12
                )
    axx.annotate('50%', xy=(-.1,135), xycoords='data', 
                ha='center', va='center', color='k',
                fontsize=12
                )
    axx.annotate('Median', xy=(1.1,131), xycoords='data', 
                ha='center', va='center', color='k',
                fontsize=12
                )
    axx.plot([.1,-.45,-.45,.1], [140,140,122,122], color='k', lw=.2)
    axx.plot([.1,-.7,-.7,.1], [150,150,112,112], color='k', lw=.2)

ax[0].set_ylim(110,260)
ax[0].set_xlim(-1.5,7.5)

fig.subplots_adjust(top=0.95, bottom=0.025, 
                    left=0.075, right=0.975,
                    hspace=0.1, wspace=0.1)

fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'EnergySystemCost.png')
fig.savefig('figs' + os.path.sep + 'energyTransitionCost' + \
            os.path.sep + 'EnergySystemCost.eps')

plt.show()
