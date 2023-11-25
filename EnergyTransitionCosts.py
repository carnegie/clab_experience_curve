import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import energySim.EnergySim as EnergySim
import energySim.EnergySimParams as EnergySimParams

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
          'Equal - mean',
          'Equal - mean Energy w/o nuclear',
          'Equal - mean Energy',
          'Equal - mean Way et al. (2022)']

# define colors for technologies
techcolors = ['black','saddlebrown','darkgray',
                  'saddlebrown','darkgray',
                  'magenta','royalblue',
                  'forestgreen','deepskyblue',
                  'orange','pink','plum','lawngreen', 'burlywood'] 

# resimulate only if required
if simulate:

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
            techExp.append([t, scenario, model.z[t][0], model.z[t][-1]])

    # create dataframe from dictionary, update columns,
    #  and focus on relevant scenarios
    df = pd.DataFrame(tcosts).stack().explode().reset_index()
    df.columns = ['Scenario',
                'Learning rate assumptions', 
                'Net Present Cost [trillion USD]']
    df = df.loc[~df['Scenario'].str.contains('nuclear|historical') ]

    # save dataframe to csv
    df.to_csv('./energySim/Costs_all.csv')

    # convert tech expansion list to dataframe
    df = pd.DataFrame(techExp, 
                    columns=['Technology', 
                            'Scenario', 
                            'Initial production [EJ]',
                                'Final production [EJ]'])
    df.to_csv('./energySim/TechnologyExpansion.csv')

# read data
df = pd.read_csv('./energySim/Costs_all.csv')

# convert scenario name to Sentence case formatting
df['Scenario'] = df['Scenario'].str.title()

# create figure
plt.figure(figsize=(15,6))

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
                    **{'showfliers':False})

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
                    left=0.1, right=0.95)

# read data
df = pd.read_csv('./energySim/TechnologyExpansion.csv')


# create figure
fig, ax = plt.subplots(5,1, 
                       sharex=True, sharey=True, 
                       figsize=(12,10))

# counter for scenarios
count = 0

# iterate over scenarios
for s in df['Scenario'].unique():

    # counter for technologies
    countl = 0

    # iterate over technologies
    for t in df['Technology'].unique():

        # select data for specific technology and scenario
        sel = df.loc[df['Technology']==t]\
                .loc[df['Scenario']==s]
        
        # plot line from initial to final cumulative production
        ax[count].plot(
            [sel['Initial production [EJ]'].values[0], 
            sel['Final production [EJ]'].values[0]], 
            [countl, countl],
            color=techcolors[countl+5], 
            label=t)
        
        # add initial point
        ax[count].scatter(
            sel['Initial production [EJ]'].values[0], 
            countl,
            color=techcolors[countl+5], 
            marker='o')
        
        # add triangle for final point to obtain arrow
        ax[count].scatter(
            sel['Final production [EJ]'].values[0], 
            countl,
            color=techcolors[countl+5], 
            marker='>')
        countl += 1

    # remove y axis ticks
    ax[count].set_yticks([])

    # add scenario as title
    ax[count].set_title(s.title())
    count += 1

# set axes scale, labels and limits 
ax[0].set_xscale('log', base=10)
ax[-1].set_xlabel('Production [EJ]')
ax[0].set_ylim(-1,countl+1)

# adjust figure
fig.subplots_adjust(hspace=0.3, wspace=0.1, 
                    top=0.95, bottom=0.1, 
                    left=0.05, right=0.725)

# add legend
fig.legend(handles = ax[0].get_legend_handles_labels()[0],
            labels = ax[0].get_legend_handles_labels()[1],
            loc='center right', 
            bbox_to_anchor=(1, 0.5)
            )

plt.show()
