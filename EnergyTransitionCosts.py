import numpy as np
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
                **{'family':'sans-serif','sans-serif':'Helvetica'})

### test
# scenario = 'fast transition'
# model = EnergySim.EnergyModel(EFgp = EnergySimParams.scenarios[scenario][0],
#                                slack = EnergySimParams.scenarios[scenario][1])
# model.simulate()
# model.computeCost(EnergySimParams.costparams, 
#       learningRateTechs=EnergySimParams.learningRateTechs)
# # # model.plotDemand()
# # # model.plotUsefulEnergy()
# # # model.plotFinalEnergy()
# # # model.plotFinalEnergyBySource()
# # # model.plotCostBySource()

### simulation loop
costs, costs2 = {}, {}
tcosts = {}
labels = ['Technology-specific - Way et al. (2022)',
          'Equal - mean',
          'Equal - mean Energy w/o nuclear',
          'Equal - mean Energy',
          'Equal - mean Way et al. (2022)']

for l in labels:
    tcosts[l] = {}
coststech = {}
nsim = 1000
colors = ['k','forestgreen',
          'lightblue','magenta','brown']
techcolors = ['black','saddlebrown','darkgray',
                  'saddlebrown','darkgray',
                  'magenta','royalblue',
                  'forestgreen','deepskyblue',
                  'orange','pink','plum','lawngreen', 'burlywood'] 

techExp = []

count = 0
for scenario in EnergySimParams.scenarios.keys():
    costs[scenario] = []
    for l in labels:
        tcosts[l][scenario] = []
    coststech[scenario] = []
    model = EnergySim.EnergyModel(EFgp = EnergySimParams.scenarios[scenario][0],
                                slack = EnergySimParams.scenarios[scenario][1])

    model.simulate()
    
    for n in range(nsim):
        for l in labels:
            tcosts[l][scenario].append(1e-12*model.computeCost(
                EnergySimParams.costsAssumptions[l],
                EnergySimParams.learningRateTechs)[1])
    for t in model.technology[5:13]:
        techExp.append([t, scenario, model.z[t][0], model.z[t][-1]])

df = pd.DataFrame(tcosts).stack().explode().reset_index()
df.columns = ['Scenario','Learning rate assumptions', 'Net Present Cost [trillion USD]']
df = df.loc[~df['Scenario'].str.contains('nuclear|historical') ]
df.to_csv('Costs_all.csv')

df = pd.read_csv('Costs_all.csv')
df['Scenario'] = df['Scenario'].str.title()

plt.figure(figsize=(15,6))
ax = sns.boxplot(data=df, hue='Scenario', 
                 y='Net Present Cost [trillion USD]', 
                 x='Learning rate assumptions', 
                 hue_order=['No Transition', 'Slow Transition', 'Fast Transition'],
                 width=0.5, 
                 whis=(5,95),
                 linewidth=1.75,
            palette='colorblind', 
            **{'showfliers':False})

plt.gca().set_xticklabels([label.get_text().replace(' - ', '\n') for label in ax.get_xticklabels()])
sns.move_legend(ax, "lower center", ncol=3, bbox_to_anchor=(0.5, -0.6))
plt.subplots_adjust(bottom=0.375, top=0.95, left=0.1, right=0.95)

df = pd.DataFrame(techExp, 
                  columns=['Technology', 
                           'Scenario', 
                           'Initial production [EJ]', 'Final production [EJ]'])

# df['Initial production [EJ]'] = np.log10(df['Initial production [EJ]'].values)
# df['Final production [EJ]'] = np.log10(df['Final production [EJ]'].values)

fig, ax = plt.subplots(5,1, sharex=True, sharey=True, figsize=(12,10))

count = 0
for s in df['Scenario'].unique():
    countl = 0
    for t in df['Technology'].unique():
        sel = df.loc[df['Technology']==t]\
                .loc[df['Scenario']==s]
        ax[count].plot(
            [sel['Initial production [EJ]'].values[0], 
            sel['Final production [EJ]'].values[0]], 
            [countl, countl],
            color=techcolors[countl+5], 
            label=t)
        ax[count].scatter(
            sel['Initial production [EJ]'].values[0], 
            countl,
            color=techcolors[countl+5], 
            marker='o')
        ax[count].scatter(
            sel['Final production [EJ]'].values[0], 
            countl,
            color=techcolors[countl+5], 
            marker='>')
        countl += 1
    ax[count].set_yticks([])
    # ax[count].set_ylabel(s.title().replace(' ', '\n'))
    ax[count].set_title(s.title())
    count += 1
ax[0].set_xscale('log', base=10)
ax[-1].set_xlabel('Production [EJ]')
ax[0].set_ylim(-1,countl+1)
fig.subplots_adjust(hspace=0.3, 
                    wspace=0.1, 
                    top=0.95, bottom=0.1, 
                    left=0.05, right=0.725)
print(ax[0].get_legend_handles_labels())
fig.legend(handles = ax[0].get_legend_handles_labels()[0],
              labels = ax[0].get_legend_handles_labels()[1],
              loc='center right', 
            #   ncol=8, 
              bbox_to_anchor=(1, 0.5)
              )
plt.show()
