import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import EnergySim, EnergySimParams
import time

matplotlib.rc('savefig', dpi=300)

sns.set_style('whitegrid')
sns.set_context('talk')
matplotlib.rc('font',
                **{'family':'sans-serif','sans-serif':'Helvetica'})


# x = [ x for x in range(2021, 2071)]
# v = np.zeros(len(x))
# v[0] = 1e-8
# gt0 = 100
# gT = 1
# t1 = 0
# t2 = 40
# # compute growth rate
# for y in x[:-1]:
#     print(y-x[0], t1, t2)
#     if y - x[0] < t1:
#         print('1')
#         gt = 0.01 * gt0
#     elif y - x[0] >= t1 and y - x[0] < t2:
#         print('2')
#         s_ = 50 * np.abs(0.01*(gT-gt0)/(t2-t1))
#         gt = 0.01 * gt0 + 0.01 * (gT - gt0)/(1+np.exp(-s_*(y - x[0] - t1 - (t2-t1)/2)))
#     else:
#         print('3')
#         gt = 0.01 * gT
#     # v[y+1-x[0]] = v[y-x[0]] + v[y-x[0]] * 1 * (1 - v[y-x[0]])
#     print(gt)
#     print(v[y-x[0]])
#     # input()
#     v[y+1-x[0]] = v[y-x[0]] * (1 + gt)
# plt.plot(x,v)
# plt.plot(x[1:], (v[1:] / v[:-1])-1)
# plt.show()

     


### test
# scenario = 'fast transition'
# model = EnergySim.EnergyModel(EFgp = EnergySimParams.scenarios[scenario][0],
#                                slack = EnergySimParams.scenarios[scenario][1])
# model.simulate()
# model.computeCost(EnergySimParams.costparams, learningRateTechs=EnergySimParams.learningRateTechs)
# # # model.plotDemand()
# # # model.plotUsefulEnergy()
# # # model.plotFinalEnergy()
# model.plotS7()
# # # model.plotBatteries()
# model.plotFinalEnergyBySource()
# model.plotP2X()
# model.plotCostBySource()

### simulation loop
# costs, costs2 = {}, {}
# tcosts, tcosts2 = {}, {}
# coststech = {}
# # start = time.time()
# nsim = 1000
# fig, ax = plt.subplots()
# fig2, ax2 = plt.subplots(1,5, figsize=(12,4), sharey=True)
# colors = ['k','forestgreen',
#           'lightblue','magenta','brown']
# techcolors = ['black','saddlebrown','darkgray',
#                   'saddlebrown','darkgray',
#                   'magenta','royalblue',
#                   'forestgreen','deepskyblue',
#                   'orange','pink','plum','lawngreen', 'burlywood'] 
# count = 0
# for scenario in EnergySimParams.scenarios.keys():
#     costs[scenario] = []
#     tcosts[scenario] = []
#     tcosts2[scenario] = []
#     coststech[scenario] = []
#     model = EnergySim.EnergyModel(EFgp = EnergySimParams.scenarios[scenario][0],
#                                 slack = EnergySimParams.scenarios[scenario][1])
#     start = time.time()
#     model.simulate()
#     for n in range(nsim):
#         # if n%(nsim/10) == 0:
#             # print(100*n/nsim, ' % of simulations completed, time elapsed: ', time.time() - start, ' seconds')
#         # costs[scenario].append(model.computeCost(EnergySimParams.costparams,
#         #                        EnergySimParams.learningRateTechs)[0])
#         # costs2[scenario].append(model.computeCost(EnergySimParams.costparams2,
#         #                        EnergySimParams.learningRateTechs2)[0])
#         # coststech[scenario].append(model.C)
#         tcosts[scenario].append(1e-12*model.computeCost(EnergySimParams.costparams,
#                                EnergySimParams.learningRateTechs)[1])
#         tcosts2[scenario].append(1e-12*model.computeCost(EnergySimParams.costparams2,
#                                  EnergySimParams.learningRateTechs2)[1])
#     # print('simulations completed in ', time.time() - start, ' seconds')
# #     costs[scenario] = np.array(costs[scenario])
# #     costs[scenario] = costs[scenario][:,1:51]
# #     ax.fill_between(range(model.y0+1, 2071),
# #                     np.percentile(costs[scenario], 2.5, axis=0)*1e-12,
# #                     np.percentile(costs[scenario], 97.5, axis=0)*1e-12,
# #                     color = colors[count], alpha=0.1)
# #     ax.plot(range(model.y0+1, 2071), np.median(costs[scenario], axis=0)*1e-12,
# #             color = colors[count])
# #     ax.plot(range(model.y0+1, 2071), np.mean(costs[scenario], axis=0)*1e-12,
# #             color = colors[count], ls='--')

# #     medcost = {}
# #     for t in coststech[scenario][0].keys():
# #         if t == 'P2X':
# #             continue
# #         medcost[t] = []
# #         for n in range(nsim):
# #             medcost[t].append(coststech[scenario][n][t])
# #         medcost[t] = np.array(medcost[t])
# #         medcost[t] = np.median(medcost[t][:,1:51], axis=0) * 1e-12
# #     medcost = pd.DataFrame(medcost, index=range(model.y0+1, 2071))
# #     medcost.plot.area(ax=ax2[count], stacked=True, legend=False, color=techcolors)

# #     count += 1
# # ax.set_xlabel('Year')
# # ax.set_ylabel('trillion USD')
# # ax.set_xlim([2018, 2072])
# # ax.set_ylim(2, 14)

# df = pd.DataFrame(tcosts)
# df2 = pd.DataFrame(tcosts2)
# df = df[['no transition', 'slow transition', 'fast transition']].stack().reset_index()
# df.columns = ['iter','scenario','cost']
# df2 = df2[['no transition', 'slow transition', 'fast transition']].stack().reset_index()
# df2.columns = ['iter','scenario','cost']

# df.to_csv('costs_Way.csv')
# df2.to_csv('costs_Way_alt.csv')

sns.set_palette('colorblind')

df = pd.read_csv('costs_Way.csv')
df2 = pd.read_csv('costs_Way_alt.csv')

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,4.5))
for s in ['No transition','Slow transition','Fast transition']:
    sns.kdeplot(data=df.loc[df['scenario']==s.lower(),:], ax=ax[0], x='cost', label=s)
    sns.kdeplot(data=df2.loc[df2['scenario']==s.lower(),:], ax=ax[1], x='cost')
# sns.kdeplot(data=df, ax=ax[0], hue='scenario', x='cost',)
# sns.kdeplot(data=df2, ax=ax[1], hue='scenario', x='cost',)
ax[0].set_xlabel('Net present cost [trillion USD]')
ax[1].set_xlabel('Net present cost [trillion USD]')
ax[0].set_title('Replication of Way et al. (2022)')
ax[1].set_title('Mean learning rate used for all technologies')
ax[0].set_ylabel('Density')
ax[0].set_yticks([])
ax[0].set_xlim((100,250))
fig.legend(loc='lower center', ncol=3)
fig.subplots_adjust(bottom=0.3, top=0.9,
                    left=0.05, right=0.95,
                    wspace=0.3)
plt.show()
