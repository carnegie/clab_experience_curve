import EnergySim
import EnergySimParams
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rc('savefig', dpi=300)
sns.set_style('whitegrid')
sns.set_context('talk')
matplotlib.rc('font',
                **{'family':'sans-serif',
                   'sans-serif':'Helvetica'})

# iterate over scenarios
for scenario in EnergySimParams.scenarios.keys():

    # pass input data to model
    model = EnergySim.EnergyModel(\
                EFgp = EnergySimParams.scenarios[scenario][0],
                slack = EnergySimParams.scenarios[scenario][1])

    # simulate model
    model.simulate()

    # plot some figures for quick check
    model.plotDemand()
    model.plotFinalEnergyBySource()
    model.plotS7()

    # compute costs with one random set of parameters
    model.computeCost(\
                EnergySimParams.costparams, 
                learningRateTechs = \
                    EnergySimParams.learningRateTechs)
    
    # plot costs
    model.plotCostBySource()

    plt.show()

plt.show()
