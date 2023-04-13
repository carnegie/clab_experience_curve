import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, os
import scipy

class Simulator():

    # initialize class (read data)
    def __init__(self, df):
        self.df = df
        self.df['Code'] = pd.factorize(self.df['Tech'])[0]
        self.select_techs()
        self.gamma = 1.0001
        self.N = 7e3

    # this can be used to select different subsets of technologies
    def select_techs(self):
        # select all techs for now
        self.techs = df['Tech'].unique()
        self.techsidx = df['Code'].unique()

    # stub simple optimization
    def optimize(self):
        res = scipy.optimize.minimize_scalar(self.simCost, method='brent', options={'disp':True})
        print(res)
        self.gamma = res.x
        self.simulate()
        self.plotTraj()
        self.plotPortfolio()

    # stub simulate fixing a parameter and retrieve associated cost
    def simCost(self, x0):
        self.gamma = x0
        return self.getObj()

    # define objective for optimization
    def getObj(self):
        self.simulate()
        return self.cost

    # standard simulation fuction
    def simulate(self):
        self.reset()
        self.portfolio = [self.units.copy()]
        while (self.flag):
            self.step()
            self.portfolio.append(self.units.copy())

    # standard step ahead
    def step(self):
        self.collectData()
        self.techMetric()
        self.computeActions()
        self.execute()
        self.updateCost()
        self.updateFlag()

    # update data points available at each step
    def collectData(self):
        for tidx in self.techsidx:
            sel = df.loc[df['Code']==tidx].copy()
            sel = sel.loc[sel['Cumulative production'] <= self.units[tidx]]
            self.obs[tidx] = sel[['Cumulative production','Unit cost']].values

    # defines the metric used to rank technology (for now last cost)
    def techMetric(self):
        self.input = np.ones(len(self.techsidx))
        for tidx in self.techsidx:
            self.input[tidx] = self.obs[tidx][-1][1]

    # defines the metric used to rank technology (for now last cost)
    def computeActions(self):
        self.actions = np.ones(len(self.techsidx))
        for tidx in self.techsidx:
            self.actions[tidx] = (1.0 / self.input[tidx])**self.gamma

    # execute the actions, potentially enforcing constrains
    def execute(self): 
        for tidx in self.techsidx:
            self.units[tidx] += self.actions[tidx]

    # update the total costs
    def updateCost(self):
        for tidx in self.techsidx:
            self.cost += self.obs[tidx][-1][1] * self.actions[tidx]

    # check if at the terminating conditions for each simualation
    def updateFlag(self):
        if sum(self.units) >= self.N:
            self.flag = False

    # reset variables before each simulation
    def reset(self):
        self.flag = True
        self.obs = [[] for x in range(df['Tech'].nunique())]
        self.units = np.ones(df['Tech'].nunique())
        self.cost = 0.0
        self.portfolio = []
    
    # plotting experienced experience curve data
    def plotTraj(self):
        fig, ax = plt.subplots()
        for tidx in self.techsidx:
            data = np.transpose(self.obs[tidx])
            ax.step(data[0], data[1], where='post')
        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)	
        ax.set_ylabel('Unit cost')
        ax.set_xlabel('Cumulative production')

    # plotting cumulat over time
    def plotPortfolio(self):
        fig, ax = plt.subplots()
        p = np.transpose(np.array(self.portfolio))
        ax.stackplot(range(len(p[0])), p)
        ax.set_ylabel('Cumulative units produced')
        ax.set_xlabel('Steps')
        fig, ax = plt.subplots()
        p = np.diff(p)
        ax.stackplot(range(len(p[0])), p)
        ax.set_ylabel('Units produced')
        ax.set_xlabel('Steps')
        fig, ax = plt.subplots()
        p = p/[sum(x) for x in np.transpose(p)] * 100
        ax.stackplot(range(len(p[0])), p)
        ax.set_ylabel('Share of units produced [%]')
        ax.set_xlabel('Steps')

# read data and create class
df = pd.read_csv('ExpCurves.csv')
simulator = Simulator(df)
# simulate and plot trajectories
simulator.simulate()
simulator.plotTraj()
simulator.plotPortfolio()
# optimize
simulator.optimize()
plt.show()
