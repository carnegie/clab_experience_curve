import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, os
import scipy

matplotlib.style.use('bmh')

class Simulator():

    # initialize class (read data)
    def __init__(self, df):
        self.df = df
        self.df['Code'] = pd.factorize(self.df['Tech'])[0]
        # use all available technologies
        self.select_techs()
        # set parameter for a general simulation
        self.gamma = -2 # regulates how many units are to be produced based on current cost
        self.aC = 0.75582862
        self.astLR = 0.45194541 
        self.aLR = 0.60583144 
        self.b = 0.03912682
        self.N = 1e3
         # number of units to be produced over the simulation
#
# 
    # this can be used to select different subsets of technologies
    def select_techs(self, N=None):
        # select all techs for now
        if N == None:
            self.techs = df['Tech'].unique()
            self.techsidx = df['Code'].unique()
        # select a subset of technologies to be used in the simulation
        else:
            subset = df.drop_duplicates(subset='Tech')\
                        .sample(N, random_state = 0).copy()
            self.techs = df['Tech'].unique()
            self.techsidx = subset['Code'].unique()

    # stub simple optimization: optimize and show results with a single simulation over 50 techs tp produce 800 units
    def optimize(self):
        # res = scipy.optimize.minimize_scalar(self.simObj, 
        #                 method='brent', options={'disp':True})
        # print(res)
        # self.gamma = res.x
        # res = scipy.optimize.minimize(self.simObj, [0.1, 0.1, 0.1, 0.1], 
        #                               method='BFGS', options={'disp':True})
        res = scipy.optimize.differential_evolution(self.simObj, 
                        bounds=[[0,1],[0,1],[0,1],[0.0001,1]], 
                        disp = True, maxiter = 5)
        print(res)
        self.aC = res.x[0]
        self.astLR = res.x[1]
        self.aLR = res.x[2]
        self.b = res.x[3]
        self.N = 8e2
        self.select_techs(30)
        self.simulate()
        self.plotTraj()
        self.plotPortfolio()

    # set value suggested by solver and return the objective after simulating
    def simObj(self, x0):
        # self.gamma = x0
        self.aC = x0[0]
        self.astLR = x0[1]
        self.aLR = x0[2]
        self.b = x0[3]
        if sum(x0[:3]) <= 0.05:
            self.b = 0.1
        print(x0)
        return self.getObjStoch()


    # define objective for optimization and how to compute it for the deterministic case
    def getObj(self):
        self.simulate()
        return self.cost

    # define objective for optimization and how to compute it for the stochastic case
    # it includes stochastic selection of technologies, number of technologies available, and number of units to be produced
    # in the future it might also include the objective function
    def getObjStoch(self):
        # define where to store objective for each simulation and values of units to be produced
        objs = []
        Nrange = [1e2, 1e4, 1e6]
        nTechrange = [df['Tech'].nunique() - 30,
                      int( df['Tech'].nunique() / 10) ]
        # for each number of units to be produced
        for N in Nrange:
            self.N = N
            # sampling ten times the number and the subset of technologies available 
            for r in range(3):
                np.random.seed(0)
                ntechs = np.random.rand()
                ntechs = int( ntechs * \
                    ( nTechrange[0] - nTechrange[1]) + \
                    nTechrange[1] )
                self.select_techs(ntechs)
                # simulate and store objective
                self.simulate()
                objs.append(self.cost / sum(self.units))
        print(np.mean(objs), np.var(objs))
        return np.mean( objs ) + np.var( objs )

    # standard simulation fuction
    def simulate(self):
        # reset variables to store trajectories
        self.reset()
        # until termination conditions is met, make a step
        while (self.flag):
            self.step()

    # standard step ahead
    def step(self):
        # update information available
        self.collectData()
        # compute a metric for each technology
        self.techMetric()
        # derive an action based on the metric
        self.computeActions()
        # execute the action and update variables
        self.execute()
        self.updateCost()
        self.updateFlag()

    # update data points available at each step
    def collectData(self):
        for tidx in self.techsidx:
            # select for each technology the data that would be available
            # at a given cumulative production level and store it in obs
            sel = df.loc[df['Code']==tidx].copy()
            sel = sel.loc[sel['Cumulative production'] <= self.units[tidx]]
            cols = ['Cumulative production','Unit cost']
            self.obs[tidx] = sel[cols].values

    # defines the metric used to rank technology (for now last cost)
    def techMetric(self):
        # multiple inputs
        self.input = [[] for l in range(len(self.units))]
        for tidx in self.techsidx:
            obt = self.obs[tidx]
            obt = np.transpose(np.log10(obt))
            lastCost = obt[1][-1]
            if len(obt[0]) > 1 :
                shortTermLR = ( obt[1][-1] - obt[1][-2]) /\
                                ( self.units[tidx] - obt[0][-2] )
                LR = np.linalg.pinv(np.transpose([obt[0]]))\
                    .dot(np.transpose([obt[1],np.ones(len(obt[1]))]))[0][0]
            else:
                LR = 0
                shortTermLR = 0
            self.input[tidx] = [-lastCost, -shortTermLR, -LR]
        # self.input = np.zeros(len(self.units))
        # for tidx in self.techsidx:
        #     self.input[tidx] = self.obs[tidx][-1][1]

    # defines the metric used to rank technology (for now last cost)
    def computeActions(self):
        self.actions = np.zeros(len(self.units))
        for tidx in self.techsidx:
            self.actions[tidx] = self.units[tidx] * \
                                    max( self.input[tidx][0] * self.aC,  + \
                                    self.input[tidx][1] * self.astLR + \
                                    self.input[tidx][2] * self.aLR +
                                    self.b , 0 ) 
        # for tidx in self.techsidx:
        #     self.actions[tidx] = (self.input[tidx])**self.gamma

    # execute the actions, potentially enforcing constrains
    def execute(self): 
        # if np.all(self.actions == self.actions[0]):
        #     self.actions = [1 / sum(self.units>0) \
        #                     for x in range(len(self.actions))]
        # else:
        #     self.actions = self.actions / sum (self.actions)
        for tidx in self.techsidx:
            self.units[tidx] += self.actions[tidx] #* 100
        # keep track of units over step by reporting each step
        self.portfolio.append(self.units.copy())

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
        for tidx in range(len(self.units)):
            if tidx not in self.techsidx:
                self.units[tidx] = 0.0
        self.cost = 0.0
        self.portfolio = [self.units.copy()]

    # plotting experienced experience curve data
    def plotTraj(self):
        fig, ax = plt.subplots()
        for tidx in range(len(self.units)):
            if len(self.obs[tidx]) > 0:
                data = np.transpose(self.obs[tidx])
                ax.step(data[0], data[1], where='post')
            else:
                ax.step([1,1],[1,1], where='post')
        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)	
        ax.set_ylabel('Unit cost')
        ax.set_xlabel('Cumulative production')

    # plotting cumulative portfolio over time
    def plotPortfolio(self):
        # cumulative portfolio
        fig, ax = plt.subplots()
        p = np.transpose(np.array(self.portfolio))
        ax.stackplot(range(len(p[0])), p)
        ax.set_ylabel('Cumulative units produced')
        ax.set_xlabel('Steps')
        # units produced at each step
        fig, ax = plt.subplots()
        p = np.diff(p)
        ax.stackplot(range(len(p[0])), p)
        ax.set_ylabel('Units produced')
        ax.set_xlabel('Steps')
        # share of units produced at each step
        fig, ax = plt.subplots()
        p = p/[sum(x) for x in np.transpose(p)] * 100
        ax.stackplot(range(len(p[0])), p)
        ax.set_ylabel('Share of units produced [%]')
        ax.set_xlabel('Steps')

# read data and create class
df = pd.read_csv('ExpCurves.csv')
simulator = Simulator(df)
# simulate and plot trajectories
simulator.select_techs(30)
simulator.simulate()
simulator.plotTraj()
simulator.plotPortfolio()
plt.show()
# optimize
simulator.optimize()
plt.show()
