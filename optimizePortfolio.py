import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, os
import scipy
import time
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

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
        self.aC = 7.88257189
        self.astLR = 6.13032912
        self.aLR = 9.8532207 
        self.b = 0.91590684
        self.kC = 6.24135516
        self.kstLR = 7.43391527
        self.kLR =  8.48427037 
        self.kb = 9.6015371
        self.N = 20
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
                        .sample(N, random_state=N).copy()
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
                        bounds=[[0,10],[0,10],[0,10],[0.0001,1],
                                [-10,10], [-10,10], [-10,10],[0,10]], 
                        disp = True, maxiter = 5)
        print(res)
        self.aC = res.x[0]
        self.astLR = res.x[1]
        self.aLR = res.x[2]
        self.b = res.x[3]

    # set value suggested by solver and return the objective after simulating
    def simObj(self, x0):
        # self.gamma = x0
        self.aC = x0[0]
        self.astLR = x0[1]
        self.aLR = x0[2]
        self.b = x0[3]
        self.kC = x0[4]
        self.kstLR = x0[5]
        self.kLR = x0[6]
        self.kb = x0[7]
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
        start = time.time()
        # define where to store objective for each simulation and values of units to be produced
        objs = []
        Nrange = [20]
        nTechrange = [50 ,
                      2 ]
        # for each number of units to be produced
        for N in Nrange:
            self.N = N
            # sampling ten times the number and the subset of technologies available 
            np.random.seed(0)
            for r in range(10):
                ntechs = np.random.rand()
                ntechs = int( ntechs * \
                    ( nTechrange[0] - nTechrange[1]) + \
                    nTechrange[1] )
                self.select_techs(ntechs)
                # simulate and store objective
                self.simulate()
                objs.append(self.cost / sum(self.units))
        # print(time.time()-start)
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
        # what are the statistics of the data that are useful to take decisions?
        # what model of the data gives you the best predictive skill? 
        self.techMetric() 
        # derive an action based on the metric
        self.computeActions()
        # the number of steps is known in advance once N is known
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
            sel = sel.loc[sel['Cumulative production'] <= max(self.units[tidx],1.0)]
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
            self.actions[tidx] = max(self.units[tidx] * \
                                    max( self.input[tidx][0] * self.aC,  + \
                                    self.input[tidx][1] * self.astLR + \
                                    self.input[tidx][2] * self.aLR + \
                                    self.b , 0.1) * \
                                ( 1 - 1.0 / \
                                max(self.kC * self.input[tidx][0] + \
                                    self.kstLR * self.input[tidx][1] + \
                                    self.kLR * self.input[tidx][2] + \
                                    self.kb, 1.1 )) - self.units[tidx] ,
                                    0 )
        if sum(self.actions) == 0:
            for tidx in self.techsidx:
                self.actions[tidx] = 0.01
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
            self.units[tidx] += self.actions[tidx] 
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
        self.units = np.ones(df['Tech'].nunique()) * 0.01
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
simulator.select_techs(5)
simulator.simulate()
print(simulator.cost)
simulator.plotTraj()
simulator.plotPortfolio()
plt.show()

# optimize
# simulator.optimize()
# plt.show()

class PymooProblem(ElementwiseProblem):
    def __init__(self):
        xl = np.zeros(8)
        # xl[4] = -10
        # xl[5] = -10
        # xl[6] = -10
        # xl[7] = -10
        
        xu = np.ones(8)*10
        xu[3] = 1
        super().__init__(n_var=8,
                     n_obj=1,
                     n_ieq_constr=0,
                     xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = simulator.simObj(x)

problem = PymooProblem()
algorithm = NSGA2(pop_size=100)
res = minimize(problem,
               algorithm,
               seed=1,
               termination=('n_eval', 1000),
               verbose=True)

print(res.X)
print(res.F)
print(res.pop.get('X'))
print(res.pop.get('F'))

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
