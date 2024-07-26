import numpy as np
import pandas as pd
import copy, os

### this script contains the parameters used 
### to simulate the energy model of Way et al. (2022)

scenarios = {}

### NO TRANSITION ###

# growth rates of carriers
# growthparams = gt0, gT, t1, t2, t3, psi
EFgp = {}
EFgp['oil','transport'] = [1.99, 0.2, 12, 20, 0, 1]
# EFgp['electricity', 'transport'] = [0.0, 0.0, 0.0, 0.0, 0, 1]
EFgp['P2Xfuels','transport'] = [0.0, 0.0, 0., 0., 0, 1]
EFgp['oil','industry'] = [1.9, 0.1, 10, 20, 0, 1]
EFgp['coal','industry'] = [1.4, 0.2, 10, 20, 0, 1]
EFgp['gas','industry'] = [2.3, 0.5, 10, 30, 0, 1]
# EFgp['electricity','industry'] = [0, 0, 0, 0, 0, 1]
EFgp['P2Xfuels','industry'] = [0, 0, 0, 0, 0, 1]
EFgp['oil','buildings'] = [-1.0, -1.0, 10, 30, 0, 1]
EFgp['coal','buildings'] = [-5.3, -6.3, 10, 20, 0, 1]
EFgp['gas','buildings'] = [1.4, 0.3, 10, 30, 0, 1]
# EFgp['electricity','buildings'] = [0.0, 0.0, 0.0, 0.0, 0, 1]
EFgp['P2Xfuels','buildings'] = [0.0, 0.0, 0., 0., 0, 1]
EFgp['coal electricity','electricity'] = [1.5, 1.0, 10, 30, 0, 1]
EFgp['gas electricity','electricity'] = [2.5, 1.5, 10, 30, 0, 1]
EFgp['nuclear electricity','electricity'] = [1.5, 1.4, 10, 20, 0, 1]
EFgp['hydroelectricity','electricity'] = [1.1, 1.1, 10, 20, 0, 1]
# EFgp['biopower electricity','electricity'] = [0, 0, 0, 0, 0, 1]
EFgp['wind electricity','electricity'] = [8, 5, 10, 30, 1, .28]
EFgp['solar pv electricity','electricity'] = [12, 5, 10, 30, 1, .28]
EFgp['EV batteries','electricity'] = [10, 0, 0, 0, 0 ,1]
EFgp['daily batteries','electricity'] = [0, 0, 0, 0, 0, 0.0001]
EFgp['multi-day batteries','electricity'] = [0, 0, 0, 0, 0, 0.0001]
EFgp['P2Xfuels','electricity'] = [0, 0, 0, 0, 0, 0]

slack = {'transport':'electricity', 'industry':'electricity',
            'buildings':'electricity', 'energy':'electricity', 
            'electricity':'biopower electricity'}

scenarios['no transition'] = [EFgp, slack]

### FAST TRANSITION ###

EFgp = {}
# EFgp['oil','transport'] = [1.99, 0.2, 12, 20, 0, 1]
EFgp['electricity','transport'] = [25, 5, 8, 20, 1, .8]
EFgp['P2Xfuels','transport'] = [60, 4, 13, 31, 1, .2]
EFgp['oil','industry'] = [0, -15, 2, 10, 0, 1]
EFgp['coal','industry'] = [0, -12, 2, 14, 0, 1]
# EFgp['gas','industry'] = [0, 0, 0, 0, 0, 1]
EFgp['electricity','industry'] = [7, 4, 8, 32, 1, .75]
EFgp['P2Xfuels','industry'] = [70, 3, 2, 41, 1, 0.245]
EFgp['oil','buildings'] = [0, -16, 1, 8, 0, 1]
EFgp['coal','buildings'] = [0, -16, 1, 8, 0, 1]
# EFgp['gas','buildings'] = [0, 0, 0, 0, 0, 1]
EFgp['electricity','buildings'] = [4, 8, 4, 30, 1, .895]
EFgp['P2Xfuels','buildings'] = [60, 4, 13, 31, 1, 0.1]
EFgp['coal electricity','electricity'] = [0, -30, 2, 18, 0, 1]
# EFgp['gas electricity','electricity'] = [2.5, 1.5, 10, 30, 0, 1]
EFgp['nuclear electricity','electricity'] = [2, -4, 5, 15, 0, 1]
EFgp['hydroelectricity','electricity'] = [2, 2, 10, 30, 60, 0.055]
EFgp['biopower electricity','electricity'] = [10, 3, 10, 16, 1, 0.042]
EFgp['wind electricity','electricity'] = [15, 8, 5, 20, 1, .3]
EFgp['solar pv electricity','electricity'] = [28, 9, 5, 22, 1, .6]
EFgp['EV batteries','electricity'] = [50, 0 ,0 ,0 ,0 ,0]
EFgp['daily batteries','electricity'] = [60, 0 ,0 ,0 ,0 , 0.2]
EFgp['multi-day batteries','electricity'] = [80, 0 ,0 ,0 ,0 , 0.1]
EFgp['P2Xfuels','electricity'] = [0, 0, 0, 0, 40, 1/12]

slack = {'transport':'oil', 'industry':'gas',
            'buildings':'gas', 'energy':'electricity', 
            'electricity':'gas electricity'}

scenarios['fast transition'] = [EFgp, slack]


### SLOW TRANSITION ###

EFgp = {}
EFgp['oil','transport'] = [0.0, -5, 15, 30, 0, 1]
# EFgp['electricity','transport'] = [25, 5, 8, 20, 1, .8]
EFgp['P2Xfuels','transport'] = [35, 12, 12, 42, 1, .18]
EFgp['oil','industry'] = [0, -6, 15, 30, 0, 1]
EFgp['coal','industry'] = [0, -6, 10, 30, 0, 1]
EFgp['gas','industry'] = [2, -6, 15, 40, 0, 1]
# EFgp['electricity','industry'] = [7, 4, 8, 32, 1, .75]
EFgp['P2Xfuels','industry'] = [45, 12, 12, 42, 1, 0.2]
EFgp['oil','buildings'] = [0, -6, 15, 30, 0, 1]
EFgp['coal','buildings'] = [0, -5, 15, 30, 0, 1]
EFgp['gas','buildings'] = [2, -9, 15, 40, 0, 1]
# EFgp['electricity','buildings'] = [4, 8, 4, 30, 1, .895]
EFgp['P2Xfuels','buildings'] = [40, 10, 12, 42, 1, 0.1]
EFgp['coal electricity','electricity'] = [1, -10, 5, 30, 0, 1]
# EFgp['gas electricity','electricity'] = [2.5, 1.5, 10, 30, 0, 1]
EFgp['nuclear electricity','electricity'] = [3, 0, 10, 30, 0, 1]
EFgp['hydroelectricity','electricity'] = [3, 1, 10, 30, 0, 1]
EFgp['biopower electricity','electricity'] = [7, 4, 10, 20, 1, 0.12]
EFgp['wind electricity','electricity'] = [12, 5, 1, 30, 1, .26]
EFgp['solar pv electricity','electricity'] = [17, 5, 1, 43, 1, .56]
EFgp['EV batteries','electricity'] = [50, 0 ,0 ,0 ,0 ,0]
EFgp['daily batteries','electricity'] = [60, 0 ,0 ,0 ,0 , 0.2]
EFgp['multi-day batteries','electricity'] = [80, 0 ,0 ,0 ,0 , 0.1]
EFgp['P2Xfuels','electricity'] = [0, 0, 0, 0, 50, 1/52]

slack = {'transport':'electricity', 'industry':'electricity',
            'buildings':'electricity', 'energy':'electricity', 
            'electricity':'gas electricity'}

scenarios['slow transition'] = [EFgp, slack]

### SLOW NUCLEAR TRANSITION ###

EFgp = {}
EFgp['oil','transport'] = [0.0, -5, 15, 30, 0, 1]
# EFgp['electricity','transport'] = [25, 5, 8, 20, 1, .8]
EFgp['P2Xfuels','transport'] = [35, 12, 12, 42, 1, .18]
EFgp['oil','industry'] = [0, -6, 15, 30, 0, 1]
EFgp['coal','industry'] = [0, -6, 10, 30, 0, 1]
EFgp['gas','industry'] = [2, -6, 15, 40, 0, 1]
# EFgp['electricity','industry'] = [7, 4, 8, 32, 1, .75]
EFgp['P2Xfuels','industry'] = [45, 12, 12, 42, 1, 0.2]
EFgp['oil','buildings'] = [0, -6, 15, 30, 0, 1]
EFgp['coal','buildings'] = [0, -5, 15, 30, 0, 1]
EFgp['gas','buildings'] = [2, -9, 15, 40, 0, 1]
# EFgp['electricity','buildings'] = [4, 8, 4, 30, 1, .895]
EFgp['P2Xfuels','buildings'] = [40, 10, 12, 42, 1, 0.1]
EFgp['coal electricity','electricity'] = [1, -10, 5, 30, 0, 1]
# EFgp['gas electricity','electricity'] = [2.5, 1.5, 10, 30, 0, 1]
EFgp['nuclear electricity','electricity'] = [8, 4, 20, 45, 0, .4]
EFgp['hydroelectricity','electricity'] = [3, 1, 10, 30, 0, 1]
EFgp['biopower electricity','electricity'] = [3, 3, 10, 20, 0, 1]
EFgp['wind electricity','electricity'] = [11, 5, 1, 30, 1, .22]
EFgp['solar pv electricity','electricity'] = [15, 5, 1, 40, 1, .28]
EFgp['EV batteries','electricity'] = [50, 0 ,0 ,0 ,0 ,0]
EFgp['daily batteries','electricity'] = [60, 0 ,0 ,0 ,0 , 0.0001]
EFgp['multi-day batteries','electricity'] = [80, 0 ,0 ,0 ,0 , 0.0001]
EFgp['P2Xfuels','electricity'] = [0, 0, 0, 0, 0, 0]

slack = {'transport':'electricity', 'industry':'electricity',
            'buildings':'electricity', 'energy':'electricity', 
            'electricity':'gas electricity'}

scenarios['slow nuclear transition'] = [EFgp, slack]


### HISTORICAL MIX ###

EFgp = {}
# EFgp['oil','transport'] = [0.0, -5, 15, 30, 0, 1]
EFgp['electricity','transport'] = [2, 2, 0, 1, 0, 1]
EFgp['P2Xfuels','transport'] = [2, 2, 0, 1, 0, 1]
EFgp['oil','industry'] = [2, 2, 0, 1, 0, 1]
EFgp['coal','industry'] = [2, 2, 0, 1, 0, 1]
# EFgp['gas','industry'] = [2, -6, 15, 40, 0, 1]
EFgp['electricity','industry'] = [2, 2, 0, 1, 0, 1]
EFgp['P2Xfuels','industry'] = [2, 2, 0, 1, 0, 1]
EFgp['oil','buildings'] = [2, 2, 0, 1, 0, 1]
EFgp['coal','buildings'] = [2, 2, 0, 1, 0, 1]
# EFgp['gas','buildings'] = [2, -9, 15, 40, 0, 1]
EFgp['electricity','buildings'] = [2, 2, 0, 1, 0, 1]
EFgp['P2Xfuels','buildings'] = [2, 2, 0, 1, 0, 1]
EFgp['coal electricity','electricity'] = [2, 2, 0, 1, 0, 1]
EFgp['gas electricity','electricity'] = [2, 2, 0, 1, 0, 1]
EFgp['nuclear electricity','electricity'] = [2, 2, 0, 1, 0, 1]
EFgp['hydroelectricity','electricity'] = [2, 2, 0, 1, 0, 1]
# EFgp['biopower electricity','electricity'] = [3, 3, 10, 20, 0, 1]
EFgp['wind electricity','electricity'] = [2, 2, 0, 1, 0, 1]
EFgp['solar pv electricity','electricity'] = [2, 2, 0, 1, 0, 1]
EFgp['EV batteries','electricity'] = [2, 0 ,0 ,0 ,0 ,0]
EFgp['daily batteries','electricity'] = [2, 0 ,0 ,0 ,0 , 0.0001]
EFgp['multi-day batteries','electricity'] = [2, 0 ,0 ,0 ,0 , 0.0001]
EFgp['P2Xfuels','electricity'] = [0, 0, 0, 0, 0, 0]

slack = {'transport':'oil', 'industry':'gas',
            'buildings':'gas', 'energy':'electricity', 
            'electricity':'biopower electricity'}

scenarios['historical mix'] = [EFgp, slack]

### COST PARAMETERS ###

costparams = {}
costparams['cgrid'] = 10.4 #  bn$(2020)/PWh
costparams['cTripleCap'] =  14.9 #  bn$(2020)/PWh
costparams['L'] = {}
costparams['c0'] = {}
costparams['z0'] = {}
costparams['mr'] = {}
costparams['k'] = {}
costparams['sigma'] = {}
costparams['omega'] = {}
costparams['sigmaOmega'] = {}
costparams['L']['nuclear electricity'] = 40
costparams['L']['hydroelectricity'] = 100
costparams['L']['biopower electricity'] = 30
costparams['L']['wind electricity'] = 30
costparams['L']['solar pv electricity'] = 30
costparams['L']['daily batteries'] = 12
costparams['L']['multi-day storage'] = 20
costparams['L']['electrolyzers'] = 10
costparams['c0']['oil (direct use)'] = 11.6
costparams['c0']['coal (direct use)'] = 1.84
costparams['c0']['gas (direct use)'] = 5.5
costparams['c0']['coal electricity'] = 16.7
costparams['c0']['gas electricity'] = 13.9
costparams['c0']['nuclear electricity'] = 25
costparams['c0']['hydroelectricity'] = 13
costparams['c0']['biopower electricity'] = 20
costparams['c0']['wind electricity'] = 11.3
costparams['c0']['solar pv electricity'] = 15.7
costparams['c0']['daily batteries'] = 86000
costparams['c0']['multi-day storage'] = 111100
costparams['c0']['electrolyzers'] = 364722
costparams['z0']['coal electricity'] = 1118
costparams['z0']['gas electricity'] = 514
costparams['z0']['nuclear electricity'] = 345
costparams['z0']['hydroelectricity'] = 519
costparams['z0']['biopower electricity'] = 37.3
costparams['z0']['wind electricity'] = 40
costparams['z0']['solar pv electricity'] = 13.3
costparams['z0']['daily batteries'] = 0.00422
costparams['z0']['multi-day storage'] = 10.8*1e-7
costparams['z0']['electrolyzers'] = 2.89*1e-7
costparams['mr']['oil (direct use)'] = 0.8128
costparams['k']['oil (direct use)'] = 0.4002
costparams['sigma']['oil (direct use)'] = 0.3037
costparams['mr']['coal (direct use)'] = 0.9499
costparams['k']['coal (direct use)'] = 0.0378
costparams['sigma']['coal (direct use)'] = 0.0902
costparams['mr']['gas (direct use)'] = 0.7455
costparams['k']['gas (direct use)'] = 0.4028
costparams['sigma']['gas (direct use)'] = 0.2617
costparams['mr']['coal electricity'] = 0.927
costparams['k']['coal electricity'] = 0.206
costparams['sigma']['coal electricity'] = 0.102
costparams['mr']['gas electricity'] = 0.827
costparams['k']['gas electricity'] = 0.485
costparams['sigma']['gas electricity'] = 0.131
costparams['omega']['nuclear electricity'] = 0.0
costparams['sigmaOmega']['nuclear electricity'] = 0.01
costparams['sigma']['nuclear electricity'] = 0.02
costparams['omega']['hydroelectricity'] = 0.0
costparams['sigmaOmega']['hydroelectricity'] = 0.01
costparams['sigma']['hydroelectricity'] = 0.01
costparams['omega']['biopower electricity'] = 0.05
costparams['sigmaOmega']['biopower electricity'] = 0.01
costparams['sigma']['biopower electricity'] = 0.02
costparams['omega']['wind electricity'] = 0.194
costparams['sigmaOmega']['wind electricity'] = 0.041
costparams['sigma']['wind electricity'] = 0.065
costparams['omega']['solar pv electricity'] = 0.319
costparams['sigmaOmega']['solar pv electricity'] = 0.043
costparams['sigma']['solar pv electricity'] = 0.111
costparams['omega']['daily batteries'] = 0.421
costparams['sigmaOmega']['daily batteries'] = 0.063
costparams['sigma']['daily batteries'] = 0.103
costparams['omega']['multi-day storage'] = 0.168
costparams['sigmaOmega']['multi-day storage'] = 0.041
costparams['sigma']['multi-day storage'] = 0.065
costparams['omega']['electrolyzers'] = 0.129
costparams['sigmaOmega']['electrolyzers'] = 0.067
costparams['sigma']['electrolyzers'] = 0.201

# set breakpoint params (in this case no breakpoints)
# breakpoints: [lognormal shape, lognormal scale, lognormal location]
# change in learning exponent at each breakpoint
# parameters: [mu=0, scale]
# breakpoint params:
# - breakpoint presence (True/False)
# - breakpoint  - lognormal distribution parameters 
# - change in learning exponent - normal parameters
costparams['breakpoints'] = {}
costparams['breakpoints']['active'] = [False]
costparams['breakpoints']['distance - lognormal'] = {}
costparams['breakpoints']['distance - lognormal']['shape'] = np.nan
costparams['breakpoints']['distance - lognormal']['scale'] = np.nan
costparams['breakpoints']['distance - lognormal']['loc'] = np.nan
costparams['breakpoints']['exponent change - normal'] = {}
costparams['breakpoints']['exponent change - normal']['mu'] = np.nan
costparams['breakpoints']['exponent change - normal']['scale'] = np.nan

### histElec has been generated 
### by running the Historical Mix scenario
### and saving the electricity demand

currentPath = os.path.dirname(__file__)

with open(currentPath+'/histElec.txt') as f:
    histElec = [float(x) for x in f.read().split('\t')[:-1]]
costparams['elecHist'] = histElec.copy()

learningRateTechs = ['nuclear electricity', 
                     'hydroelectricity', 
                     'biopower electricity',
                     'wind electricity', 
                     'solar pv electricity', 
                     'daily batteries',
                     'multi-day storage', 
                     'electrolyzers']

### obtain average experience curve params from Way et al. (2022)

# average learning rate
avgLR = []
for el in costparams['omega'].keys():
    if el in learningRateTechs and el not in ['nuclear electricity','hydroelectricity']:
        avgLR.append(costparams['omega'][el])
avgLR = np.mean(avgLR)

# standard error of learning rate
stderrLR = []
for el in costparams['sigmaOmega'].keys():
    if el in learningRateTechs and el not in ['nuclear electricity','hydroelectricity']:
        stderrLR.append(costparams['sigmaOmega'][el])
stderrLR = np.mean(stderrLR)

# standard deviation of errors
stdN = []
for el in costparams['sigma'].keys():
    if el in learningRateTechs and el not in ['nuclear electricity','hydroelectricity']:
        stdN.append(costparams['sigma'][el])
stdN = np.mean(stdN)


# copy params for alternative cost assumptions
costparams2 = copy.deepcopy(costparams)
costparams3 = copy.deepcopy(costparams)
costparams4 = copy.deepcopy(costparams)
costparams5 = copy.deepcopy(costparams)

# read alternative learning rate assumptions
expCurveParams = pd.read_csv(currentPath+'/ExpCurveParams.csv')

# modify learning rate assumptions
for t in learningRateTechs:

    if t in ['nuclear electricity','hydroelectricity']:
        continue

    ### all techs based
    costparams2['omega'][t] = \
         - expCurveParams.loc[\
            expCurveParams['Aggregation'] == 'All techs',\
                        'Mean slope'].values[0]
    costparams2['sigmaOmega'][t] = 0#\
    #     expCurveParams.loc[\
    #         expCurveParams['Aggregation'] == 'All techs',\
    #                     'Standard error of mean slope'].values[0]
    # costparams2['sigma'][t] = 0#\
    #     expCurveParams.loc[\
    #         expCurveParams['Aggregation'] == 'All techs',\
    #                     'Standard deviation of errors'].values[0]

    ### energy sector based
    costparams3['omega'][t] = \
        - expCurveParams.loc[\
            expCurveParams['Aggregation'] == 'Energy sector',\
                        'Mean slope'].values[0]
    costparams3['sigmaOmega'][t] = 0#\
    #     expCurveParams.loc[\
    #         expCurveParams['Aggregation'] == 'Energy sector',\
    #                     'Standard error of mean slope'].values[0]
    # costparams3['sigma'][t] = 0#\
    #     expCurveParams.loc[\
    #         expCurveParams['Aggregation'] == 'Energy sector',\
    #                     'Standard deviation of errors'].values[0]

    ### energy sector based - way et al average & worst case
    costparams4['omega'][t] = avgLR
    costparams4['sigmaOmega'][t] = 0#stderrLR
    # costparams4['sigma'][t] = 0#stdN

    ### energy sector based - without nuclear
    costparams5['omega'][t] = \
        - expCurveParams.loc[\
            expCurveParams['Aggregation'] == \
                'Energy sector without nuclear',\
                        'Mean slope'].values[0]
    costparams5['sigmaOmega'][t] = 0#\
    #     expCurveParams.loc[\
    #         expCurveParams['Aggregation'] == \
    #             'Energy sector without nuclear',\
    #                     'Standard error of mean slope'].values[0]
    # costparams5['sigma'][t] = 0#\
    #     expCurveParams.loc[\
    #         expCurveParams['Aggregation'] == \
    #             'Energy sector without nuclear',\
    #                     'Standard deviation of errors'].values[0]
    
# create labels different cost assumptions
costsAssumptions = {}
labels = ['Technology-specific - Way et al. (2022)',
          'Technology-mean - PCDB',
          'Technology-mean - PCDB (Energy), no nuclear',
          'Technology-mean - PCDB (Energy)',
          'Technology-mean - Way et al. (2022)']

# assign label to cost assumptions
costsAssumptions['Technology-specific - Way et al. (2022)'] = costparams
# costsAssumptions['Technology-mean - PCDB'] = costparams2
# costsAssumptions['Technology-mean - PCDB-Energy'] = costparams3
# costsAssumptions['Technology-mean - PCDB-Energy, no nuclear'] = costparams5
# costsAssumptions['Technology-mean - Way et al. (2022)'] = costparams4

# create breakpoint scenario

costparams2 = copy.deepcopy(costparams)

# set breakpoint params (in this case breakpoints)

costparams2['breakpoints'] = {}
costparams2['breakpoints']['active'] = [True]
costparams2['breakpoints']['distance - lognormal'] = {}
costparams2['breakpoints']['distance - lognormal']['shape'] = 0.6075000635903898
costparams2['breakpoints']['distance - lognormal']['loc'] = -0.14027836943191924
costparams2['breakpoints']['distance - lognormal']['scale'] = 0.8774674628197169
costparams2['breakpoints']['exponent change - normal'] = {}
costparams2['breakpoints']['exponent change - normal']['mu'] = 0
costparams2['breakpoints']['exponent change - normal']['scale'] = 0.15243863713865616-0


costsAssumptions['Breakpoint - Way et al. (2022)'] = costparams2
