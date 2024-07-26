import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

# model class
class EnergyModel:

    # initialize the model with data
    def __init__(self, EFgp, slack):
        # growth rates of technologies
        self.EFgp = EFgp
        
        # initialize technologies, carriers, and sectors
        self.technology = ['oil (direct use)','coal (direct use)',
              'gas (direct use)','coal electricity',
              'gas electricity','nuclear electricity',
              'hydroelectricity','biopower electricity',
              'wind electricity','solar pv electricity',
              'daily batteries','multi-day storage',
              'electrolyzers','electricity networks', 'P2X']
        self.carrier = ['oil','coal','gas','electricity','P2Xfuels']
        self.sector = ['transport', 'industry', 'buildings', 'energy']

        # mapping inputs
        # technology to carriers 
        # carriers to sectors
        self.carrierInputs = [[0],[1],[2],[3,4,5,6,7,8,9]]
        self.sectorInputs = [[0,3,4],[0,1,2,3,4],[0,1,2,3,4],[3]]


        # define slack variables per sector - no transition below
        self.slack = slack

        # define final year of simulation
        self.yend = 2100
        
        # this is step 1 in Way et al. (2022) SM page 26
        # 1) Specify the quantity of useful energy 
        # consumed in each sector
        # useful energy demand in EJ - exogenously defined
        # year for which data are given

        # initial year
        self.y0 = 2020

        # initialize demand dict, 
        # each key is a sector and the item is an array
        self.demand = {}
        for s in self.sector:
            self.demand[s] = np.zeros(self.yend - self.y0 + 1)
        
        # set initial demand values
        self.demand['transport'][0] = 28.6
        self.demand['industry'][0] = 70.6
        self.demand['buildings'][0] = 72.7
        self.demand['energy'][0] = 15.5

        # set demand growth rate and compute demand trajectory
        self.dgrowth = 0.02
        for y in range(self.y0, self.yend):
            for d in self.demand.keys():
                self.demand[d][y+1-self.y0] = \
                    self.demand[d][y-self.y0] * \
                    (1 + self.dgrowth)

        # set efficiencies from carrier to sector
        self.efficiency = {}
        for s in self.sector:
            for c in self.carrier:
                self.efficiency[c,s] = 0.0
        self.efficiency['oil','transport'] = 0.25
        self.efficiency['electricity','transport'] = 0.8
        self.efficiency['P2Xfuels','transport'] = 0.5
        self.efficiency['oil','industry'] = 0.6
        self.efficiency['coal','industry'] = 0.6
        self.efficiency['gas','industry'] = 0.6
        self.efficiency['electricity','industry'] = 0.8
        self.efficiency['P2Xfuels','industry'] = 0.6
        self.efficiency['oil','buildings'] = 0.7
        self.efficiency['coal','buildings'] = 0.6
        self.efficiency['gas','buildings'] = 0.6
        self.efficiency['electricity','buildings'] = 1.0
        self.efficiency['P2Xfuels','buildings'] = 0.6
        self.efficiency['oil','energy'] = 0.6
        self.efficiency['coal','energy'] = 0.6
        self.efficiency['gas','energy'] = 0.6
        self.efficiency['electricity','energy'] = 1.0

        # create dict for final energy supply in EJ
        # each key is a tuple (carrier, sector) and the item is an array
        self.EF = {}
        for s in self.sector:
            for c in self.carrier:
                self.EF[c,s] = np.zeros(self.yend - self.y0 + 1)

        # initialize final energy supply values
        self.EF['oil','transport'][0] = 110.1
        self.EF['electricity','transport'][0] = 1.34
        self.EF['P2Xfuels','transport'][0] = 2.95e-4
        self.EF['oil','industry'][0] = 12.5
        self.EF['coal','industry'][0] = 33.3
        self.EF['gas','industry'][0] = 27.0
        self.EF['electricity','industry'][0] = 33.6
        self.EF['P2Xfuels','industry'][0] = 2.95e-4
        self.EF['oil','buildings'][0] = 13.8
        self.EF['coal','buildings'][0] = 5.23
        self.EF['gas','buildings'][0] = 29.3
        self.EF['electricity','buildings'][0] = 42.3
        self.EF['P2Xfuels','buildings'][0] = 2.95e-4
        self.EF['oil','energy'][0] = 13.9#[13.9]
        self.EF['coal','energy'][0] = 17.1#[17.1]
        self.EF['gas','energy'][0] = 16.5#[16.5]
        self.EF['electricity','energy'][0] = 15.5

        # useful to account for VRE intermittency
        # self.elec represent the total electricity generation
        self.elec = np.zeros(self.yend - self.y0 + 1)


        # create dict of useful energy supply in EJ
        # each key is a tuple (carrier, sector) and the item is an array
        self.EU = {}
        for s in self.sector:
            for c in self.carrier:
                self.EU[c,s] = np.zeros(self.yend - self.y0 + 1)
                self.EU[c,s][0] = \
                    self.EF[c,s][0] * self.efficiency[c,s]

        # set initial values for useful energy
        self.EU['oil','transport'][0] = 27.5
        self.EU['electricity','transport'][0] = 1.07
        self.EU['P2Xfuels','transport'][0] = 1.48e-4
        self.EU['oil','industry'][0] = 7.5
        self.EU['coal','industry'][0] = 20.0
        self.EU['gas','industry'][0] = 16.2
        self.EU['electricity','industry'][0] = 26.9
        self.EU['P2Xfuels','industry'][0] = 1.77e-4
        self.EU['oil','buildings'][0] = 9.7
        self.EU['coal','buildings'][0] = 3.14
        self.EU['gas','buildings'][0] = 17.6
        self.EU['electricity','buildings'][0] = 42.3
        self.EU['P2Xfuels','buildings'][0] = 1.77e-4
        self.EU['oil','energy'][0] = 8.3
        self.EU['coal','energy'][0] = 10.3
        self.EU['gas','energy'][0] = 9.9
        self.EU['electricity','energy'][0] = 15.5

        # cap on maximum fraction of useful energy 
        # provided by electricty to each sector
        self.xi = {}
        self.xi['electricity','transport'] = 0.8
        self.xi['electricity','industry'] = 0.75
        self.xi['electricity','buildings'] = 0.9

        # efficiency of fossil fuel power plants
        self.zeta = {}
        self.zeta['coal electricity'] = 0.4
        self.zeta['gas electricity'] = 0.5

        # create dict for energy produced by technology
        # each key is a technology and the item is an array
        self.q = {}
        for t in self.technology:
            self.q[t] = np.zeros(self.yend - self.y0 + 1)

        # set initial values for energy produced by technology
        self.q['coal electricity'][0] = 35.7
        self.q['gas electricity'][0] = 22.9
        self.q['nuclear electricity'][0] = 10.0
        self.q['hydroelectricity'][0] = 15.2
        self.q['biopower electricity'][0] = 2.55
        self.q['wind electricity'][0] = 5.75
        self.q['solar pv electricity'][0] = 3.0
        self.q['daily batteries'][0] = 2.23/1000
        self.q['multi-day storage'][0] = 10.8*1e-7
        self.q['electrolyzers'][0] = 0.0001
        self.q['electricity networks'][0] = 0.0001
        self.q['qgrid'] = np.zeros(self.yend - self.y0 + 1)
        self.q['qtransport'] = np.zeros(self.yend - self.y0 + 1)
        self.q['P2X'] = np.zeros(self.yend - self.y0 + 1)
        self.q['P2X'][0] = 3 * 2.95e-4
        self.q['qgrid'][0] = 0.17/1000
        self.q['qtransport'][0] = 2.06/1000

        # this array is used to represent phase in of P2X fuels
        self.piP2X = np.zeros(self.yend - self.y0 + 1)
        self.piP2X[0] = 1e-10

        # create dict for cost of energy produced by technology
        # considering when the technology is built
        self.Q = {}
        for t in self.technology:
            self.Q[t] = \
                np.zeros((self.yend - self.y0 + 1, 
                          self.yend - self.y0 + 1))


    ### plotting functions
    def plotDemand(self):
        fig, ax = plt.subplots(figsize=(8,6))
        df = pd.DataFrame(self.demand, 
                          index=range(self.y0,self.yend + 1), 
                          columns=self.demand.keys())
        df.plot.area(stacked=True, lw=0, ax=ax)
        plt.xlim(2018,2075)
        plt.ylabel('EJ')
        plt.xlabel('Year')
        plt.title('Demand')

    def plotFinalEnergyBySource(self):
        colors = ['black','saddlebrown','darkgray',
                  'saddlebrown','darkgray',
                  'magenta','royalblue',
                  'forestgreen','deepskyblue',
                  'orange','pink','plum','lawngreen', 'burlywood'] 
        df = pd.DataFrame(self.q, 
                          index=range(self.y0,self.yend + 1),
                          columns = self.q.keys())
        cols = df.columns[[not(x) in ['qgrid','qtransport',
                                      'electricity networks',
                                      'electrolyzers'] 
                                      for x in df.columns]]
        df = df[cols]

        fig, ax = plt.subplots(figsize=(8,6))
        df.plot.area(stacked=True, lw=0, ax=ax,
                     color=colors, legend=False)
        ax.plot(range(self.y0,self.yend+1), 
                 [sum(
                     [sum(
                         [self.EF[c,s][y - self.y0] 
                          for c in self.carrier]
                         ) for s in self.sector]
                         ) for y in range(self.y0, self.yend+1)],
                           'k--', lw=2)
        ax.set_title('Final energy by source')
        ax.set_xlim(2018,2075)
        ax.set_ylim(0,800)
        ax.set_ylabel('EJ')
        ax.set_xlabel('Year')

        fig, ax = plt.subplots(figsize=(8,6))
        df = df.loc[df.index<2071]
        df = df[[x for x in df.columns if 'direct use' not in x]]
        df.plot(color=colors[3:], legend=False, ax=ax)
        ax.set_yscale('log', base=10)
        ax.set_ylim(1e-7, 1e3)
        ax.set_ylabel('EJ')
        ax.set_xlabel('Year')
        ax.set_title('Electricity generation, storage'
                     ' capacity and P2X production')

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(100*(self.q['P2X'][1:]/self.q['P2X'][:-1]-1))
        ax.set_ylim(-20, 120)
        ax.set_ylabel('%')
        ax.set_xlabel('Year')
        ax.set_title('P2X growth rate check')

        fig, ax = plt.subplots(figsize=(8,6))
        df.plot.area(stacked=True, lw=0, ax=ax,
                     color=colors[3:], legend=False)
        ax.plot(\
            range(self.y0+1,2071), 
            np.sum(np.array(\
                [self.EF['electricity',s][1:51] for s in self.sector]
                            ), axis=0), 'k--')
        ax.set_ylim(0, 600)
        ax.set_ylabel('EJ')
        ax.set_xlabel('Year')
        ax.set_title('Electricity generation by primary source')

        fig, ax = plt.subplots(figsize=(8,6))
        df['tot'] = df[df.columns[:-1]].sum(axis=1)
        df[df.columns[:-2]] = \
            100*df[df.columns[:-2]].div(df['tot'], axis=0)
        df[df.columns[:-2]].plot.area(\
            stacked=True, lw=0, 
            color=colors[3:], legend=False, ax=ax)
        ax.set_ylim(0, 150)
        ax.set_ylabel('%')
        ax.set_xlabel('Year')
        ax.set_title('Electricity gen. mix by primary source')


        fig, ax = plt.subplots(figsize=(8,6))
        df[df.columns[:-2]].plot(color=colors[3:], legend=False, ax=ax)
        df['vre'] = df['solar pv electricity'] + df['wind electricity']
        df['non-vre'] = 100 - df['vre']
        df['non-vre'].plot(color='k', lw=2, ls=':', ax=ax)
        df['vre'].plot(color='k', lw=2, ls='--', ax=ax)
        ax.set_ylim(0, 110)
        ax.set_ylabel('%')
        ax.set_xlabel('Year')
        ax.set_title('Electricity gen. mix by primary source')


    def plotCostBySource(self):

        colors = ['black','saddlebrown','darkgray',
                  'saddlebrown','darkgray',
                  'magenta','royalblue',
                  'forestgreen','deepskyblue',
                  'orange','pink','plum','lawngreen', 'burlywood']
        
        df = pd.DataFrame(self.C, 
                          index=range(self.y0,self.yend + 1),
                          columns = self.C.keys())
        
        cols = df.columns[[not(x) in ['qgrid','qtransport',
                                      'P2X'] for x in df.columns]]
        df = df[cols]

        # from usd to trillion USD
        df[cols] = df[cols] * 1e-12

        # from billion to trillion
        df['electricity networks'] = self.gridInv * 1e-3 
        
        fig, ax = plt.subplots(figsize=(8,6))
        df.plot.area(stacked=True, lw=0, ax=ax,
                     color=colors, legend=False)
        ax.plot(range(self.y0,self.yend+1), 
                 [sum(
                     [sum(
                         [self.EF[c,s][y - self.y0] 
                          for c in self.carrier]
                         ) for s in self.sector]
                         ) for y in range(self.y0, self.yend+1)], 
                         'k--', lw=2,
                         )
        ax.set_title('Final cost by source')
        ax.set_xlim(2018,2075)
        ax.set_ylim(0,12)
        ax.set_ylabel('Trillion USD')
        ax.set_xlabel('Year')

        df = pd.DataFrame(self.c, index=range(self.y0,self.yend + 1),
                            columns = self.c.keys())
        cols = df.columns[[not(x) in ['qgrid','qtransport',
                                      'P2X'] for x in df.columns]]
        df = df[cols]

        fig, ax = plt.subplots(figsize=(8,6))
        df.plot(color=colors, legend=False, ax=ax)
        ax.set_title('LCOE')
        ax.set_ylabel('USD/GJ')
        ax.set_xlabel('Year')
        ax.set_yscale('log', base=10)


    def plotFinalEnergy(self):
        df = pd.DataFrame(self.EF, 
                          index=range(self.y0,self.yend + 1), 
                            columns=self.EF.keys())
        df.columns = \
            [str(a[0]+'_'+a[1]) for a in df.columns.to_flat_index()]
        for s in self.sector:
            df_ = df[[x for x in df.columns if x.split('_')[1]==s]].copy()
            df_.plot.area(stacked=True, lw=0, 
                          color=['black','saddlebrown',
                                 'darkgray','lightskyblue','lime'])
            plt.title('Final energy - '+s)
            plt.xlim(2018,2075)
            plt.ylim(0,300)
            plt.ylabel('EJ')
            plt.xlabel('Year')

    def plotS7(self):

        colors = ['black','saddlebrown',
                  'darkgray','lightskyblue','lime']

        fig, ax = plt.subplots(4,4, sharex=True, figsize=(14,10))

        counts = 0
        for s in self.sector:
            countc = 0
            for c in self.carrier:
                ax[counts][0].plot(range(self.y0+1,self.yend+1), 
                            100 * \
                                (self.EU[c,s][1:] / \
                                 (1e-16+self.EU[c,s][:-1])
                                   - 1),
                            color=colors[countc] )
                countc += 1
            counts += 1
        
        [ax[x][0].set_ylim(-20, 80) for x in range(4)]
        [ax[x][0].set_xlim(2018, 2075) for x in range(4)]
        ax[0][0].set_title('Useful energy growth rates [%]')
        ax[0][0].set_ylabel('Transport')
        ax[1][0].set_ylabel('Industry')
        ax[2][0].set_ylabel('Buildings')
        ax[3][0].set_ylabel('Energy sector')


        counts = 0
        for s in self.sector:
            countc = 0
            for c in self.carrier:
                ax[counts][1].plot(range(self.y0+1,2071), 
                                   self.EU[c,s][1:51],
                                    color=colors[countc] )
                countc += 1
            if not s=='energy':
                ax[counts][1].plot(range(self.y0+1,2071), 
                                sum([self.EU[c,s][1:51] 
                                     for c in self.carrier]),
                                'k--' )
            
            counts += 1
        
        [ax[x][1].set_ylim(1e-4, 1e3) for x in range(4)]
        [ax[x][1].set_yscale('log', base=10) for x in range(4)]
        ax[0][1].set_title('Useful energy [EJ]')


        counts = 0
        df = pd.DataFrame(self.EU, 
                          index=range(self.y0, self.yend + 1), 
                columns=self.EU.keys())
        df.columns = \
            [str(a[0]+'_'+a[1]) for a in df.columns.to_flat_index()]
        df = df.loc[(df.index>2020) & (df.index<2071)]
        for s in self.sector:
            df_ = df[[x for x in df.columns if x.split('_')[1]==s]].copy()
            df_.plot.area(stacked=True, lw=0, 
                          color=colors, ax=ax[counts][2],
                          legend=False)
            counts += 1
        [ax[x][2].set_ylim(0, 300) for x in range(4)]
        ax[0][2].set_title('Useful energy [EJ]')

        counts = 0
        df = pd.DataFrame(self.EF, 
                          index=range(self.y0, self.yend + 1), 
                columns=self.EF.keys())
        df.columns = \
            [str(a[0]+'_'+a[1]) for a in df.columns.to_flat_index()]
        df = df.loc[(df.index>2020) & (df.index<2071)]

        for s in self.sector:
            df_ = df[[x for x in df.columns if x.split('_')[1]==s]].copy()
            df_.plot.area(stacked=True, lw=0, 
                          color=colors, ax=ax[counts][3],
                          legend=False)
            counts += 1
        [ax[x][3].set_ylim(0, 300) for x in range(4)]
        ax[0][3].set_title('Final energy [EJ]')

        fig.subplots_adjust(hspace=0.3, wspace=0.3, 
                            top=0.95, bottom=0.075,
                            left=0.07, right=0.95)


    # simulation
    def simulate(self):
        for y in range(self.y0,self.yend):

            # this is step 2 from Way et al. (2022) SM page 26
            # 2 Specify the quantities of energy carriers 
            # used in each sectorfor each sector
            for s in self.sector:
                # get the defined slack variable
                sl = self.slack[s]

                # for each carrier
                carrs = [self.carrier[x] for x in \
                         self.sectorInputs[self.sector.index(s)]]

                for c in carrs:
                    # if carrier not slack 
                    if not (c==sl):

                        # get growth rate parameters
                        try:
                            gt0, gT, t1, t2, t3, psi = self.EFgp[c,s]
                        # if growth parameters not available, 
                        # consider no useful energy from that carrier
                        except KeyError:
                            self.EU[c,s][y+1-self.y0] = 0.0
                            continue

                        # compute growth rate
                        if y - self.y0 < t1:
                            gt = 0.01 * gt0
                        elif y - self.y0 >= t1 and y - self.y0 < t2:
                            s_ = 50 * np.abs(0.01*(gT-gt0)/(t2-t1))
                            gt = 0.01 * gt0 + \
                                0.01 * (gT - gt0) / \
                                    (1 + \
                                     np.exp(\
                                         -s_*(y - self.y0 -\
                                               t1 - (t2-t1)/2)))
                        else:
                            gt = 0.01 * gT

                        # compute useful energy
                        EUp = self.EU[c,s][y-self.y0]
                        if c == 'electricity':
                            maxcap = self.xi[c,s] * \
                                self.demand[s][y+1-self.y0]
                        else:
                            maxcap = psi * self.demand[s][y+1-self.y0]
                        EUf = min(EUp * (1 + gt), maxcap)
                        
                        # backcalculate final energy
                        self.EU[c,s][y+1-self.y0] = EUf
                        self.EF[c,s][y+1-self.y0] = \
                            self.EU[c,s][y+1-self.y0] / self.efficiency[c,s]

                # compute useful energy from slack carrier
                self.EU[sl,s][y+1-self.y0] = \
                    max(0, self.demand[s][y+1-self.y0] - \
                        sum([self.EU[c,s][y+1-self.y0] \
                             for c in carrs if c != sl]))
                self.EF[sl,s][y+1-self.y0] = \
                    self.EU[sl,s][y+1-self.y0] / self.efficiency[sl,s]

            # get slack technology for electricity carrier
            sl = self.slack[self.carrier[self.carrier.index('electricity')]]
            # get total electricity generated
            self.elec[y+1-self.y0] = \
                sum([self.EF['electricity',s][y+1-self.y0] \
                        for s in self.sector])

            # 3 Specify the proportion of 
            # non-dispatchable generation in the electricity mix
            # 4 Increase P2X fuel production 
            # to account for VRE intermittency

            # compute P2X fuel needs
            gt0, gT, t1, t2, t3, psi = \
                        self.EFgp['P2Xfuels','electricity']  
            gt0 = 0
            gT = 20
            t1 = 13
            t2 = t3
            # compute growth rate
            if y - self.y0 < t1:
                gt = 0.01 * gt0
            elif y - self.y0 >= t1 and y - self.y0 < t2:
                s_ = 50 * np.abs(0.01*(gT-gt0)/(t2-t1))
                gt = 0.01 * gt0 +\
                    0.01 * (gT - gt0) / \
                        (1+np.exp(\
                            -s_*(y - self.y0 - t1 - (t2-t1)/2)))
            else:
                gt = 0.01 * gT
            # adjust P2X fuel production
            self.piP2X[y+1-self.y0] = min(5*gt,1)

            # compute P2X fuel production
            self.q['P2X'][y+1-self.y0] = \
                sum([self.EF['P2Xfuels',s][y+1-self.y0] \
                            for s in self.sector]) + \
                2 * psi * \
                    (self.EFgp['solar pv electricity','electricity'][-1] + \
                        self.EFgp['wind electricity','electricity'][-1]) * \
                    self.elec[y+1-self.y0] * \
                    min(self.piP2X[y+1-self.y0], 1)

            # 5 Increase total electricity generation if required
            # to account for electrolytic production of P2X fuels.
            self.elec[y+1-self.y0] = \
                self.elec[y+1-self.y0] - \
                    self.EFgp['P2Xfuels','electricity'][-1] * \
                    (self.EFgp['solar pv electricity','electricity'][-1] + \
                        self.EFgp['wind electricity','electricity'][-1]) * \
                    self.elec[y+1-self.y0] * \
                    min(self.piP2X[y+1-self.y0], 1) + \
                1/0.7 * self.q['P2X'][y+1-self.y0]
            

            # 6 Specify the quantity of electricity produced 
            # by each electricity generation technology`

            # allocate electricity to technologies
            for t in [self.technology[x] \
                      for x in self.carrierInputs\
                            [self.carrier.index('electricity')]]:
                if not (t==sl):

                    # get growth rate parameters
                    try:
                        gt0, gT, t1, t2, t3, psi = \
                            self.EFgp[t,'electricity']
                    # if growth parameters not available, 
                    # consider no useful energy from that carrier
                    except KeyError:
                        self.q[t][y+1-self.y0] = 0.0
                        continue

                    # compute growth rate
                    if y - self.y0 < t1:
                        gt = 0.01 * gt0
                    elif y - self.y0 >= t1 and y - self.y0 < t2:
                        s_ = 50 * np.abs(0.01*(gT-gt0)/(t2-t1))
                        gt = 0.01 * gt0 + \
                            0.01 * (gT - gt0) / \
                                (1+np.exp(-s_*\
                                          (y - self.y0 - t1 - (t2-t1)/2)))
                    else:
                        gt = 0.01 * gT
                    
                    # compute generation from technology
                    qp = self.q[t][y-self.y0]
                    if y - self.y0 > t3:
                        maxcap = psi * self.elec[y+1-self.y0]
                    else:
                        maxcap = self.elec[y+1-self.y0]
                    qf = min(qp * (1 + gt), maxcap)
                    self.q[t][y+1-self.y0] = qf
            
            # compute electricity slack generation
            self.q[sl][y+1-self.y0] = \
                max(0, self.elec[y+1-self.y0] - \
                    sum([self.q[t][y+1-self.y0] for t in \
                        [self.technology[x] for x in \
                            self.carrierInputs\
                                [self.carrier.index('electricity')]] \
                                    if t != sl]))


            # 7 Calculate the quantities of fossil fuel energy carriers
            #  required by the energy sector

            # get the value of fossil fuel for end use and electricty
            ff_eu_elec2020 = \
                self.EF['oil','transport'][2020-self.y0] + \
                sum([sum([self.EF[c,s][2020-self.y0] \
                          for s in self.sector[1:3]]) \
                            for c in self.carrier[:3]]) + \
                self.q['coal electricity'][2020-self.y0] / \
                    self.zeta['coal electricity'] + \
                self.q['gas electricity'][2020-self.y0] / \
                    self.zeta['gas electricity'] 


            ff_eu_elec = \
                self.EF['oil','transport'][y+1-self.y0] + \
                sum([sum([self.EF[c,s][y+1-self.y0] \
                          for s in self.sector[1:3]]) \
                            for c in self.carrier[:3]]) + \
                self.q['coal electricity'][y+1-self.y0] / \
                    self.zeta['coal electricity'] + \
                self.q['gas electricity'][y+1-self.y0] / \
                    self.zeta['gas electricity'] 

            # derive energy sector demand for fossil fuels 
            self.EF['oil','energy'][y+1-self.y0] = \
                13.9 / (13.9 + 17.1 + 16.5) * \
                    ff_eu_elec * (13.9+17.1+16.5) / \
                        ff_eu_elec2020
            self.EF['coal','energy'][y+1-self.y0] = \
                17.1 / (13.9 + 17.1 + 16.5) * \
                    ff_eu_elec * (13.9+17.1+16.5) / \
                        ff_eu_elec2020
            self.EF['gas','energy'][y+1-self.y0] = \
                16.5 / (13.9 + 17.1 + 16.5) * \
                    ff_eu_elec * (13.9+17.1+16.5) / \
                        ff_eu_elec2020
            self.EU['oil','energy'][y+1-self.y0] = \
                self.EF['oil','energy'][y+1-self.y0] * \
                    self.efficiency['oil','energy']
            self.EU['gas','energy'][y+1-self.y0] = \
                self.EF['gas','energy'][y+1-self.y0] * \
                    self.efficiency['gas','energy']
            self.EU['coal','energy'][y+1-self.y0] = \
                self.EF['coal','energy'][y+1-self.y0] * \
                    self.efficiency['coal','energy']
            
            # compute direct use of fossil fuels
            self.q['oil (direct use)'][y+1-self.y0] = \
                sum([self.EF['oil', s][y+1-self.y0] \
                            for s in self.sector])
            self.q['coal (direct use)'][y+1-self.y0] = \
                sum([self.EF['coal', s][y+1-self.y0] \
                            for s in self.sector])
            self.q['gas (direct use)'][y+1-self.y0] = \
                sum([self.EF['gas', s][y+1-self.y0] \
                            for s in self.sector])
            
            # 8 Calculate the quantity of daily-cycling batteries required
            # compute short term batteries
            gt0, gT, t1, t2, t3, psi = \
                self.EFgp['daily batteries','electricity']
            self.q['qgrid'][y+1 - self.y0] = \
                min((1 + 0.01*gt0) * self.q['qgrid'][y-self.y0], 
                    psi/365*(self.q['solar pv electricity'][y+1-self.y0] + \
                                self.q['wind electricity'][y+1-self.y0]))
            gt0, gT, t1, t2, t3, psi = \
                self.EFgp['EV batteries','electricity']
            self.q['qtransport'][y+1-self.y0] = \
                min((1 + 0.01*gt0) * self.q['qtransport'][y-self.y0], 
                    1/365 * \
                        (self.EF['electricity','transport'][y+1-self.y0]))
            self.q['daily batteries'][y+1-self.y0] = \
                self.q['qgrid'][y+1-self.y0] + \
                    self.q['qtransport'][y+1-self.y0]

            # 9 Calculate the quantity of 
            # multi-day storage batteries required
            
            # compute long term storage
            gt0, gT, t1, t2, t3, psi = \
                self.EFgp['multi-day batteries','electricity']
            self.q['multi-day storage'][y+1-self.y0] = \
                min((1 + 0.01*gt0) * \
                            self.q['multi-day storage'][y-self.y0], \
                    psi/365 * (\
                        self.q['solar pv electricity'][y+1-self.y0] + \
                        self.q['wind electricity'][y+1-self.y0]))
            
            # 10 Calculate the quantity 
            # of electrolyzers required for P2X fuel production
            # compute electrolyzers
            self.q['electrolyzers'][y+1-self.y0] = \
                1/(24*365*0.5*0.7) * self.q['P2X'][y+1-self.y0]



    def computeCost(self, costparams, learningRateTechs=None):

        if learningRateTechs == None:
            self.learningRateTechs = self.technology[5:13]
        else:
            self.learningRateTechs = learningRateTechs

        # initialize cost dictionaries
        self.gridInv = np.zeros(self.yend - self.y0 + 1)
        self.c = {}
        for t in self.technology:
            self.c[t] = np.zeros(self.yend - self.y0 + 1)
        self.C = {}
        for t in self.technology:
            self.C[t] = np.zeros(self.yend - self.y0 + 1)

        # grid costs ( in billion USD )
        for y in range(self.y0, self.yend+1):
            self.gridInv[y-self.y0] = \
                1/3.6*costparams['cgrid'] * \
                    (costparams['elecHist'][y-self.y0] - \
                1/2 * (self.elec[y-self.y0] - \
                        costparams['elecHist'][y-self.y0])) + \
                1 / 3.6 * costparams['cTripleCap'] * \
                    1/2 * (self.elec[y-self.y0] - \
                           costparams['elecHist'][y-self.y0])

        # compute vintaging model
        for t in self.learningRateTechs:
            self.Q[t][0][0] = self.q[t][0]
            for y in range(self.y0+1, self.yend):
                for tau in range(self.y0, y+1):
                    if tau-self.y0 == 0:
                        self.Q[t][y-self.y0][tau-self.y0] = \
                            max(0, 
                                self.q[t][0] * \
                                    (1 - (y-self.y0)/costparams['L'][t]))
                    if tau == y and y-self.y0 > 0:
                        self.Q[t][y-self.y0][tau-self.y0] = \
                            max(0, 
                                self.q[t][y-self.y0] - \
                                    sum([\
                                        self.Q[t][y-self.y0][tau_-self.y0] \
                                            for tau_ in range(self.y0,y)]))
                    if tau < y and tau - self.y0 > 0 and \
                        y - self.y0 > 0 and \
                            y - self.y0 <= costparams['L'][t]:
                        self.Q[t][y-self.y0][tau-self.y0] = \
                                    self.Q[t][y-self.y0-1][tau-self.y0]
                    if tau < y and tau - self.y0 > 0 and \
                        y-self.y0 > costparams['L'][t]:
                        self.Q[t][y-self.y0][tau-self.y0] = \
                            max(0, 
                                self.Q[t][y-self.y0-1][tau-self.y0] - \
                                    self.Q[t]\
                                        [y-self.y0-costparams['L'][t]]\
                                            [tau-self.y0])
            
            # normalization over generation produced
            # assume generation is equally spread 
            # over the different years of installation
            # (needed if generation is decreasing over time, rarely)
            if t in self.learningRateTechs:
                for y in range(self.y0+1, self.yend):
                    self.Q[t][y-self.y0] = \
                        self.Q[t][y-self.y0] * \
                            self.q[t][y-self.y0] / \
                                sum(self.Q[t][y-self.y0])
        
        # compute unit cost of technologies

        #initialize unit cost and cumulative production dictionaries
        self.u = {}
        self.z = {}
        self.omega = {}

        # for each technology
        for t in self.technology[:13]:
        
            #initialize production and unit cost arrays and 
            self.z[t] = np.zeros(self.yend - self.y0 + 1)
            self.omega[t] = 0.0
            self.u[t] = np.zeros(self.yend - self.y0 + 1)

            # if breakpoints are used, initialize breaks
            if costparams['breakpoints']['active'][0]:
                self.breaks = []
                self.lrchanges = []
        
            # iterate over the years
            for y in range(self.y0, self.yend):

                # if year 0, initialize unit cost
                if y == self.y0:
                    self.c[t][0] = costparams['c0'][t]

                    # if experience curves are used
                    # get initial cumulative production
                    # and sample uncertain parameters
                    if t in self.learningRateTechs:
                        self.z[t][0] = costparams['z0'][t]
                        self.omega[t] = np.random.normal(\
                            costparams['omega'][t], \
                                costparams['sigmaOmega'][t])

                        # sample first breakpoint
                        if costparams['breakpoints']['active'][0]:
                            self.breaks.append(\
                                self.z[t][0] * \
                                    10**scipy.stats.lognorm.rvs(\
                                        costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['shape'], 
                                        costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['loc'], 
                                        costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['scale'], )
                            )
                            self.lrchanges.append(scipy.stats.norm.rvs(\
                                costparams['breakpoints']
                                            ['exponent change - normal']['mu'], \
                                    costparams['breakpoints']
                                            ['exponent change - normal']['scale']))
                            

                
                # if not using experience curves
                # use ar1 model
                if t not in self.learningRateTechs:
                    self.c[t][y+1-self.y0] = \
                        np.exp(\
                            costparams['mr'][t] * \
                                np.log(self.c[t][y-self.y0]) + \
                            np.random.normal(0, \
                                            costparams['sigma'][t]) + \
                            costparams['k'][t])
                
                # if using learning curves, add production to cumulative
                # sample random disturbance and compute unit cost
                # using wright's law with autocorrelated errors
                else:
                    self.z[t][y+1-self.y0] = \
                        self.z[t][y-self.y0] + \
                            self.q[t][y-self.y0]
                    self.u[t][y+1-self.y0] = \
                        np.random.normal(0, 
                                        np.sqrt(\
                                            costparams['sigma'][t]**2 / \
                                                        (1 + 0.19**2) ) )
                    
                    # is the additional production crossing a breakpoint?
                    if costparams['breakpoints']['active'][0] and\
                        self.z[t][y+1-self.y0] > self.breaks[-1]:

                        ## store breakpoints and learning exponent change 
                        breaks = [self.breaks[-1]]
                        lrchanges = [self.lrchanges[-1]]
                        
                        ## sample until next cumulative production is met
                        while self.z[t][y+1-self.y0] > self.breaks[-1]:
                            self.breaks.append(\
                                self.breaks[-1] * \
                                    10**scipy.stats.lognorm.rvs(\
                                        costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['shape'], 
                                        costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['loc'], 
                                        costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['scale'], )
                                )
                            breaks.append(self.breaks[-1])
                            lrchanges.append(scipy.stats\
                                .norm.rvs(\
                                    costparams['breakpoints']
                                            ['exponent change - normal']['mu'], \
                                    costparams['breakpoints']
                                            ['exponent change - normal']['scale']) + \
                                -0.26585740971836214 * self.lrchanges[-1])
                            self.lrchanges.append(lrchanges[-1])
                        
                        ## compute from last production to first breakpoint
                        ## noise is added in this step as it depends on 
                        ## the previous production
                        tempc = np.exp((np.log(self.c[t][y-self.y0]) - \
                                        self.omega[t] * \
                                        np.log(breaks[0] / \
                                                self.z[t][y-self.y0]) + \
                                        self.u[t][y-self.y0] + \
                                            0.19 * self.u[t][y-self.y0]))
                        self.omega[t] += lrchanges[0]


                        ## compute from first to last breakpoint
                        for i in range(1,len(breaks)-1):
                            tempc = np.exp((np.log(tempc) - \
                                            self.omega[t] * \
                                            np.log(breaks[i] / \
                                                    breaks[i-1])))
                            self.omega[t] += lrchanges[i]




                        
                        ## compute from last breakpoint to current production
                        self.c[t][y+1-self.y0] = tempc


                    # if no breakpoints are used - standard learning curve
                    else:
                        self.c[t][y+1-self.y0] = \
                            np.exp((np.log(self.c[t][y-self.y0]) - \
                                    self.omega[t] * \
                                    np.log(self.z[t][y+1-self.y0] / \
                                            self.z[t][y-self.y0]) + \
                                    self.u[t][y+1-self.y0] + \
                                        0.19 * self.u[t][y-self.y0]))
            
            # if t in self.learningRateTechs:
            #     plt.figure()
            #     plt.plot(self.z[t], self.c[t], marker='.')
            #     plt.xscale('log')
            #     plt.yscale('log')
            #     if costparams['breakpoints']['active'][0]:
            #         [plt.axvline(br, ls='--') for br in self.breaks]
            #     plt.title(t + costparams['breakpoints']['active'][0]*' with breakpoints')



        # compute total cost of technologies
        for t in self.technology[:13]:
            for y in range(self.y0, self.yend+1):
                if t not in self.learningRateTechs:
                    self.C[t][y-self.y0] = \
                        self.c[t][y-self.y0] * \
                            1e9 * self.q[t][y-self.y0]
                elif self.technology.index(t) >= 10:
                    cfr = 0.08 * (1 + 0.08)**costparams['L'][t] / \
                            ((1 + 0.08)**costparams['L'][t] - 1)
                    self.C[t][y-self.y0] = \
                        sum([self.c[t][tau-self.y0] *\
                              1e9 * cfr * \
                                self.Q[t][y-self.y0][tau-self.y0] \
                                    for tau in range(self.y0, y+1)])
                else:
                    self.C[t][y-self.y0] = \
                        sum([self.c[t][tau-self.y0] * \
                             1e9 * self.Q[t][y-self.y0][tau-self.y0] \
                                    for tau in range(self.y0, y+1)])

        # adding grid costs (which are in billion USD)
        self.C[self.technology[13]] = self.gridInv * 1e9

        # compute total cost
        self.totalCost = np.zeros(self.yend - self.y0 + 1)
        for y in range(self.y0, self.yend+1):
            for t in self.technology[:14]:
                self.totalCost[y-self.y0] += self.C[t][y-self.y0]

        # compute discounted cost
        # discount rate is sate to be 2% (0.02) 
        # as in main case of Way et al. (2022)
        self.discountedCost = np.zeros(self.yend - self.y0 + 1)
        for y in range(self.y0, 2070+1):
            self.discountedCost[y-self.y0] = \
                np.exp( - 0.02 * (y - self.y0) ) * \
                            self.totalCost[y-self.y0]

        self.discountedCost = np.sum(self.discountedCost)

        return self.totalCost, self.discountedCost