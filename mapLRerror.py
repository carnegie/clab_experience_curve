import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('talk')
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'


def compute_cost_lr(lr, inter, p0, pend):

    p0 = 10**p0
    pend = 10**pend
    inter = 10**inter

    # if cost is c(p) = c0 * p^lr
    # is integral is c(p) = c0 * 1/(lr+1) * p^(lr+1)

    integral = inter * 1/(lr+1) * pend**(lr+1) - inter * 1/(lr+1) * p0**(lr+1)
    unitcost = inter * (pend/p0)**lr

    return integral, unitcost

def compute_cost_lr_lafond(lr, c0, p0, pend):

    p0 = 10**p0
    pend = 10**pend
    c0 = 10**c0

    # if cost is c(p) = c0 * (p/p0)^lr
    # the integral is c(p) = c0 * (1/p0)^lr * 1/(lr+1) * (p)^(lr+1)

    integral = c0 * (1/p0)**lr * 1/(lr+1) * pend**(lr+1) - \
                c0 * (1/p0)**lr * 1/(lr+1) * p0**(lr+1)
    unitcost = c0 * (pend/p0)**lr
    
    return integral, unitcost

def compute_cost_lr_lafond_error(lr, c0, p0, pend, lrerror):

    integral, uc = compute_cost_lr_lafond(lr, c0, p0, pend)
    integral_error, uc_error = compute_cost_lr_lafond(lr+lrerror, c0, p0, pend)
    integral_error = integral_error - integral
    uc_error = uc_error - uc

    return integral_error, uc_error

def compute_cost_lr_error(lr, inter, p0, pend, lrerror):
    
        integral, uc = compute_cost_lr(lr, inter, p0, pend)
        integral_error, uc_error = compute_cost_lr(lr+lrerror, inter, p0, pend)
        integral_error = integral_error - integral
        uc_error = uc_error - uc
    
        return integral_error, uc_error

def compute_cost_breakpoint(lr, c0, p0, pend, lrchange_std, bp_meandist):
    # compute cost for a model with a breakpoint
    # the breakpoint is defined by the mean distance between breakpoints
    # and the standard deviation of the change in learning rate
    # the learning rate is the same before and after the breakpoint

    p0 = 10**p0
    pend = 10**pend
    c0 = 10**c0

    # if pend>100:
    #     plt.figure()

    # if cost is c(p) = c0 * (p/p0)^lr
    # the integral is c(p) = c0 * (1/p0)^lr * 1/(lr+1) * (p)^(lr+1)

    integrals = []
    unitcosts = []
    breakpoints = []
    for i in range(1000):
        lr_ = lr
        p0_ = p0
        c0_ = c0
        integral = 0
        lrchange = 0
        # compute the breakpoint and learning rate change
        bp = 10**(np.log10(p0_)+np.random.exponential(1/bp_meandist))
        bps = []
        c0s = [c0_]
        p0s = [p0_]
        while bp < pend:
            bps.append(bp)

            integral += c0_ * (1/p0_)**lr_ * 1/(lr_+1) * bp**(lr_+1) - \
                        c0_ * (1/p0_)**lr_ * 1/(lr_+1) * p0_**(lr_+1)
            
            c0_ = c0_ * ((bp/p0_)**lr_)
            c0s.append(c0_)

            p0_ = bp
            p0s.append(p0_)
            lrchange = np.random.normal(0, lrchange_std)
            lr_ = lr_ + lrchange
            bp = 10**(np.log10(p0_)+np.random.exponential(1/bp_meandist))
        
        integral += c0_ * (1/p0_)**lr_ * 1/(lr_+1) * bp**(lr_+1) - \
                    c0_ * (1/p0_)**lr_ * 1/(lr_+1) * p0_**(lr_+1)
        unitcost = c0_ * ((pend/p0_)**lr_)

        p0s.append(pend)
        c0s.append(unitcost)

        # if pend>1000 and lr < 0:
        #     plt.plot(p0s, c0s)
        #     plt.plot([p0,pend],[c0,c0*(pend/p0)**lr])
        #     plt.xscale('log', base=10)
        #     plt.yscale('log', base=10)
        #     plt.show()

        breakpoints.append(bps)
        # if pend>100:
        #     plt.plot(breakpoints)
        integrals.append(integral)
        unitcosts.append(unitcost)
    integral = np.median(integrals)
    unitcost = np.median(unitcosts)

    # if pend>100:
        
    #     plt.show()
    
    return integral, unitcost


# this part is needed for integral cost calculation

# set ranges for initial production, final production, initial cost
# prod0 = np.arange(0, 7.1, 5)
# prod0 = 10**(prod0)
# prodend = np.arange(1.0, 7.1, 1)
# prodend = 10**(prodend)
# cost0 = np.arange(-3, 3.1, 5)
# cost0 = 10**(cost0)

# # set learning rates 
# lr = np.arange(-10, 41, 20)
# lr = np.log2(-(lr/100 - 1))

# # set learning rate errors
# lrerror = np.arange(-20, 21, 1)
# lrerror = np.log2(-(lrerror/100 - 1))

## this part is for unit cost error estimation

# set ranges for initial production, final production, initial cost
prod0 = np.array([0])
prod0 = 10**(prod0)
prodend = np.arange(0, 7.1, .1)
prodend = 10**(prodend)
cost0 = np.array([0])
cost0 = 10**(cost0)

# set learning rates 
lr = np.arange(0, 41, 20)
# lr = np.log2(-(lr/100 - 1))

# set learning rate errors
lrerror = np.arange(-20, 21, 1)
# lrerror = np.log2(-(lrerror/100 - 1))


# print(np.array([[100*(2**lr_ - 2**(lr_+lrerror_)) for lr_ in lr] for lrerror_ in lrerror]))
# print(np.array([[100*(2**( lrerror_))) for lr_ in lr] for lrerror_ in lrerror]))
# exit()

# prepare dataset
LR_error  = []
for p0 in prod0:
    for pend in prodend:
        for c0 in cost0:
            for l in lr:
                for e in lrerror:
                    LR_error.append([p0, 
                                     p0*pend, 
                                     c0, 
                                     l, 
                                     e, 
                                    *compute_cost_lr(np.log2(1 - l/100), 
                                                    np.log10(c0), 
                                                    np.log10(p0), 
                                                    np.log10(p0*pend)),
                                    *compute_cost_lr(np.log2(1 - (l + e)/100),
                                                    np.log10(c0),
                                                    np.log10(p0),
                                                    np.log10(p0*pend)),
                                    *compute_cost_lr_lafond(np.log2(1 - l/100), 
                                                           np.log10(c0),
                                                           np.log10(p0),
                                                           np.log10(p0*pend)),
                                    *compute_cost_lr_lafond(np.log2(1 - (l+e)/100), 
                                                              np.log10(c0),
                                                              np.log10(p0),
                                                              np.log10(p0*pend)),
                                    # *compute_cost_breakpoint(np.log2(1 - l/100), 
                                    #                          np.log10(c0),
                                    #                          np.log10(p0),
                                    #                          np.log10(p0*pend),
                                    #                          -np.log2(-(35/100 - 1)),
                                    #                          1)
                    ])


LR_error = pd.DataFrame(LR_error, columns = ['p0', 
                                             'pend', 
                                             'c0', 
                                             'lr', 
                                             'lrerror', 
                                             'integral', 
                                             'unitCost',
                                             'integral_error',
                                             'unitCost_error',
                                             'lafond_integral',
                                             'lafond_unitCost',
                                             'lafond_integral_error',
                                             'lafond_unitCost_error',
                                            #  'breakpoint_integral',
                                            #  'breakpoint_unitCost'
                                             ])

LR_error['error'] = LR_error['lafond_unitCost_error']/LR_error['lafond_unitCost']

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(7,9))

count = 0
for p0 in prod0:
    for c0 in cost0:
        for l in lr:
            sub_LR_error = LR_error.loc[LR_error['p0'] == p0].\
                    loc[LR_error['c0'] == c0].\
                    loc[LR_error['lr'] == l]

            pivot = sub_LR_error.pivot(index='pend', columns='lrerror', values='error')

            cf = ax[count].contourf(pivot.index, pivot.columns, pivot.values.T,
                        levels=[0.1, 1/5, 1/2, 1/1.2, 1.2, 2, 5, 10],
                        norm=matplotlib.colors.LogNorm(),cmap='RdBu', extend='both')
            # cf = plt.pcolormesh(pivot.index, pivot.columns, pivot.values.T,
            #             # levels=[1e-2, 0.1, 0.9, 1/0.9, 10,100],
            #             norm=matplotlib.colors.LogNorm(vmin=1e-4, vmax=1e4),cmap='RdBu')
            ax[count].set_xscale('log', base=10)
            count += 1
fig.subplots_adjust(right=0.8, bottom=0.1, top=0.95, hspace=.3)
cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
cbar = fig.colorbar(cf, label='Unit cost error (prediction / observation)',
                    cax=cbar_ax, extend='both')
cbar.set_ticks([0.1, 1/5, 1/2, 1/1.2, 1.2, 2, 5, 10])
cbar.set_ticklabels(['0.1', '0.2', '0.5', '0.8', '1.2', '2', '5', '10'])
ax[0].set_title('Observed learning rate = 0%')
ax[1].set_title('Observed learning rate = 20%')
ax[2].set_title('Observed learning rate = 40%')
ax[-1].set_xlabel('Multiplicative increase in production')
ax[1].set_ylabel('Learning rate error (%)')

# LR_error['error'] = LR_error['lafond_unitCost_error']/LR_error['breakpoint_unitCost']

# for p0 in prod0:
#     for c0 in cost0:
#         for l in lr:
#             sub_LR_error = LR_error.loc[LR_error['p0'] == p0].\
#                     loc[LR_error['c0'] == c0].\
#                     loc[LR_error['lr'] == 100 * (1 - 2**l)]


#             plt.figure()
#             pivot = sub_LR_error.pivot(index='pend', columns='lrerror', values='error')

#             cf = plt.contourf(pivot.index, pivot.columns, pivot.values.T,
#                         levels=[1e-2, 0.1, np.log10(2), 0.5, 0.8, 1.2, 2, 1/np.log10(2), 10,100],
#                         norm=matplotlib.colors.LogNorm(),cmap='RdBu')
#             plt.xscale('log', base=10)
#             plt.colorbar(cf, label='Unit cost (prediction / observation)')
#             plt.xlabel('Multiplicative increase in production')
#             plt.ylabel('Learning rate error (%)')


plt.show()

                                                    
                    


