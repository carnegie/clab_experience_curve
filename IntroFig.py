import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import cmcrameri
import matplotlib.animation as animation

matplotlib.rc('savefig', dpi=300)


matplotlib.rc('font',
                **{'family':'sans-serif','sans-serif':'Helvetica'})

DF = pd.read_csv('ExpCurves.csv')


techs = ['Transistor','Fotovoltaica']

cmap = cmcrameri.cm.hawaii
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

# fig, ax = plt.subplots(1,2,figsize=(10,5), layout='constrained')
# count = 0
# for tech in techs:
#     df = DF.loc[DF['Tech']==tech]

#     x, y, = np.log10(df[['Cumulative production']].values), \
#             np.log10(df[['Unit cost']].values)
    
#     # for i in range(2, len(y)):
#     #     model = sm.OLS(y[:i], sm.add_constant(x[:i]))
#     #     result = model.fit()
#     #     ax[count].plot(10**x, 10**result.predict(sm.add_constant(x)), 
#     #              color = cmap(i/len(y)), alpha=0.5,
#     #              zorder=-1)

#     df.plot.scatter(x='Cumulative production', 
#                     y='Unit cost', 
#                     logx=True, logy=True,
#                     s=50, zorder=1,
#                     color='k', edgecolor='k', 
#                     ax=ax[count])

#     if tech == 'Fotovoltaica':
#         tech = 'Solar PV CAPEX [$/W]'
#     title = str(tech) + ' (' + \
#             str(int(df['Year'].values[0])) + '-' +\
#             str(int(df['Year'].values[-1])) + ')'
#     ax[count].set_title(title)
#     count += 1
# ax[0].set_ylim(6e-9,1.8e1)
# ax[1].set_ylim(0.5,2e4)
# cbar = fig.colorbar(
#     matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
#                           ticks = [0,1], )
# cbar.set_ticklabels(['1969','2005'])
# plt.subplots_adjust(wspace=0.5, right=0.9, left=0.1)
# plt.show()



# fig, ax = plt.subplots(1,2,figsize=(10,5), layout='constrained')
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
# ax[0].set_ylim(6e-9,1.8e1)
# ax[1].set_ylim(0.5,2e4)
# ax[0].set_xlim(0.5,1.5e10)
# ax[1].set_xlim(4e-5, 1.5e4)
# ax[0].set_xlabel('Cumulative production')
# ax[1].set_xlabel('Cumulative production')
# ax[0].set_ylabel('Unit cost')
# ax[1].set_ylabel('Unit cost')
# cbar = fig.colorbar(
#     matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
#                           ticks = [0,1], )
# plt.subplots_adjust(wspace=0.5, right=0.9, left=0.1)
# cbar.set_ticklabels(['first two points','full dataset'])
# count = 0
# for tech in techs:
#     df = DF.loc[DF['Tech']==tech]

#     if tech == 'Fotovoltaica':
#         tech = 'Solar PV CAPEX [$/W]'
#     title = str(tech) + ' (' + \
#             str(int(df['Year'].values[0])) + '-' +\
#             str(int(df['Year'].values[-1])) + ')'
#     ax[count].set_title(title)
#     count += 1

# def animate(yy):
#     count = 0
#     for tech in techs:
#         df = DF.loc[DF['Tech']==tech]

#         x, y, year, = np.log10(df[['Cumulative production']].values), \
#                 np.log10(df[['Unit cost']].values), \
#                 df[['Year']].values
        
#         x_cal, y_cal = x[year<=yy], y[year<=yy]

#         for ln in ax[count].get_lines():
#             ln.set_alpha(0.2)

#         if len(x_cal) >= 2:

#             model = sm.OLS(y_cal, sm.add_constant(x_cal))
#             result = model.fit()
#             ln1 = ax[count].plot(10**x_cal, 10**result.predict(sm.add_constant(x_cal)), 
#                         color = cmap(len(y_cal)/len(y)), alpha=1.0,
#                         zorder=1, lw=2)
#             x_val, y_val = x[year>=yy], y[year>=yy]
#             model = sm.OLS(y_val, sm.add_constant(x_val))
#             result = model.fit()
#             ln2 = ax[count].plot(10**x_val, 10**result.predict(sm.add_constant(x_val)), 
#                         color = cmap(len(y_cal)/len(y)), alpha=1.0,
#                         zorder=1, lw=2)


#             ax[count].scatter(10**x_cal, 10**y_cal,
#                         zorder=-1, color='k',
#                         edgecolor='k')
#             ax[count].scatter(10**x_val, 10**y_val,
#                         zorder=-1, color='silver',
#                         alpha=0.3,
#                         edgecolor='k')

#         count += 1

# ani = animation.FuncAnimation(fig, animate, repeat=False,
#                                     frames=range(1959,2010), interval=200)

# # To save the animation using Pillow as a gif
# # writer = animation.PillowWriter(fps=5,
# #                                 metadata=dict(artist='Me'))
# # ani.save('scatter.gif', writer=writer)



fig, ax = plt.subplots(1,2,figsize=(9,5), 
                       layout='constrained'
                       )
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
# ax[0].set_ylim(6e-9,1.8e1)
ax[1].set_ylim(0.5,2e4)
# ax[0].set_xlim(0.5,1.5e10)
ax[1].set_xlim(4e-5, 1.5e4)
ax[0].set_xlabel('Observed learning exponent')
ax[1].set_xlabel('Cumulative production')
ax[0].set_ylabel('Future learning exponent')
ax[1].set_ylabel('Unit cost')
cbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                          ticks = [0,1], 
                          ax = ax,
                          location='bottom',
                          shrink=0.5)
# plt.subplots_adjust(
#                     wspace=0.2, 
#                     right=0.9, left=0.15, 
#                     hspace=0.2,bottom=0.2
#                     )
cbar.set_ticklabels(['1959','2005'])
df = DF.loc[DF['Tech']=='Fotovoltaica']
title = 'Solar PV CAPEX [$/W]'
title = title + ' (' + \
            str(int(df['Year'].values[0])) + '-' +\
            str(int(df['Year'].values[-1])) + ')'
fig.suptitle(title)

ax[0].set_xlim(-0.6,0.2)
ax[0].set_ylim(-0.6,0.2)
ax[0].axhline(0, color='silver', zorder=-1, lw=.5)
ax[0].axvline(0, color='silver', zorder=-1, lw=.5)
ax[0].plot([-0.6,0.2],[-0.6,0.2], color='k', 
           zorder=-1, lw=1.5, linestyle='--')
ax[0].annotate('Future cost decline is faster',
                xy=(-0.0, -0.3), xycoords='data',
                ha='center', va='center')
ax[0].annotate('Future cost decline is slower',
                xy=(-0.25, -0.025), xycoords='data',
                ha='center', va='center')
ax[0].set_aspect('equal')

df = DF.loc[DF['Tech']=='Fotovoltaica']

x, y, year, = np.log10(df[['Cumulative production']].values), \
        np.log10(df[['Unit cost']].values), \
        df[['Year']].values

m = sm.OLS(y, sm.add_constant(x))
r = m.fit()

ax[1].plot(10**x, 10**r.predict(sm.add_constant(x)), 
            color = 'firebrick', alpha=1.0,
            zorder=1, lw=2)
ax[1].scatter(10**x, 10**y, color='k')
l = ax[0].axvline(r.params[1], color='firebrick')

plt.show()

l.remove()

for s in ax[1].collections:
    s.remove()

for l in ax[1].lines:
    l.remove()

for yy in range(1959,2007):
    for tech in ['Fotovoltaica']:
        df = DF.loc[DF['Tech']==tech]

        x, y, year, = np.log10(df[['Cumulative production']].values), \
                np.log10(df[['Unit cost']].values), \
                df[['Year']].values
        
        x_cal, y_cal = x[year<=yy], y[year<=yy]

        for ln in ax[1].get_lines():
            ln.set_alpha(0.05)
        
        if len(x_cal) >= 2:

            model = sm.OLS(y_cal, sm.add_constant(x_cal))
            result_cal = model.fit()
            ln1 = ax[1].plot(10**x_cal, 10**result_cal.predict(sm.add_constant(x_cal)), 
                        color = cmap(len(y_cal)/len(y)), alpha=1.0,
                        zorder=1, lw=2)
            x_val, y_val = x[year>=yy], y[year>=yy]
            model = sm.OLS(y_val, sm.add_constant(x_val))
            result_val = model.fit()
            ln2 = ax[1].plot(10**x_val, 10**result_val.predict(sm.add_constant(x_val)), 
                        color = cmap(len(y_cal)/len(y)), alpha=1.0,
                        zorder=1, lw=2)

            ax[1].scatter(10**x_cal, 10**y_cal,
                        zorder=-1, color='k',
                        edgecolor='k')
            ax[1].scatter(10**x_cal[-1], 10**y_cal[-1],
                        zorder=-1, color=cmap(len(y_cal)/len(y)),
                        edgecolor='r')
            ax[1].scatter(10**x_val, 10**y_val,
                        zorder=-1, color='silver',
                        alpha=0.3,
                        edgecolor='k')

            ax[0].scatter(result_cal.params[1], result_val.params[1], 
                          color=cmap(len(y_cal)/len(y))) 
for s in ax[0].collections:
    s.remove()
for l in ax[1].lines:
    l.remove()
for c in ax[1].collections:
    c.remove()

def animate(yy):
    for tech in ['Fotovoltaica']:
        df = DF.loc[DF['Tech']==tech]

        x, y, year, = np.log10(df[['Cumulative production']].values), \
                np.log10(df[['Unit cost']].values), \
                df[['Year']].values
        
        x_cal, y_cal = x[year<=yy], y[year<=yy]

        for ln in ax[1].get_lines():
            ln.set_alpha(0.05)
        
        # for p in ax[0].collections:
        #     p.set_color('k')

        if len(x_cal) >= 2:

            model = sm.OLS(y_cal, sm.add_constant(x_cal))
            result_cal = model.fit()
            ln1 = ax[1].plot(10**x_cal, 10**result_cal.predict(sm.add_constant(x_cal)), 
                        color = cmap(len(y_cal)/len(y)), alpha=1.0,
                        zorder=1, lw=2)
            x_val, y_val = x[year>=yy], y[year>=yy]
            model = sm.OLS(y_val, sm.add_constant(x_val))
            result_val = model.fit()
            ln2 = ax[1].plot(10**x_val, 10**result_val.predict(sm.add_constant(x_val)), 
                        color = cmap(len(y_cal)/len(y)), alpha=1.0,
                        zorder=1, lw=2)


            ax[1].scatter(10**x_cal, 10**y_cal,
                        zorder=-1, color='k',
                        edgecolor='k')
            ax[1].scatter(10**x_cal[-1], 10**y_cal[-1],
                        zorder=-1, color=cmap(len(y_cal)/len(y)),
                        edgecolor='r')
            ax[1].scatter(10**x_val, 10**y_val,
                        zorder=-1, color='silver',
                        alpha=0.3,
                        edgecolor='k')

            ax[0].scatter(result_cal.params[1], result_val.params[1], 
                          color=cmap(len(y_cal)/len(y))) 
            
ani = animation.FuncAnimation(fig, animate, repeat=True,
                                    frames=range(1959,2007), interval=200)

# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=5,
                                metadata=dict(artist='Me'))
ani.save('scatter_solar.gif', writer=writer)

plt.show()