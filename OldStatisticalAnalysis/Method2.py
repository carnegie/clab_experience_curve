import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from mpl_toolkits.mplot3d import axes3d 
import cmcrameri as cm
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.style.use('ggplot')
df = pd.read_csv('ExpCurves.csv')
# df = pd.read_csv('NormalizedExpCurves.csv')
# df['Cumulative production'] = df['Normalized cumulative production']
# df['Unit cost'] = df['Normalized unit cost']

method = 'regression'
# method = 'slope'

# get slope for all technologies
slopes = []
# slopes_pl = []
for tech in df['Tech'].unique():
	sel = df.loc[df['Tech']==tech]
	x = np.log10(sel['Cumulative production'].values)
	y = np.log10(sel['Unit cost'].values)
	model = sm.OLS(y, sm.add_constant(x))
	result = model.fit()
	slopes.append([tech, result.params[1]])
slopes = pd.DataFrame(slopes, columns=['Tech', 'Slope'])
# exit()

# get error using technology specific slope
# add information about the technology to make count of technologies in bin
dferr = []
dferr2 = []
counterr = 0
for tech in df['Tech'].unique():
	# computing average technological slope based on all other technologies
	slopeall = np.mean(slopes.loc[slopes['Tech'] != tech,'Slope'].values)
	# computing technology specific slope
	sel = df.loc[df['Tech']==tech]
	x = np.log10(sel['Cumulative production'].values)
	y = np.log10(sel['Unit cost'].values)
	H = len(x)
    # select N points before midpoint and compute slope
	for i in range(H):
		for N in range(i-1, -1, -1):
			slope = (y[i] - y[N]) /\
				(x[i] - x[N])
			# add linear regression method
			if method=='regression':
				model = sm.OLS(y[N:i+1], sm.add_constant(x[N:i+1]))
				result = model.fit()
				slope = result.params[1]
			# compute error associated using slope M points after midpoint
			for M in range(i+1, H):
				pred =  y[i] + slope * (x[M] - x[i])
				# if method=='regression':
				# 	pred = result.params[0] + slope * x[M]
				pred2 =  y[i] + slopeall * (x[M] - x[i])
				error = (y[M] - (pred)) 
				error2 = (y[M] - (pred2)) 
				dferr.append([x[i] - x[N],
								# (10**x[M] - 10**x[i])/(10**x[i] - 10**x[N]),
								# (10**x[M] - 10**x[i]),
								(x[M] - x[i]),
								error, tech])
				dferr2.append([x[i] - x[N],
								# (10**x[M] - 10**x[i])/(10**x[i] - 10**x[N]),
								# (10**x[M] - 10**x[i]),
								(x[M] - x[i]),
								error2, tech])
				if np.abs(error2) < np.abs(error):
					counterr += 1

print('Percentage of cases where error', 
      ' is lower with average technological slope: ',
      round(100*counterr/len(dferr),2), '%')

dferr = pd.DataFrame(dferr, 
                     columns = ['Log of ratios for predictor',
                                'Log of ratios for prediction',
                                'Error', 'Tech'])
dferr2 = pd.DataFrame(dferr2, 
                     columns = ['Log of ratios for predictor',
                                'Log of ratios for prediction',
                                'Error', 'Tech'])

# fig, ax = plt.subplots(1,2)
# dferr.plot(x='Log of ratios for predictor', y='Error',
# 		marker='o', kind='scatter', ax=ax[0], color='darkmagenta', alpha=0.5)
# dferr2.plot(x='Log of ratios for predictor', y='Error',
# 		marker = '^', kind='scatter', ax=ax[1], color='forestgreen', alpha=0.5)
# # plt.show()

# fig, ax = plt.subplots()
# im = ax.scatter(dferr['Log of ratios for predictor'],dferr['Log of ratios for prediction'],
# 	   		marker='o', c=np.abs(dferr['Error'])-np.abs(dferr2['Error']), alpha=0.1,
# 			cmap='RdBu', norm=matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1))
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label('Error difference')
# ax.set_xlabel('Log of ratios for predictor')
# ax.set_ylabel('Log of ratios for prediction')
# # plt.show()

# ptiles_or, ptiles2_or, count_or, counttech_or = [], [], [], []
# ptiles_on, ptiles2_on, count_on, counttech_on = [], [], [], []
# dxs = np.arange(0,np.log10(dferr['Log of ratios for prediction'].max()),0.2)
# dxs = [10**x for x in dxs]
# dxs = [0,*dxs]
# # dxs = np.arange(0,dferr['Log of ratios for prediction'].max(),0.2)
# dxsor_plot, dxson_plot = [], []
# for idx in range(len(dxs)-1):
# 	# if not dferr.loc[(dferr['Log of ratios for predictor']>=dxs[idx]) &\
# 	# 	    			(dferr['Log of ratios for predictor']<dxs[idx+1])].empty and \
# 	# 				dferr.loc[(dferr['Log of ratios for predictor']>=dxs[idx]) &\
# 	# 	    			(dferr['Log of ratios for predictor']<dxs[idx+1]),'Tech'].nunique()>10:
# 	# 	spt = dferr.loc[(dferr['Log of ratios for predictor']>=dxs[idx]) &\
# 	# 						(dferr['Log of ratios for predictor']<dxs[idx+1])].copy()
# 	# 	spt2 = dferr2.loc[(dferr2['Log of ratios for predictor']>=dxs[idx]) &\
# 	# 						(dferr2['Log of ratios for predictor']<dxs[idx+1])].copy()
# 	# 	for tt in spt['Tech'].unique():
# 	# 		spt.loc[spt['Tech']==tt,'Weights'] = 1/spt.loc[spt['Tech']==tt].count()[0]
# 	# 		spt2.loc[spt2['Tech']==tt,'Weights'] = 1/spt2.loc[spt2['Tech']==tt].count()[0]
# 	# 	spt = spt.sort_values(by='Error', ascending=True)
# 	# 	spt2 = spt2.sort_values(by='Error', ascending=True)
# 	# 	cumsum = spt['Weights'].cumsum()
# 	# 	cumsum2 = spt2['Weights'].cumsum()
# 	# 	pt, pt2 = [], []
# 	# 	for q in [25,50,75]:
# 	# 		cutoff = spt['Weights'].sum() * q/100
# 	# 		cutoff2 = spt2['Weights'].sum() * q/100
# 	# 		pt.append(spt['Error'][cumsum >= cutoff].iloc[0])
# 	# 		pt2.append(spt2['Error'][cumsum2 >= cutoff2].iloc[0])
# 	# 	# pt = dferr.loc[(dferr['Log of ratios for predictor']>=dxs[idx]) &\
# 	# 	# 					(dferr['Log of ratios for predictor']<dxs[idx+1]),'Error']\
# 	# 	# 					.quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]).values
# 	# 	# pt2 = dferr2.loc[(dferr2['Log of ratios for predictor']>=dxs[idx]) &\
# 	# 	# 					(dferr2['Log of ratios for predictor']<dxs[idx+1]),'Error']\
# 	# 	# 					.quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]).values
# 	# 	count_or.append(dferr2.loc[(dferr2['Log of ratios for predictor']>=dxs[idx]) &\
# 	# 						(dferr2['Log of ratios for predictor']<dxs[idx+1])].count()[0])
# 	# 	counttech_or.append(dferr2.loc[(dferr2['Log of ratios for predictor']>=dxs[idx]) &\
# 	# 						(dferr2['Log of ratios for predictor']<dxs[idx+1]),'Tech'].nunique())
# 	# 	ptiles_or.append(pt)
# 	# 	ptiles2_or.append(pt2)
# 	# 	dxsor_plot.append(dxs[idx])
# 	if not dferr.loc[(dferr['Log of ratios for prediction']>=dxs[idx]) &\
# 		    			(dferr['Log of ratios for prediction']<dxs[idx+1])].empty and \
# 					dferr.loc[(dferr['Log of ratios for prediction']>=dxs[idx]) &\
# 		    			(dferr['Log of ratios for prediction']<dxs[idx+1]),'Tech'].nunique()>=10:
# 		spt = dferr.loc[(dferr['Log of ratios for prediction']>=dxs[idx]) &\
# 							(dferr['Log of ratios for prediction']<dxs[idx+1])].copy()
# 		spt2 = dferr2.loc[(dferr2['Log of ratios for prediction']>=dxs[idx]) &\
# 							(dferr2['Log of ratios for prediction']<dxs[idx+1])].copy()
# 		# spt = spt.loc[(spt['Log of ratios for predictor']>=1) &\
# 		# 					(spt['Log of ratios for predictor']<2)].copy()
# 		# spt2 = spt2.loc[(spt2['Log of ratios for predictor']>=1) &\
# 		# 					(spt2['Log of ratios for predictor']<2)].copy()
# 		for tt in spt['Tech'].unique():
# 			spt.loc[spt['Tech']==tt,'Weights'] = 1/spt.loc[spt['Tech']==tt].count()[0]
# 			spt2.loc[spt2['Tech']==tt,'Weights'] = 1/spt2.loc[spt2['Tech']==tt].count()[0]
# 		spt = spt.sort_values(by='Error', ascending=True)
# 		spt2 = spt2.sort_values(by='Error', ascending=True)
# 		cumsum = spt['Weights'].cumsum().round(4)
# 		cumsum2 = spt2['Weights'].cumsum().round(4)
# 		pt, pt2 = [], []
# 		for q in [0,10,25,50,75,90,100]:
# 			cutoff = spt['Weights'].sum() * min(1.0,q/100)
# 			cutoff2 = spt2['Weights'].sum() * min(1.0,q/100)
# 			pt.append(spt['Error'][cumsum >= cutoff.round(4)].iloc[0])
# 			pt2.append(spt2['Error'][cumsum2 >= cutoff2.round(4)].iloc[0])
# 		# pt = dferr.loc[(dferr['Log of ratios for prediction']>=dxs[idx]) &\
# 		# 					(dferr['Log of ratios for prediction']<dxs[idx+1]),'Error']\
# 		# 					.quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).values
# 		# pt2 = dferr2.loc[(dferr2['Log of ratios for prediction']>=dxs[idx]) &\
# 		# 					(dferr2['Log of ratios for prediction']<dxs[idx+1]),'Error']\
# 		# 					.quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).values
# 		count_on.append(dferr2.loc[(dferr2['Log of ratios for prediction']>=dxs[idx]) &\
# 							(dferr2['Log of ratios for prediction']<dxs[idx+1])].count()[0])
# 		counttech_on.append(dferr2.loc[(dferr2['Log of ratios for prediction']>=dxs[idx]) &\
# 							(dferr2['Log of ratios for prediction']<dxs[idx+1]),'Tech'].nunique())
# 		ptiles_on.append(pt)
# 		ptiles2_on.append(pt2)
# 		dxson_plot.append(dxs[idx])
	

# # ptiles_or = np.array(ptiles_or)
# # ptiles2_or = np.array(ptiles2_or)
# ptiles_on = np.array(ptiles_on)
# ptiles2_on = np.array(ptiles2_on)

# fig, ax = plt.subplots(3,1,figsize=(10,8), sharex=True)
# # ax[0][0].plot(dxsor_plot, ptiles_or[:,1], color='darkmagenta')
# # ax[0][0].plot(dxsor_plot, ptiles2_or[:,1], color='forestgreen')
# # # for idx in range(3,-1,-1):
# # for idx in range(1,-1,-1):
# # 	ax[0][0].fill_between(dxsor_plot, ptiles_or[:,idx], 
# # 		    ptiles_or[:,idx+1], alpha=0.05+idx*0.2,
# # 			  color='darkmagenta', lw=0)
# # 	ax[0][0].fill_between(dxsor_plot, ptiles_or[:,-idx-1], 
# # 		    ptiles_or[:,-(idx+1)-1], alpha=0.05+idx*0.2,
# # 			  color='darkmagenta', lw=0)
# # 	ax[0][0].fill_between(dxsor_plot, ptiles2_or[:,idx], 
# # 		    ptiles2_or[:,idx+1], alpha=0.05+idx*0.2,
# # 			  color='forestgreen', lw=0)
# # 	ax[0][0].fill_between(dxsor_plot, ptiles2_or[:,-idx-1], 
# # 		    ptiles2_or[:,-(idx+1)-1], alpha=0.05+idx*0.2,
# # 			  color='forestgreen', lw=0)
# # ax[1][0].plot(dxsor_plot, count_or, color='k', zorder=-1)
# # ax102 = ax[1][0].twinx()
# # ax102.plot(dxsor_plot, counttech_or, color='r', zorder=-1)
# # ax102.set_yscale('log', base=10)
# # ax102.spines['right'].set_color('red')
# # ax102.tick_params(axis='y', colors='red')
# # ax102.set_ylim((9,101))
# # # ax102.set_ylabel('Number of technologies', color='red')
# # ax102.minorticks_off()
# # ax[1][0].set_yscale('log', base=10)
# # ax[1][0].set_ylabel('Number of data points')
# # ax[1][0].set_xlabel('Orders of magnitude used for predictor')
# # ax[0][0].set_ylabel('Error in log10 space')

# ax[0].plot(dxson_plot, ptiles_on[:,3], color='darkmagenta')
# ax[1].plot(dxson_plot, ptiles2_on[:,3], color='forestgreen')
# # for idx in range(3,-1,-1):
# for idx in range(3,-1,-1):
# 	ax[0].fill_between(dxson_plot, ptiles_on[:,idx], 
# 		    ptiles_on[:,idx+1], alpha=0.05+idx*0.2,
# 			  color='darkmagenta', lw=0)
# 	ax[0].fill_between(dxson_plot, ptiles_on[:,-idx-1], 
# 		    ptiles_on[:,-(idx+1)-1], alpha=0.05+idx*0.2,
# 			  color='darkmagenta', lw=0)
# 	ax[1].fill_between(dxson_plot, ptiles2_on[:,idx], 
# 		    ptiles2_on[:,idx+1], alpha=0.05+idx*0.2,
# 			  color='forestgreen', lw=0)
# 	ax[1].fill_between(dxson_plot, ptiles2_on[:,-idx-1], 
# 		    ptiles2_on[:,-(idx+1)-1], alpha=0.05+idx*0.2,
# 			  color='forestgreen', lw=0)
# ax[2].plot(dxson_plot, counttech_on, color='k')
# ax112 = ax[2].twinx()
# ax112.plot(dxson_plot, count_on, color='r')
# # ax112.set_yscale('log', base=10)
# ax112.spines['right'].set_color('red')
# ax112.tick_params(axis='y', colors='red')
# ax112.set_ylabel('Number of points', color='red')
# # ax112.set_ylim((9,101))
# ax112.minorticks_off()
# # ax[1].set_yscale('log', base=10)
# ax[2].set_ylabel('Number of technologies')
# ax[2].set_xlabel('Prediction interval / Predictor interval')
# ax[2].set_xscale('symlog')
# # ax[2].set_xticks([0,np.log10(3),np.log10(10),np.log10(30),np.log10(100),np.log10(300),np.log10(1000)],
# # 		      ['0','3x','10x','30x','100x','300x','1000x'])
# # orders = [10**(x) for x in range(0,6)]
# # ax[2].set_xticks([np.log10(x) for x in orders],
# # 		      [str(x)+'x' for x in orders])
# ax[0].set_ylabel('Actual/Predicted')
# ax[0].set_yticks([-np.log10(10),-np.log10(3),0,np.log10(3),np.log10(10),],
# 		      ['0.1','0.7','0','3','10'])
# # ax[0].set_yticks([-np.log10(1e4),-np.log10(1000),-np.log10(100),-np.log10(10),-np.log10(3),0,np.log10(3),np.log10(10),np.log10(100),np.log10(1000)],
# # 		      ['0.0001','0.001','0.01','0.1','0.7','0','3','10','100','1000'])
# ax[1].set_ylabel('Actual/Predicted')
# ax[1].set_yticks([-np.log10(10),-np.log10(3),0,np.log10(3),np.log10(10),],
# 		      ['0.1','0.7','0','3','10'])
# # ax[1].set_yticks([-np.log10(1e4),-np.log10(1000),-np.log10(100),-np.log10(10),-np.log10(3),0,np.log10(3),np.log10(10),np.log10(100),np.log10(1000)],
# # 		      ['0.0001','0.001','0.01','0.1','0.7','0','3','10','100','1000'])

# ax[0].set_ylim(-1,1)
# ax[0].grid(axis='y')
# ax[1].set_ylim(-1,1)
# ax[1].grid(axis='y')
# ax[2].set_ylim(0,90)
# legend_elements = [matplotlib.lines.Line2D([0],[0], color='darkmagenta', lw=1, label='Technology-specific slope (median)'),
# 		   matplotlib.lines.Line2D([0],[0], color='forestgreen', lw=1, label='Average technological slope (median)'),
# 		   matplotlib.patches.Patch(facecolor='darkmagenta', edgecolor='k', alpha=0.05, label='Min to max percentile'),
# 		   matplotlib.patches.Patch(facecolor='forestgreen', edgecolor='k', alpha=0.05, label='Min to max percentile'),
# 		   matplotlib.patches.Patch(facecolor='darkmagenta', edgecolor='k', alpha=0.25, label='10th to 90th percentile'),
# 		   matplotlib.patches.Patch(facecolor='forestgreen', edgecolor='k', alpha=0.25, label='10th to 90th percentile'),
# 		   matplotlib.patches.Patch(facecolor='darkmagenta', edgecolor='k', alpha=0.65, label='25th to 75th percentile'),
# 		   matplotlib.patches.Patch(facecolor='forestgreen', edgecolor='k', alpha=0.65, label='25th to 75th percentile'),
# 		#    matplotlib.patches.Patch(facecolor='darkmagenta', edgecolor='k', alpha=0.65, label='40th to 60th percentile'),
# 		#    matplotlib.patches.Patch(facecolor='forestgreen', edgecolor='k', alpha=0.65, label='40th to 60th percentile'),
# 		   matplotlib.lines.Line2D([0],[0], color='k', lw=1, label='Number of technologies'),
# 		   matplotlib.lines.Line2D([0],[0], color='r', lw=1, label='Number of points')]
# order = [0,2,4,6,8,10,1,3,5,7,9]
# order = [0,2,4,6,8,1,3,5,7,9]
# legend_elements = [legend_elements[x] for x in order]
# fig.legend(handles=legend_elements, ncol=2, loc='lower center')
# fig.subplots_adjust(bottom=0.2, right=0.9, left=0.1, top=0.9)
# # ax[0][0].set_title('Increasing predictor calibration interval')
# ax[0].set_title('Prediction error')
# # plt.show()
# # ax[0].annotate('Underestimating', xy=(np.log10(2),0.5), xycoords='data', fontsize=16,
# # 				horizontalalignment='center', verticalalignment='top')
# # ax[0].annotate('Overestimating', xy=(np.log10(2), -0.5), xycoords='data', fontsize=16,
# # 				horizontalalignment='center', verticalalignment='top')
# plt.show()

# ## plot with equal number of points

# npoints = 1000
# hnpoints = int(npoints/2)
# ptiles_or, ptiles2_or, counttech_or = [], [], []
# ptiles_on, ptiles2_on, counttech_on = [], [], []
# dferr_or = dferr.sort_values(by='Log of ratios for predictor')
# dferr2_or = dferr2.sort_values(by='Log of ratios for predictor')
# dferr_on = dferr.sort_values(by='Log of ratios for prediction')
# dferr2_on = dferr2.sort_values(by='Log of ratios for prediction')
# for point in range(hnpoints, dferr_or['Log of ratios for predictor'].count()-hnpoints, hnpoints):
# 	subset = dferr_or.iloc[point-hnpoints:point+hnpoints]
# 	subset2 = dferr2_or.iloc[point-hnpoints:point+hnpoints]
# 	pt = subset['Error'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).values
# 	pt2 = subset2['Error'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).values
# 	ptiles_or.append(pt)
# 	ptiles2_or.append(pt2)
# 	subset = dferr_on.iloc[point-hnpoints:point+hnpoints]
# 	subset2 = dferr2_on.iloc[point-hnpoints:point+hnpoints]
# 	pt = subset['Error'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).values
# 	pt2 = subset2['Error'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).values
# 	ptiles_on.append(pt)
# 	ptiles2_on.append(pt2)

# xaxor = dferr_or.iloc[hnpoints:-hnpoints:hnpoints]['Log of ratios for predictor'].values
# xaxon = dferr_on.iloc[hnpoints:-hnpoints:hnpoints]['Log of ratios for prediction'].values
# ptiles_or = np.array(ptiles_or)
# ptiles2_or = np.array(ptiles2_or)
# ptiles_on = np.array(ptiles_on)
# ptiles2_on = np.array(ptiles2_on)

# fig, ax = plt.subplots(2,2, sharey='row', figsize=(10,6), sharex=True)
# ax[0][0].plot(xaxor, ptiles_or[:,4], color='darkmagenta')
# ax[0][0].plot(xaxor, ptiles2_or[:,4], color='forestgreen')
# for idx in range(3,-1,-1):
# 	ax[0][0].fill_between(xaxor, ptiles_or[:,idx], 
# 		    ptiles_or[:,idx+1], alpha=0.05+idx*0.2,
# 			  color='darkmagenta', lw=0)
# 	ax[0][0].fill_between(xaxor, ptiles_or[:,-idx-1], 
# 		    ptiles_or[:,-(idx+1)-1], alpha=0.05+idx*0.2,
# 			  color='darkmagenta', lw=0)
# 	ax[0][0].fill_between(xaxor, ptiles2_or[:,idx], 
# 		    ptiles2_or[:,idx+1], alpha=0.05+idx*0.2,
# 			  color='forestgreen', lw=0)
# 	ax[0][0].fill_between(xaxor, ptiles2_or[:,-idx-1], 
# 		    ptiles2_or[:,-(idx+1)-1], alpha=0.05+idx*0.2,
# 			  color='forestgreen', lw=0)
# ax[1][0].set_xlabel('Orders of magnitude used for predictor')
# ax[0][0].set_ylabel('Error')

# ax[0][1].plot(xaxon, ptiles_on[:,4], color='darkmagenta')
# ax[0][1].plot(xaxon, ptiles2_on[:,4], color='forestgreen')
# for idx in range(3,-1,-1):
# 	ax[0][1].fill_between(xaxon, ptiles_on[:,idx], 
# 		    ptiles_on[:,idx+1], alpha=0.05+idx*0.2,
# 			  color='darkmagenta', lw=0)
# 	ax[0][1].fill_between(xaxon, ptiles_on[:,-idx-1], 
# 		    ptiles_on[:,-(idx+1)-1], alpha=0.05+idx*0.2,
# 			  color='darkmagenta', lw=0)
# 	ax[0][1].fill_between(xaxon, ptiles2_on[:,idx], 
# 		    ptiles2_on[:,idx+1], alpha=0.05+idx*0.2,
# 			  color='forestgreen', lw=0)
# 	ax[0][1].fill_between(xaxon, ptiles2_on[:,-idx-1], 
# 		    ptiles2_on[:,-(idx+1)-1], alpha=0.05+idx*0.2,
# 			  color='forestgreen', lw=0)
# ax[1][1].set_xlabel('Orders of magnitude used in prediction')
# # ax[0][1].set_ylabel('Error')

# ax[1][0].plot(xaxor, 3*np.zeros(len(xaxor)), color='k', marker='|',lw=0, alpha=0.5)
# ax[1][1].plot(xaxon, 3*np.zeros(len(xaxon)), color='k', marker='|',lw=0, alpha=0.5)

# legend_elements = [matplotlib.lines.Line2D([0],[0], color='darkmagenta', lw=1, label='Technology-specific slope (median)'),
# 		   matplotlib.lines.Line2D([0],[0], color='forestgreen', lw=1, label='Average technological slope (median)'),
# 		   matplotlib.patches.Patch(facecolor='darkmagenta', edgecolor='k', alpha=0.05, label='10th to 90th percentile'),
# 		   matplotlib.patches.Patch(facecolor='forestgreen', edgecolor='k', alpha=0.05, label='10th to 90th percentile'),
# 		   matplotlib.patches.Patch(facecolor='darkmagenta', edgecolor='k', alpha=0.25, label='20th to 80th percentile'),
# 		   matplotlib.patches.Patch(facecolor='forestgreen', edgecolor='k', alpha=0.25, label='20th to 80th percentile'),
# 		   matplotlib.patches.Patch(facecolor='darkmagenta', edgecolor='k', alpha=0.45, label='30th to 70th percentile'),
# 		   matplotlib.patches.Patch(facecolor='forestgreen', edgecolor='k', alpha=0.45, label='30th to 70th percentile'),
# 		   matplotlib.patches.Patch(facecolor='darkmagenta', edgecolor='k', alpha=0.65, label='40th to 60th percentile'),
# 		   matplotlib.patches.Patch(facecolor='forestgreen', edgecolor='k', alpha=0.65, label='40th to 60th percentile'),
# 		   matplotlib.lines.Line2D([0],[0], color='k', lw=0, marker='|', label='Central data points used to estimate statistics ('+str(hnpoints)+' points on each side, overlap < 50%)')]
# order = [0,2,4,6,8,10,1,3,5,7,9]
# legend_elements = [legend_elements[x] for x in order]
# fig.legend(handles=legend_elements, ncol=2, loc='lower center')
# fig.subplots_adjust(bottom=0.325, right=0.95, left=0.1, top=0.9)
# ax[0][0].set_title('Increasing predictor calibration interval')
# ax[0][1].set_title('Increasing prediction interval')
# ax[1][0].set_yticks([])
# ax[1][0].yaxis.grid(False)
# ax[1][1].set_yticks([])
# ax[1][1].yaxis.grid(False)

# select ratios to be plotted
frac = []
for x in range(17):
	frac.append(0.001 * (10**(0.25*x)) )
# for x in range(41):
# 	frac.append(0.001 * (10**(0.1*x)) )
# frac = np.log10([1,1.2,1.5,2,5,10,20,50,100,1e3,1e10])

mean1 = np.empty((len(frac)-1,len(frac)-1))
mean2 = np.empty((len(frac)-1,len(frac)-1))
std1 = np.empty((len(frac)-1,len(frac)-1))
std2 = np.empty((len(frac)-1,len(frac)-1))
# median1 = np.empty((len(frac)-1,len(frac)-1))
# median2 = np.empty((len(frac)-1,len(frac)-1))
meandiff = np.empty((len(frac)-1,len(frac)-1))
fracavg = np.empty((len(frac)-1,len(frac)-1))
count = np.empty((len(frac)-1,len(frac)-1))
counttech = np.empty((len(frac)-1,len(frac)-1))

for i in range(1, len(frac)):
	for j in range(1, len(frac)):
		select1 = dferr.loc[
					(dferr['Log of ratios for predictor']>frac[i-1]) &\
					(dferr['Log of ratios for predictor']<=frac[i]) & \
					(dferr['Log of ratios for prediction']>frac[j-1]) &\
					(dferr['Log of ratios for prediction']<=frac[j])
					].copy()
		select2 = dferr2.loc[
					(dferr2['Log of ratios for predictor']>frac[i-1]) &\
					(dferr2['Log of ratios for predictor']<=frac[i]) & \
					(dferr2['Log of ratios for prediction']>frac[j-1]) &\
					(dferr2['Log of ratios for prediction']<=frac[j])
					].copy()
		# weight each technology equally in each bin
		if select1.empty and select2.empty:
			mean1[i-1, j-1] = np.nan
			mean2[i-1, j-1] = np.nan
			std1[i-1, j-1] = np.nan
			std2[i-1, j-1] = np.nan
			meandiff[i-1,j-1] = np.nan
			fracavg[i-1,j-1] = np.nan
			count[i-1,j-1] = 0.0
			counttech[i-1,j-1] = 0.0
		else:
			# mean1[i-1,j-1] = np.mean(select1['Error'].values**2)**0.5
			# mean2[i-1,j-1] = np.mean(select2['Error'].values**2)**0.5
			# std1[i-1,j-1] = np.mean((select1['Error'].values - mean1[i-1,j-1])**2)**0.5
			# std2[i-1,j-1] = np.mean((select2['Error'].values - mean2[i-1,j-1])**2)**0.5
			## weighting each technology the same in each bin
			mean1[i-1,j-1] = 0.0
			mean2[i-1,j-1] = 0.0
			std1[i-1,j-1] = 0.0
			std2[i-1,j-1] = 0.0
			ntechs = select1['Tech'].nunique()
			for tech in select1['Tech'].unique():
				sel1 = select1.loc[select1['Tech']==tech].copy()
				sel2 = select2.loc[select2['Tech']==tech].copy()
				# mean1[i-1,j-1] += 1/(ntechs * sel1.count()[0]) * np.sum(sel1['Error'].values**2)
				# mean2[i-1,j-1] += 1/(ntechs * sel2.count()[0]) * np.sum(sel2['Error'].values**2)
				# std1[i-1,j-1] += 1/(ntechs) * (1/sel1.count()[0] * \
				# 	np.sum(
				# 		(sel1['Error'].values - \
				# 		(1/sel1.count()[0]) * np.sum(sel1['Error'].values**2)**0.5
				# 		)**2))
					# (np.percentile(sel1['Error'].values - \
					# ((1/sel1.count()[0]) * np.sum(sel1['Error'].values**2))**0.5, 90))
				# std2[i-1,j-1] += 1/(ntechs) * (1/sel2.count()[0] * \
				# 	np.sum(
				# 		(sel2['Error'].values - \
				# 		(1/sel2.count()[0]) * np.sum(sel2['Error'].values**2)**0.5
				# 		)**2))
					# (np.percentile(sel2['Error'].values - \
					# ((1/sel2.count()[0]) * np.sum(sel2['Error'].values**2))**0.5,90))
				m1 = 1/(sel1.count()[0]) * np.sum(sel1['Error'].values**2)
				mean1[i-1,j-1] += 1/(ntechs) * m1
				m2 = 1/(sel2.count()[0]) * np.sum(sel2['Error'].values**2)
				mean2[i-1,j-1] += 1/(ntechs) * m2
				s1 = 1/(sel1.count()[0]) * np.sum((sel1['Error'].values - m1**0.5)**2)
				std1[i-1, j-1] += 1/(ntechs) * s1 
				s2 = 1/(sel2.count()[0]) * np.sum((sel2['Error'].values - m2**0.5)**2)
				std2[i-1, j-1] += 1/(ntechs) * s2 
				fracavg[i-1,j-1] += 1/ntechs * np.sum(sel2['Error'].values**2 < 
									sel1['Error'].values**2)/\
									sel2['Error'].count() * 100
			mean1[i-1,j-1] = mean1[i-1,j-1]**0.5
			mean2[i-1,j-1] = mean2[i-1,j-1]**0.5
			std1[i-1,j-1] = std1[i-1,j-1]**0.5
			std2[i-1,j-1] = std2[i-1,j-1]**0.5
			meandiff[i-1,j-1] = mean2[i-1,j-1] - mean1[i-1,j-1]

			# fracavg[i-1,j-1] = np.sum(select2['Error'].values**2 < 
			# 						select1['Error'].values**2)/\
			# 						select2['Error'].count() * 100
			count[i-1,j-1] = (select1['Error'].count())
			counttech[i-1,j-1] = (select1['Tech'].nunique())

plt.figure()
mean = mean1[::-1,:]
im = plt.imshow(mean, aspect='auto', norm='log')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('RMSE')
maxmean1 = np.nanmax(mean1)
minmean1 = np.nanmin(mean1)
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)
plt.suptitle('Technology-specific slope')

plt.figure()
mean = mean2[::-1,:]
im = plt.imshow(mean, aspect='auto', norm=matplotlib.colors.LogNorm(vmin=minmean1, vmax=maxmean1))
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('RMSE')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)
plt.suptitle('Average technology slope')

fig, ax = plt.subplots(4,1, sharex=True, sharey=True)
cmapp = matplotlib.cm.get_cmap('Purples')
cmapg = matplotlib.cm.get_cmap('Greens')
linestyles = ['-','--','-.',':']
for i in range(7,11):
	ax[i-7].plot(10**np.array(frac[7:11]),mean1[i,7:11], color=cmapp(0.7))
	ax[i-7].plot(10**np.array(frac[7:11]),mean2[i,7:11], color=cmapg(0.7))
	ax[i-7].set_title('Predictor range (ratio of last to first point) '+str(round(10**frac[i],2))+' to '+str(round(10**frac[i+1],2)))
ax[3].set_xlabel('Ratio of future cumulative production to current cumulative production')
ax[2].annotate('RMSE (log of actual to predicted ratio)', 
	       xy=(0.025, 0.5), ha='center', va='center', 
		   xycoords='figure fraction', rotation=90)
fig.legend(['Technology-specific slope','Average technological slope'], loc='lower center', ncol=2)
plt.subplots_adjust(bottom=0.125, left=0.1, right=0.9, top=0.9, hspace=0.5)
plt.show()

plt.figure()
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1)
for i in range(mean1.shape[0]):
	for j in range(mean1.shape[1]-1,-1,-1):
		if count[i,j] > 0:
			plt.scatter(i,j,
		      color=matplotlib.cm.viridis(norm(mean1[i,j])),
			  s=2+counttech[i,j]*2, alpha=0.5, lw=0.5, edgecolor='k')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
plt.colorbar(matplotlib.cm.ScalarMappable(cmap='viridis', norm=norm), 
	     label='RMSE')
legend_elements = [matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=2**0.5, color='k', label='1 data point'),
		   matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=(2+np.max(counttech*2))**0.5, color='k', label=str(int(np.max(counttech)))+' technologies')]
plt.gcf().legend(handles=legend_elements, ncol=2, title='Number of technologies', loc='lower center')
plt.subplots_adjust(bottom=0.4, left=0.2, right=0.95, top=0.9)
plt.title('Technology-specific')

plt.figure()
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1)
for i in range(mean2.shape[0]):
	for j in range(mean2.shape[1]-1,-1,-1):
		if count[i,j] > 0:
			plt.scatter(i,j,
		      color=matplotlib.cm.viridis(norm(mean2[i,j])),
			  s=2+counttech[i,j]*2, alpha=0.5, lw=0.5, edgecolor='k')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
plt.colorbar(matplotlib.cm.ScalarMappable(cmap='viridis', norm=norm), 
	     label='RMSE')
legend_elements = [matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=2**0.5, color='k', label='1 technology'),
		   matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=(2+np.max(counttech)*2)**0.5, color='k', label=str(int(np.max(counttech)))+' technologies')]
plt.gcf().legend(handles=legend_elements, ncol=2, title='Number of technologies', loc='lower center')
plt.subplots_adjust(bottom=0.4, left=0.2, right=0.95, top=0.9)
plt.title('Average technology')

plt.figure()
print(np.nanmin(std1), np.nanmax(std1))
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1)
for i in range(std1.shape[0]):
	for j in range(std1.shape[1]-1,-1,-1):
		if count[i,j] > 0:
			plt.scatter(i,j,
		      color=matplotlib.cm.viridis(norm(std1[i,j])),
			  s=2+counttech[i,j]*2, alpha=0.5, lw=0.5, edgecolor='k')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
plt.colorbar(matplotlib.cm.ScalarMappable(cmap='viridis', norm=norm), 
	     label='Std of RMSE')
legend_elements = [matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=2**0.5, color='k', label='1 data point'),
		   matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=(2+np.max(counttech*2))**0.5, color='k', label=str(int(np.max(counttech)))+' technologies')]
plt.gcf().legend(handles=legend_elements, ncol=2, title='Number of technologies', loc='lower center')
plt.subplots_adjust(bottom=0.4, left=0.2, right=0.95, top=0.9)
plt.title('Technology-specific')

plt.figure()
print(np.nanmin(std1), np.nanmax(std1))
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1)
for i in range(std1.shape[0]):
	for j in range(std1.shape[1]-1,-1,-1):
		if count[i,j] > 0:
			plt.scatter(i,j,
		      color=matplotlib.cm.viridis(norm(std2[i,j])),
			  s=2+counttech[i,j]*2, alpha=0.5, lw=0.5, edgecolor='k')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
plt.colorbar(matplotlib.cm.ScalarMappable(cmap='viridis', norm=norm), 
	     label='Std of RMSE')
legend_elements = [matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=2**0.5, color='k', label='1 technology'),
		   matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=(2+np.max(counttech)*2)**0.5, color='k', label=str(int(np.max(counttech)))+' technologies')]
plt.gcf().legend(handles=legend_elements, ncol=2, title='Number of technologies', loc='lower center')
plt.subplots_adjust(bottom=0.4, left=0.2, right=0.95, top=0.9)
plt.title('Average technology')

plt.figure()
mean = meandiff[::-1,:]
divnorm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
im = plt.imshow(mean, aspect='auto', 
		norm=divnorm, 
		cmap='RdBu_r')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('RMSE difference')
# cbar.set_ticks([-300,-200,-100,0,1])
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)
plt.suptitle('Average technology - Technology-specific')


plt.figure()
for i in range(mean1.shape[0]):
	for j in range(mean1.shape[1]-1,-1,-1):
		if count[i,j] > 0:
			plt.scatter(i,j,
		      color=matplotlib.cm.RdBu_r(divnorm(meandiff[i,j].flatten())),
			  s=2+counttech[i,j]*2, alpha=0.5, lw=0.5, edgecolor='k')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
plt.colorbar(matplotlib.cm.ScalarMappable(cmap='RdBu_r', norm=divnorm), 
	     label='RMSE difference')
legend_elements = [matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=2**0.5, color='k', label='1 data point'),
		   matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=(2+np.max(counttech)*2)**0.5, color='k', label=str(int(np.max(counttech)))+' technologies')]
plt.gcf().legend(handles=legend_elements, ncol=2, title='Number of data points', loc='lower center')
plt.subplots_adjust(bottom=0.4, left=0.2, right=0.95, top=0.9)
plt.title('Average technology - Technology-specific')

sel = df.loc[df['Tech']=='Photovoltaics']
x, y = np.log10(sel['Cumulative production'].values), np.log10(sel['Unit cost'].values)
H = len(x)
for i in range(H):
	for N in range(i-1, -1, -1):
		slope = (y[i] - y[N]) /\
			(x[i] - x[N])
		# add linear regression method
		if method=='regression':
			model = sm.OLS(y[N:i+1], sm.add_constant(x[N:i+1]))
			result = model.fit()
			slope = result.params[1]
		# compute error associated using slope M points after midpoint
		plt.figure(figsize=(8,7))
		plt.scatter(10**x, 10**y, edgecolor='r', facecolor='none', zorder=10)
		plt.scatter(10**x[N:i+1], 10**y[N:i+1], color='r', zorder=10)
		for M in range(i+1, H):
			pred =  y[i] + slope * (x[M] - x[i])
			# if method=='regression':
			# 	pred = result.params[0] + slope * x[M]
			pred2 =  y[i] + slopeall * (x[M] - x[i])
			plt.scatter(10**x[M], 10**pred, color='b', zorder=1)
			plt.scatter(10**x[M], 10**pred2, color='g', zorder=1)
			plt.xscale('log', base=10)
			plt.yscale('log', base=10)
			predordiff = x[i] - x[N]
			predondiff = x[M] - x[i]
			idxor = np.where(frac >= predordiff)[0][0]-1
			idxon = np.where(frac >= predondiff)[0][0]-1
			plt.plot([10**x[M],10**x[M]],
	    		[10**(pred-mean1[idxor][idxon]),10**(pred+mean1[idxor][idxon])], ':',
	    		marker='_', color='b', 
				alpha = min(1.0, 0.1 + 0.9*counttech[idxor][idxon]/np.max(counttech)))
			plt.plot([10**x[M],10**x[M]],
	    		[10**(pred2-mean2[idxor][idxon]),10**(pred2+mean2[idxor][idxon])], '--',
	    		marker='_', color='g', 
				alpha = min(1.0, 0.1 + 0.9*counttech[idxor][idxon]/np.max(counttech)))
		plt.title(sel['Tech'].values[0])
		plt.xlabel('Cumulative production')
		plt.ylabel('Unit cost')
		legend_elements = [
			matplotlib.lines.Line2D([0],[0], lw=0, label='$\\bf{Data\ points}$'),
			matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=5, color='none', markeredgecolor='r', label='Observations'),
			matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=5, color='r', label='Observations used to build technology-specific predictive model'),
			matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=5, color='b', label='Prediction - Technology-specific'),
			matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=5, color='g', label='Prediction - Average technological slope'),
			matplotlib.lines.Line2D([0],[0], lw=0, label='$\\bf{Uncertainty\ around\ predictions}$'),
			matplotlib.lines.Line2D([0],[0], lw=1, ls=':', markersize=5, color='b', label='+/- 1 RMSE (Technology-specific)'),
			matplotlib.lines.Line2D([0],[0], lw=1, ls='--', markersize=5, color='g', label='+/- 1 RMSE (Average technological slope)'),
			matplotlib.lines.Line2D([0],[0], lw=0, label='$\\bf{Number\ of\ technologies\ used\ to\ estimate\ uncertainty\ (both\ methods)}$'),
			matplotlib.lines.Line2D([0],[0], lw=1, alpha=1, color='k', label=str(int(np.max(counttech)))+' technologies'),
			matplotlib.lines.Line2D([0],[0], lw=1, alpha=0.1, color='k', label='1 technology')
		]
		plt.gcf().legend(handles=legend_elements, loc='lower center')
		plt.subplots_adjust(bottom=0.425, right=0.97, top=0.95, left=0.08)
		plt.pause(.01)
		input()
		plt.close()

# plt.figure()
# for i in range(mean1.shape[0]):
# 	for j in range(mean1.shape[1]-1,-1,-1):
# 		if counttech[i,j] > 0:
# 			plt.scatter(i,j,
# 		      color=matplotlib.cm.RdBu_r(divnorm(meandiff[i,j].flatten())),
# 			  s=2+counttech[i,j]*2, alpha=0.5, lw=0.5, edgecolor='k')
# plt.gca().set_xticks([x for x in range(len(frac)-1)], 
#         [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
#         rotation = 90)
# plt.gca().set_yticks([x for x in range(len(frac)-1)], 
#         [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])])
# plt.ylabel('Log of cumulative production ratios for prediction')
# plt.xlabel('Log of cumulative production ratios for predictor')
# plt.colorbar(matplotlib.cm.ScalarMappable(cmap='RdBu_r', norm=divnorm), 
# 	     label='RMSE difference')
# legend_elements = [matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=2**0.5, color='k', label='1 technology'),
# 		   matplotlib.lines.Line2D([0],[0], lw=0, marker='o', markersize=(2+np.max(counttech)*2)**0.5, color='k', label=str(int(np.max(counttech)))+' technologies')]
# plt.gcf().legend(handles=legend_elements, ncol=2, title='Number of technologies', loc='lower center')
# plt.subplots_adjust(bottom=0.4, left=0.2, right=0.95, top=0.9)
# plt.title('Average technology - Technology-specific')


plt.figure()
perc = fracavg[::-1,:]
divnorm = matplotlib.colors.TwoSlopeNorm(vcenter=50)
im = plt.imshow(perc, aspect='auto', norm=divnorm, cmap='RdBu')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('Cases where average technological slope has lower squared errors')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)
# plt.suptitle('Average technology - Technology-specific')

count = count[::-1,:]

plt.figure()
im = plt.imshow(np.log10(count), aspect='auto')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('Bin size')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)

counttech = counttech[::-1,:]

plt.figure()
im = plt.imshow(counttech, aspect='auto')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('Bin size')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)

counttech = counttech[::-1,:]

prob = 100*(perc > 50) * count/np.sum(count)
print(sum(sum(prob)))
prob2 = -100*(perc < 50) * count/np.sum(count)
print(sum(sum(prob2)))
prob[prob==0] = prob2[prob==0]
prob[prob==0] = np.nan

plt.figure()
divnorm = matplotlib.colors.TwoSlopeNorm(vcenter=0)
im = plt.imshow(prob, aspect='auto', norm=divnorm, cmap='RdBu')
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('Porbability improvement of more accuracy with average technological slope')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)


plt.figure()
for i in range(mean1.shape[0]):
	plt.plot(mean1[i,:], color='darkmagenta', alpha=0.25)
	plt.fill_between([x for x in range(mean1.shape[1])], mean1[i,:], mean1[i,:]+std1[i,:], alpha=0.05, color='darkmagenta')	
	plt.plot(mean2[i,:], color='forestgreen', alpha=0.25)
	plt.fill_between([x for x in range(mean2.shape[1])], mean2[i,:], mean2[i,:]+std2[i,:], alpha=0.05, color='forestgreen')	
plt.yscale('log', base=10)
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.xlabel('Log of cumulative production ratios for prediction')

plt.figure()
for i in range(mean1.shape[1]):
	plt.plot(mean1[:,i], color='darkmagenta', alpha=0.25, label='Technology-specific slope')
	plt.fill_between([x for x in range(mean1.shape[0])], mean1[:,i], mean1[:,i]+std1[:,i], alpha=0.05, color='darkmagenta')	
	plt.plot(mean2[:,i], color='forestgreen', alpha=0.25, label='Average technological slope')
	plt.fill_between([x for x in range(mean2.shape[0])], mean2[:,i], mean2[:,i]+std2[:,i], alpha=0.05, color='forestgreen')	
plt.yscale('log', base=10)
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.xlabel('Log of cumulative production ratios for predictor')

count = count[::-1,:]

predonavg1 = []
predonstd1 = []
predonavg2 = []
predonstd2 = []
for i in range(mean1.shape[0]):
	valueavg1 = 0.0
	valuestd1 = 0.0
	valueavg2 = 0.0
	valuestd2 = 0.0
	for j in range(mean1.shape[1]):
		if counttech[i,j] > 0:
			# valueavg1 += mean1[i,j] * counttech[i,j]/np.sum(counttech[i,:])
			# valuestd1 += std1[i,j] * counttech[i,j]/np.sum(counttech[i,:])
			# valueavg2 += mean2[i,j] * counttech[i,j]/np.sum(counttech[i,:])
			# valuestd2 += std2[i,j] * counttech[i,j]/np.sum(counttech[i,:])
			valueavg1 += mean1[i,j] * count[i,j]/np.sum(count[i,:])
			valuestd1 += std1[i,j] * count[i,j]/np.sum(count[i,:])
			valueavg2 += mean2[i,j] * count[i,j]/np.sum(count[i,:])
			valuestd2 += std2[i,j] * count[i,j]/np.sum(count[i,:])
			# valueavg1 += mean1[i,j] * 1/16/16
			# valuestd1 += std1[i,j] * 1/16/16
			# valueavg2 += mean2[i,j] * 1/16/16
			# valuestd2 += std2[i,j] * 1/16/16
	predonavg1.append(valueavg1)
	predonstd1.append(valuestd1)
	predonavg2.append(valueavg2)
	predonstd2.append(valuestd2)

predonavg1 = np.array(predonavg1)
predonavg2 = np.array(predonavg2)
predonstd1 = np.array(predonstd1)
predonstd2 = np.array(predonstd2)
print(predonavg1, predonstd1)
plt.figure()
plt.plot(predonavg1, color='darkmagenta', alpha=0.25, label='Technology-specific slope')
plt.fill_between([x for x in range(mean1.shape[0])], predonavg1, predonavg1+predonstd1, color='darkmagenta', alpha=0.1)
plt.plot(predonavg2, color='forestgreen', alpha=0.25, label='Average technological slope')
plt.fill_between([x for x in range(mean1.shape[0])], predonavg2, predonavg2+predonstd2, color='forestgreen', alpha=0.1)
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.xlabel('Log of cumulative production ratios for predictor')
plt.ylabel('RMSE (+stdev(RMSE) )')
plt.gcf().legend(loc='lower center')
plt.subplots_adjust(bottom=0.4, top=0.99)


predonavg1 = []
predonstd1 = []
predonavg2 = []
predonstd2 = []
for j in range(mean1.shape[1]):
	valueavg1 = 0.0
	valuestd1 = 0.0
	valueavg2 = 0.0
	valuestd2 = 0.0
	for i in range(mean1.shape[0]):
		if counttech[i,j] > 0:
			# valueavg1 += mean1[i,j] * counttech[i,j]/np.sum(counttech[:,j])
			# valuestd1 += std1[i,j] * counttech[i,j]/np.sum(counttech[:,j])
			# valueavg2 += mean2[i,j] * counttech[i,j]/np.sum(counttech[:,j])
			# valuestd2 += std2[i,j] * counttech[i,j]/np.sum(counttech[:,j])
			valueavg1 += mean1[i,j] * count[i,j]/np.sum(count[:,j])
			valuestd1 += std1[i,j] * count[i,j]/np.sum(count[:,j])
			valueavg2 += mean2[i,j] * count[i,j]/np.sum(count[:,j])
			valuestd2 += std2[i,j] * count[i,j]/np.sum(count[:,j])
			# valueavg1 += mean1[i,j] * 1/16/16
			# valuestd1 += std1[i,j] * 1/16/16
			# valueavg2 += mean2[i,j] * 1/16/16
			# valuestd2 += std2[i,j] * 1/16/16
	predonavg1.append(valueavg1)
	predonstd1.append(valuestd1)
	predonavg2.append(valueavg2)
	predonstd2.append(valuestd2)

predonavg1 = np.array(predonavg1)
predonavg2 = np.array(predonavg2)
predonstd1 = np.array(predonstd1)
predonstd2 = np.array(predonstd2)
print(predonavg1, predonstd1)
plt.figure()
plt.plot(predonavg1, color='darkmagenta', alpha=0.25, label='Technology-specific slope')
plt.fill_between([x for x in range(mean1.shape[0])], predonavg1, predonavg1+predonstd1, color='darkmagenta', alpha=0.1)
plt.plot(predonavg2, color='forestgreen', alpha=0.25, label='Average technological slope')
plt.fill_between([x for x in range(mean1.shape[0])], predonavg2, predonavg2+predonstd2, color='forestgreen', alpha=0.1)
plt.gca().set_xticks([x for x in range(len(frac)-1)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.xlabel('Log of cumulative production ratios for prediction')
plt.ylabel('RMSE (+stdev(RMSE) )')
plt.gcf().legend(loc='lower center')
plt.subplots_adjust(bottom=0.4, top=0.99)
# 	plt.figure()
# 	plt.plot(mean1[i,:], color='forestgreen', alpha=0.25)
# 	plt.fill_between([x for x in range(mean1.shape[1])], mean1[i,:], mean1[i,:]+std1[i,:], alpha=0.05, color='forestgreen')	
# 	plt.plot(mean2[i,:], color='darkmagenta', alpha=0.25)
# 	plt.fill_between([x for x in range(mean2.shape[1])], mean2[i,:], mean2[i,:]+std2[i,:], alpha=0.05, color='darkmagenta')	

# for i in range(mean1.shape[1]):
# 	plt.figure()
# 	plt.plot(mean1[:,i], color='forestgreen', alpha=0.25)
# 	plt.fill_between([x for x in range(mean1.shape[0])], mean1[:,i], mean1[:,i]+std1[:,i], alpha=0.05, color='forestgreen')	
# 	plt.plot(mean2[:,i], color='darkmagenta', alpha=0.25)
# 	plt.fill_between([x for x in range(mean2.shape[0])], mean2[:,i], mean2[:,i]+std2[:,i], alpha=0.05, color='darkmagenta')	

# plt.yscale('log', base=10)
plt.show()
