import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from mpl_toolkits.mplot3d import axes3d 
import cmcrameri as cm
import scipy
matplotlib.use('macosx')
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.style.use('ggplot')
df = pd.read_csv('ExpCurves.csv')
# df = pd.read_csv('NormalizedExpCurves.csv')
# df['Cumulative production'] = df['Normalized cumulative production']
# df['Unit cost'] = df['Normalized unit cost']

cmapp = matplotlib.colormaps['Purples']
cmapg = matplotlib.colormaps['Greens']

method = 'regression'

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
		# for N in range(0-(i==0), -1, -1):
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
								(x[M] - x[i]),
								error, tech])
				dferr2.append([x[i] - x[N],
								(x[M] - x[i]),
								error2, tech])
				if np.abs(error2) < np.abs(error):
					counterr += 1

dferr = pd.DataFrame(dferr, 
					 columns = ['Log of ratios for predictor',
								'Log of ratios for prediction',
								'Error', 'Tech'])
dferr2 = pd.DataFrame(dferr2, 
					 columns = ['Log of ratios for predictor',
								'Log of ratios for prediction',
								'Error', 'Tech'])

# count = 0
# fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
# data = []
# for tech in df['Tech'].unique():
# 	ax[0].plot([count for x in range(len(dferr.loc[dferr['Tech']==tech,'Log of ratios for predictor']))],10**dferr.loc[dferr['Tech']==tech,'Log of ratios for predictor'])
# 	ax[1].plot([count for x in range(len(dferr.loc[dferr['Tech']==tech,'Log of ratios for prediction']))],10**dferr.loc[dferr['Tech']==tech,'Log of ratios for prediction'])
# 	predr = [dferr.loc[dferr['Tech']==tech,'Log of ratios for predictor'].min(),
# 	  dferr.loc[dferr['Tech']==tech,'Log of ratios for predictor'].max()]
# 	predn = [dferr.loc[dferr['Tech']==tech,'Log of ratios for prediction'].min(),
# 	  dferr.loc[dferr['Tech']==tech,'Log of ratios for prediction'].max()]
# 	data.append([tech, predr[0], predr[1], predn[0], predn[1]])
# 	count += 1
# data = pd.DataFrame(data, columns=['Tech', 'Min predictor', 'Max predictor', 'Min prediction', 'Max prediction'])
# predr = [0]
# predn = [0]
# [predr.append(x) for x in list(data['Min predictor'].values)]
# [predn.append(x) for x in list(data['Min prediction'].values)]
# [predr.append(x) for x in list(data['Max predictor'].values)]
# [predn.append(x) for x in list(data['Max prediction'].values)]
# predr.sort()
# predn.sort()
# predr_techs = [[]]
# for x in predr[1:]:
# 	predr_techs.append(predr_techs[-1].copy())
# 	if x in data['Min predictor'].values:
# 		predr_techs[-1].append(data.loc[data['Min predictor']==x, 'Tech'].values[0])
# 	if x in data['Max predictor'].values:
# 		predr_techs[-1] = [t for t in predr_techs[-1] if t not in data.loc[data['Max predictor']==x, 'Tech'].values].copy()
# predn_techs = [[]]
# for x in predn[1:]:
# 	[predn_techs.append(predn_techs[-1].copy())]
# 	if x in data['Min prediction'].values:
# 		predn_techs[-1].append(data.loc[data['Min prediction']==x, 'Tech'].values[0])
# 	if x in data['Max prediction'].values:
# 		predn_techs[-1] = [t for t in predn_techs[-1] if t not in data.loc[data['Max prediction']==x, 'Tech'].values].copy()
# fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
# ax[0].plot(10**np.array(predr), [len(x) for x in predr_techs])
# ax[1].plot(10**np.array(predn), [len(x) for x in predn_techs])
# ax[1].set_xscale('log', base=10)
# # plt.show()

# # for different max ranges
maxsrange = [0.5, 1, 2]
rangess = {0.5: [-1, -0.5, 0, 0.5], 1: [0,1/3,2/3,1], 2: [-1,0.5,1,2]}
rangess = {0.5: [-1, -0.5, 0, 0.5], 1: [0,1/3,2/3,1], 2: [-1,0.5,1,2]}
rangess = {0.5: [1+10**-1, 1+10**-0.5, 1+10**0, 10**0.5], 1: [1+10**0,10**(1/3),10**(2/3),10], 2: [1+10**0,10**(0.5),10,100]}

# maxsrange = [np.log10(10**(2/4)), 10**(1), 10**3]
# rangess = {10**(2/4): [10**i*2/4/4 for i in range(1,4)],
# 	   10**(1): [10**i/4 for i in range(1,4)],
# 	   10**(3): [10**i*3/4 for i in range(1,4)],
# 	   }
techsranges = []
for maxrange in maxsrange:

	## select only technologies available in all intervals
	frac = [*rangess[maxrange]]
	frac = [np.log10(x) for x in frac]
	techs1, techs2 = [], []
	for i in range(len(frac)-1):
		for j in range(len(frac)-1):
			sel1 = dferr.loc[
					(dferr['Log of ratios for predictor']>=frac[0]) &\
					(dferr['Log of ratios for predictor']<=frac[i+1]) & \
					(dferr['Log of ratios for prediction']>frac[j]) &\
					(dferr['Log of ratios for prediction']<=frac[j+1])
					].copy()
			sel2 = dferr2.loc[
					(dferr2['Log of ratios for predictor']>=frac[0]) &\
					(dferr2['Log of ratios for predictor']<=frac[i+1]) & \
					(dferr2['Log of ratios for prediction']>frac[j]) &\
					(dferr2['Log of ratios for prediction']<=frac[j+1])
					].copy()
			techs1.append(sel1['Tech'].unique())
			techs2.append(sel2['Tech'].unique())
	techss = []
	for t in techs1[0]:
		flag = 0
		for j in range(1, len(techs1)):
			if t not in techs1[j]:
				flag = 1
		if flag==0:
			techss.append(t)
	for t in techss:
		for j in range(len(techs2)):
			if t not in techs2[0]:
				techss.remove(t)


	print(maxrange)
	print(techss)
	select1 = dferr.loc[dferr['Tech'].isin(techss)]
	select2 = dferr2.loc[dferr2['Tech'].isin(techss)]
	techsranges.append(techss)
	frac = [*rangess[maxrange]]
	frac = [np.log10(x) for x in frac]
	print(frac)
	# lineplot with confidence intervals
	figl, axl = plt.subplots(3,1, sharex=True, sharey=True, figsize=(6,8))
	dj = (frac[-1] - frac[0])/100
	jfrac = np.arange(frac[0]-dj/2,frac[-1]+dj*3/2,dj)
	print(jfrac[0]+dj/2, jfrac[-2]+dj/2)
	count = np.zeros((len(frac)-1,len(jfrac)-1))
	counttech = np.zeros((len(frac)-1,len(jfrac)-1))
	for i in range(len(frac)-1):
		pct1 = []
		pct2 = []
		for j in range(len(jfrac)-1):
			sel1 = select1.loc[
				(select1['Log of ratios for predictor']>=frac[0]) &\
				(select1['Log of ratios for predictor']<=frac[i+1]) & \
				(select1['Log of ratios for prediction']>jfrac[j]) &\
				(select1['Log of ratios for prediction']<=jfrac[j+1])
				].copy()
			sel2 = select2.loc[
				(select2['Log of ratios for predictor']>=frac[0]) &\
				(select2['Log of ratios for predictor']<=frac[i+1]) & \
				(select2['Log of ratios for prediction']>jfrac[j]) &\
				(select2['Log of ratios for prediction']<=jfrac[j+1])
				].copy()
			for tt in sel1['Tech'].unique():
				sel1.loc[sel1['Tech']==tt,'Weights'] = 1/sel1.loc[sel1['Tech']==tt].count()[0]
				sel2.loc[sel2['Tech']==tt,'Weights'] = 1/sel2.loc[sel2['Tech']==tt].count()[0]
			counttech[i,j] = sel1['Tech'].nunique()
			sel1 = sel1.sort_values(by='Error', ascending=True)
			sel2 = sel2.sort_values(by='Error', ascending=True)
			cumsum = sel1['Weights'].cumsum().round(4)
			cumsum2 = sel2['Weights'].cumsum().round(4)
			pt1, pt2 = [], []
			for q in [0,25,50,75,100]:
			# for q in [0,10,20,30,40,50,60,70,80,90,100]:
				cutoff = sel1['Weights'].sum() * q/100
				cutoff2 = sel2['Weights'].sum() * q/100
				pt1.append(sel1['Error'][cumsum >= cutoff.round(4)].iloc[0])
				pt2.append(sel2['Error'][cumsum2 >= cutoff2.round(4)].iloc[0])
			pct1.append([i,j,*pt1])
			pct2.append([i,j,*pt2])
		pct1 = np.array(pct1)
		pct2 = np.array(pct2)
		axl[i].plot([10**(x+dj/2)  for x in jfrac[:-1]],10**pct1[:,4], color=cmapp(0.7), lw=2)
		axl[i].plot([10**(x+dj/2)  for x in jfrac[:-1]],10**pct2[:,4], color=cmapg(0.7), lw=2)
		for r in range(1,-1,-1):
			axl[i].fill_between([10**(x+dj/2)  for x in jfrac[:-1]], 10**pct1[:,2+r], 10**pct1[:,-r-1], alpha=0.1+0.2*r, color=cmapp(0.7), zorder=-2-r, lw=0)
			axl[i].fill_between([10**(x+dj/2)  for x in jfrac[:-1]], 10**pct2[:,2+r], 10**pct2[:,-r-1], alpha=0.1+0.2*r, color=cmapg(0.7), zorder=-2-r, lw=0)
		# axl[i].plot([10**x - 1 for x in jfrac[:-1]],10**pct1[:,7], color=cmapp(0.7), lw=2)
		# axl[i].plot([10**x - 1 for x in jfrac[:-1]],10**pct2[:,7], color=cmapg(0.7), lw=2)
		# for r in range(4,-1,-1):
		# 	axl[i].fill_between([10**x - 1 for x in jfrac[:-1]], 10**pct1[:,2+r], 10**pct1[:,-r-1], alpha=0.1+0.2*r, color=cmapp(0.7), zorder=-2-r, lw=0)
		# 	axl[i].fill_between([10**x - 1 for x in jfrac[:-1]], 10**pct2[:,2+r], 10**pct2[:,-r-1], alpha=0.1+0.2*r, color=cmapg(0.7), zorder=-2-r, lw=0)
		axl[i].set_ylim(8*1e-2,12)
		axl[i].set_yscale('log', base=10)
		axl[i].set_xscale('log', base=10)
		# axl[i].set_title('Predictor range (ratio of last to first point): from 10^'+str(np.log10(10**frac[i]-1).round(2))+' to 10^'+str(np.log10(10**frac[i+1]-1).round(2)))
	axl[2].set_xlabel('Ratio of future cumulative production to current cumulative production')
	axl[1].annotate('Prediction error (actual to predicted ratio)', 
			xy=(0.025, 0.5), ha='center', va='center', 
			xycoords='figure fraction', rotation=90)
	legend_elements = [
		matplotlib.lines.Line2D([0],[0], c=cmapp(0.7), lw=2, label='Technology-specific slope'),
		matplotlib.lines.Line2D([0],[0], c=cmapg(0.7), lw=2, label='Average technological slope')
	]
	figl.legend(handles=legend_elements, loc='lower center', ncol=2)
	figl.suptitle('Number of technologies always available over the intervals considered: '+str(len(techss)))
	plt.subplots_adjust(bottom=0.125, left=0.1, right=0.9, top=0.875, hspace=0.5)

	# if maxrange != maxsrange[0]:
	# 	for testmaxrange in maxsrange:
	# 		if testmaxrange < maxrange:
	# 			frac = [*rangess[testmaxrange]]
	# 			frac = [np.log10(x) for x in frac]
	# 			mean1 = np.zeros((len(frac)-1,len(frac)-1))
	# 			mean2 = np.zeros((len(frac)-1,len(frac)-1))
	# 			pct1 = []
	# 			pct2 = []
	# 			figl, axl = plt.subplots(3,1, sharex=True, sharey=True, figsize=(6,8))
	# 			dj = (frac[-1] - frac[0])/30
	# 			jfrac = np.arange(frac[0],frac[-1]+dj,dj)
	# 			count = np.zeros((len(frac)-1,len(jfrac)-1))
	# 			counttech = np.zeros((len(frac)-1,len(jfrac)-1))
	# 			for i in range(len(frac)-1):
	# 				pct1 = []
	# 				pct2 = []
	# 				for j in range(len(jfrac)-1):
	# 					sel1 = select1.loc[
	# 						(select1['Log of ratios for predictor']>=frac[0]) &\
	# 						(select1['Log of ratios for predictor']<=frac[i+1]) & \
	# 						(select1['Log of ratios for prediction']>jfrac[j]) &\
	# 						(select1['Log of ratios for prediction']<=jfrac[j+1])
	# 						].copy()
	# 					sel2 = select2.loc[
	# 						(select2['Log of ratios for predictor']>=frac[0]) &\
	# 						(select2['Log of ratios for predictor']<=frac[i+1]) & \
	# 						(select2['Log of ratios for prediction']>jfrac[j]) &\
	# 						(select2['Log of ratios for prediction']<=jfrac[j+1])
	# 						].copy()
	# 					for tt in sel1['Tech'].unique():
	# 						sel1.loc[sel1['Tech']==tt,'Weights'] = 1/sel1.loc[sel1['Tech']==tt].count()[0]
	# 						sel2.loc[sel2['Tech']==tt,'Weights'] = 1/sel2.loc[sel2['Tech']==tt].count()[0]
	# 					counttech[i,j] = sel1['Tech'].nunique()
	# 					sel1 = sel1.sort_values(by='Error', ascending=True)
	# 					sel2 = sel2.sort_values(by='Error', ascending=True)
	# 					cumsum = sel1['Weights'].cumsum().round(4)
	# 					cumsum2 = sel2['Weights'].cumsum().round(4)
	# 					pt1, pt2 = [], []
	# 					for q in [0,25,50,75,100]:
	# 						cutoff = sel1['Weights'].sum() * q/100
	# 						cutoff2 = sel2['Weights'].sum() * q/100
	# 						pt1.append(sel1['Error'][cumsum >= cutoff.round(4)].iloc[0])
	# 						pt2.append(sel2['Error'][cumsum2 >= cutoff2.round(4)].iloc[0])
	# 					pct1.append([i,j,*pt1])
	# 					pct2.append([i,j,*pt2])
	# 				pct1 = np.array(pct1)
	# 				pct2 = np.array(pct2)
	# 				axl[i].plot([10**x  for x in jfrac[:-1]],10**pct1[:,4], color=cmapp(0.7), lw=2)
	# 				axl[i].plot([10**x  for x in jfrac[:-1]],10**pct2[:,4], color=cmapg(0.7), lw=2)
	# 				for r in range(1,-1,-1):
	# 					axl[i].fill_between([10**x  for x in jfrac[:-1]], 10**pct1[:,2+r], 10**pct1[:,-r-1], alpha=0.1+0.2*r, color=cmapp(0.7), zorder=-2-r, lw=0)
	# 					axl[i].fill_between([10**x  for x in jfrac[:-1]], 10**pct2[:,2+r], 10**pct2[:,-r-1], alpha=0.1+0.2*r, color=cmapg(0.7), zorder=-2-r, lw=0)
	# 				axl[i].set_ylim(8*1e-2,12)
	# 				axl[i].set_yscale('log', base=10)
	# 				axl[i].set_xscale('log', base=10)
	# 				axl[i].set_title('Predictor range (ratio of last to first point): from 10^'+str(np.log10(10**frac[i]-1).round(2))+' to 10^'+str(np.log10(10**frac[i+1]-1).round(2)))
	# 			axl[2].set_xlabel('Ratio of increase in cumulative production to current cumulative production')
	# 			axl[1].annotate('Prediction error (actual to predicted ratio)', 
	# 					xy=(0.025, 0.5), ha='center', va='center', 
	# 					xycoords='figure fraction', rotation=90)
	# 			legend_elements = [
	# 				matplotlib.lines.Line2D([0],[0], c=cmapp(0.7), lw=2, label='Technology-specific slope'),
	# 				matplotlib.lines.Line2D([0],[0], c=cmapg(0.7), lw=2, label='Average technological slope')
	# 			]
	# 			figl.legend(handles=legend_elements, loc='lower center', ncol=2)
	# 			figl.suptitle('Number of technologies always available over the intervals considered: '+str(len(techss)))
	# 			plt.subplots_adjust(bottom=0.125, left=0.1, right=0.9, top=0.875, hspace=0.5)


	# mean1 = np.zeros((len(frac)-1,len(frac)-1))
	# mean2 = np.zeros((len(frac)-1,len(frac)-1))
	# pct1 = []
	# pct2 = []
	# count = np.zeros((len(frac)-1,len(frac)-1))
	# counttech = np.zeros((len(frac)-1,len(frac)-1))
	# for i in range(len(frac)-1):
	# 	for j in range(len(frac)-1):
	# 		sel1 = select1.loc[
	# 				(select1['Log of ratios for predictor']>frac[i]) &\
	# 				(select1['Log of ratios for predictor']<=frac[i+1]) & \
	# 				(select1['Log of ratios for prediction']>frac[j]) &\
	# 				(select1['Log of ratios for prediction']<=frac[j+1])
	# 				].copy()
	# 		sel2 = select2.loc[
	# 				(select2['Log of ratios for predictor']>frac[i]) &\
	# 				(select2['Log of ratios for predictor']<=frac[i+1]) & \
	# 				(select2['Log of ratios for prediction']>frac[j]) &\
	# 				(select2['Log of ratios for prediction']<=frac[j+1])
	# 				].copy()
	# 		# # RMSE COMPUTATION
	# 		# mean1[i,j] = 0.0
	# 		# mean2[i,j] = 0.0
	# 		# ntechs = sel1['Tech'].nunique()
	# 		# print(sel1['Tech'].nunique(), sel2['Tech'].nunique())
	# 		# for tech in sel1['Tech'].unique():
	# 		# 	s1 = sel1.loc[sel1['Tech']==tech].copy()
	# 		# 	s2 = sel2.loc[sel2['Tech']==tech].copy()
	# 		# 	m1 = 1/(s1.count()[0]) * np.sum(s1['Error'].values**2)
	# 		# 	mean1[i,j] += 1/(ntechs) * m1
	# 		# 	m2 = 1/(s2.count()[0]) * np.sum(s2['Error'].values**2)
	# 		# 	mean2[i,j] += 1/(ntechs) * m2
	# 		# mean1[i,j] = mean1[i,j]**0.5
	# 		# mean2[i,j] = mean2[i,j]**0.5
	# 		# count[i,j] = (sel1['Error'].count())
	# 		# counttech[i,j] = (sel1['Tech'].nunique())
	# 		# PERCENTILES COMPUTATION
	# 		for tt in sel1['Tech'].unique():
	# 			sel1.loc[sel1['Tech']==tt,'Weights'] = 1/sel1.loc[sel1['Tech']==tt].count()[0]
	# 			sel2.loc[sel2['Tech']==tt,'Weights'] = 1/sel2.loc[sel2['Tech']==tt].count()[0]
	# 		counttech[i,j] = sel1['Tech'].nunique()
	# 		sel1 = sel1.sort_values(by='Error', ascending=True)
	# 		sel2 = sel2.sort_values(by='Error', ascending=True)
	# 		cumsum = sel1['Weights'].cumsum().round(4)
	# 		cumsum2 = sel2['Weights'].cumsum().round(4)
	# 		pt1, pt2 = [], []
	# 		for q in [0,25,50,75,100]:
	# 			cutoff = sel1['Weights'].sum() * q/100
	# 			cutoff2 = sel2['Weights'].sum() * q/100
	# 			pt1.append(sel1['Error'][cumsum >= cutoff.round(4)].iloc[0])
	# 			pt2.append(sel2['Error'][cumsum2 >= cutoff2.round(4)].iloc[0])
	# 		pct1.append([i,j,*pt1])
	# 		pct2.append([i,j,*pt2])

	# pct1 = pd.DataFrame(pct1, columns=['Predictor','Prediction','whislo','q1','med','q3','whishi'])
	# pct2 = pd.DataFrame(pct2, columns=['Predictor','Prediction','whislo','q1','med','q3','whishi'])
	# # fig, ax = plt.subplots(3,1, sharex=True, sharey=True, figsize=(6,8))
	# # cmapp = matplotlib.colormaps['Purples']
	# # cmapg = matplotlib.colormaps['Greens']
	# # for i in range(len(frac)-1):
	# # 	ax[i].bar(np.array([0,5,10])-0.5, mean1[i,:], color=cmapp(0.7))
	# # 	ax[i].bar(np.array([0,5,10])+0.5, mean2[i,:], color=cmapg(0.7))
	# # 	ax[i].set_xticks([0,5,10],['From '+str(int(10**frac[x]))+' to '+str(int(10**frac[x+1])) for x in range(len(frac)-1)])
	# # 	ax[i].set_title('Predictor range (ratio of last to first point) '+str(round(10**frac[i],2))+' to '+str(round(10**frac[i+1],2)))
	# # ax[2].set_xlabel('Ratio of future cumulative production to current cumulative production')
	# # ax[1].annotate('RMSE (log of actual to predicted ratio)', 
	# # 		xy=(0.025, 0.5), ha='center', va='center', 
	# # 		xycoords='figure fraction', rotation=90)
	# # fig.legend(['Technology-specific slope','Average technological slope'], loc='lower center', ncol=2)
	# # fig.suptitle('Number of technologies always available over the intervals considered: '+str(int(counttech[i,j])))
	# # plt.subplots_adjust(bottom=0.125, left=0.1, right=0.9, top=0.875, hspace=0.5)
	
	# fig, ax = plt.subplots(3,1, sharex=True, sharey=True, figsize=(6,8))
	# fig2, ax2 = plt.subplots(3,1, sharex=True, sharey=True, figsize=(6,8))
	# for i in range(len(frac)-1):
	# 	pcline1, pcline2 = [], []
	# 	for j in range(len(frac)-1):
	# 		p1 = pct1.loc[(pct1['Predictor']==i) & (pct1['Prediction']==j)]
	# 		p2 = pct2.loc[(pct2['Predictor']==i) & (pct2['Prediction']==j)]
	# 		s1 , s2 = {}, {}
	# 		for c in p1.columns[2:]:
	# 			s1[c] = 10**p1[c].values[0]
	# 			s2[c] = 10**p2[c].values[0]
	# 		props1 = dict(color=cmapp(0.7), lw=2)
	# 		props2 = dict(color=cmapg(0.7), lw=2)
	# 		ax[i].bxp([s1], positions=[j*5-0.5], showfliers=False, 
	#      		boxprops = props1, whiskerprops = props1, capprops = props1)
	# 		ax[i].bxp([s2], positions=[j*5+0.5], showfliers=False, 
	#      		boxprops = props2, whiskerprops = props2, capprops = props2)
	# 		pcline1.append([s1[x] for x in p1.columns[2:]])
	# 		pcline2.append([s2[x] for x in p1.columns[2:]])
	# 	pcline1 = np.array(pcline1)
	# 	pcline2 = np.array(pcline2)
	# 	ax2[i].plot(frac[:-1],pcline1[:,2], color=cmapp(0.7), lw=2)
	# 	ax2[i].plot(frac[:-1],pcline2[:,2], color=cmapg(0.7), lw=2)
	# 	for r in range(1,-1,-1):
	# 		ax2[i].fill_between(frac[:-1], pcline1[:,r], pcline1[:,-r-1], alpha=0.1+0.2*r, color=cmapp(0.7), zorder=-2-r, lw=0)
	# 		ax2[i].fill_between(frac[:-1], pcline2[:,r], pcline2[:,-r-1], alpha=0.1+0.2*r, color=cmapg(0.7), zorder=-2-r, lw=0)
	
	# 	ax[i].set_ylim(8*1e-2,12)
	# 	ax[i].set_yscale('log', base=10)
	# 	ax[i].set_xticks([0,5,10],['From 10^'+str(np.log10(10**frac[x]-1).round(2))+' to 10^'+str(np.log10(10**frac[x+1]-1).round(2)) for x in range(len(frac)-1)])
	# 	ax[i].set_title('Predictor range (ratio of last to first point): from 10^'+str(np.log10(10**frac[i]-1).round(2))+' to 10^'+str(np.log10(10**frac[i+1]-1).round(2)))
	# ax[2].set_xlabel('Ratio of increase in cumulative production to current cumulative production')
	# ax[1].annotate('Prediction error (actual to predicted ratio)', 
	# 		xy=(0.025, 0.5), ha='center', va='center', 
	# 		xycoords='figure fraction', rotation=90)
	# legend_elements = [
	# 	matplotlib.lines.Line2D([0],[0], c=cmapp(0.7), lw=2, label='Technology-specific slope'),
	# 	matplotlib.lines.Line2D([0],[0], c=cmapg(0.7), lw=2, label='Average technological slope')
	# ]
	# fig.legend(handles=legend_elements, loc='lower center', ncol=2)
	# fig.suptitle('Number of technologies always available over the intervals considered: '+str(int(counttech[i,j])))
	# plt.subplots_adjust(bottom=0.125, left=0.1, right=0.9, top=0.875, hspace=0.5)

	# if maxrange != maxsrange[0]:
	# 	for testmaxrange in maxsrange:
	# 		if testmaxrange < maxrange:
	# 			frac = [*rangess[testmaxrange]]
	# 			frac = [np.log10(10**x) + 1 for x in frac]
	# 			mean1 = np.zeros((len(frac)-1,len(frac)-1))
	# 			mean2 = np.zeros((len(frac)-1,len(frac)-1))
	# 			pct1 = []
	# 			pct2 = []
	# 			count = np.zeros((len(frac)-1,len(frac)-1))
	# 			counttech = np.zeros((len(frac)-1,len(frac)-1))
	# 			for i in range(len(frac)-1):
	# 				for j in range(len(frac)-1):
	# 					sel1 = select1.loc[
	# 							(select1['Log of ratios for predictor']>frac[i]) &\
	# 							(select1['Log of ratios for predictor']<=frac[i+1]) & \
	# 							(select1['Log of ratios for prediction']>frac[j]) &\
	# 							(select1['Log of ratios for prediction']<=frac[j+1])
	# 							].copy()
	# 					sel2 = select2.loc[
	# 							(select2['Log of ratios for predictor']>frac[i]) &\
	# 							(select2['Log of ratios for predictor']<=frac[i+1]) & \
	# 							(select2['Log of ratios for prediction']>frac[j]) &\
	# 							(select2['Log of ratios for prediction']<=frac[j+1])
	# 							].copy()
	# 					# # RMSE COMPUTATION
	# 					# mean1[i,j] = 0.0
	# 					# mean2[i,j] = 0.0
	# 					# ntechs = sel1['Tech'].nunique()
	# 					# print(sel1['Tech'].nunique(), sel2['Tech'].nunique())
	# 					# for tech in sel1['Tech'].unique():
	# 					# 	s1 = sel1.loc[sel1['Tech']==tech].copy()
	# 					# 	s2 = sel2.loc[sel2['Tech']==tech].copy()
	# 					# 	m1 = 1/(s1.count()[0]) * np.sum(s1['Error'].values**2)
	# 					# 	mean1[i,j] += 1/(ntechs) * m1
	# 					# 	m2 = 1/(s2.count()[0]) * np.sum(s2['Error'].values**2)
	# 					# 	mean2[i,j] += 1/(ntechs) * m2
	# 					# mean1[i,j] = mean1[i,j]**0.5
	# 					# mean2[i,j] = mean2[i,j]**0.5
	# 					# count[i,j] = (sel1['Error'].count())
	# 					# counttech[i,j] = (sel1['Tech'].nunique())
	# 					# PERCENTILES COMPUTATION
	# 					for tt in sel1['Tech'].unique():
	# 						sel1.loc[sel1['Tech']==tt,'Weights'] = 1/sel1.loc[sel1['Tech']==tt].count()[0]
	# 						sel2.loc[sel2['Tech']==tt,'Weights'] = 1/sel2.loc[sel2['Tech']==tt].count()[0]
	# 					counttech[i,j] = sel1['Tech'].nunique()
	# 					sel1 = sel1.sort_values(by='Error', ascending=True)
	# 					sel2 = sel2.sort_values(by='Error', ascending=True)
	# 					cumsum = sel1['Weights'].cumsum().round(4)
	# 					cumsum2 = sel2['Weights'].cumsum().round(4)
	# 					pt1, pt2 = [], []
	# 					for q in [0,25,50,75,100]:
	# 						cutoff = sel1['Weights'].sum() * q/100
	# 						cutoff2 = sel2['Weights'].sum() * q/100
	# 						pt1.append(sel1['Error'][cumsum >= cutoff.round(4)].iloc[0])
	# 						pt2.append(sel2['Error'][cumsum2 >= cutoff2.round(4)].iloc[0])
	# 					pct1.append([i,j,*pt1])
	# 					pct2.append([i,j,*pt2])
	# 			print(counttech)
	# 			pct1 = pd.DataFrame(pct1, columns=['Predictor','Prediction','whislo','q1','med','q3','whishi'])
	# 			pct2 = pd.DataFrame(pct2, columns=['Predictor','Prediction','whislo','q1','med','q3','whishi'])
	# 			# fig, ax = plt.subplots(3,1, sharex=True, sharey=True, figsize=(6,8))
	# 			# cmapp = matplotlib.colormaps['Purples']
	# 			# cmapg = matplotlib.colormaps['Greens']
	# 			# for i in range(len(frac)-1):
	# 			# 	ax[i].bar(np.array([0,5,10])-0.5, mean1[i,:], color=cmapp(0.7))
	# 			# 	ax[i].bar(np.array([0,5,10])+0.5, mean2[i,:], color=cmapg(0.7))
	# 			# 	ax[i].set_xticks([0,5,10],['From '+str(int(10**frac[x]))+' to '+str(int(10**frac[x+1])) for x in range(len(frac)-1)])
	# 			# 	ax[i].set_title('Predictor range (ratio of last to first point) '+str(round(10**frac[i],2))+' to '+str(round(10**frac[i+1],2)))
	# 			# ax[2].set_xlabel('Ratio of future cumulative production to current cumulative production')
	# 			# ax[1].annotate('RMSE (log of actual to predicted ratio)', 
	# 			# 		xy=(0.025, 0.5), ha='center', va='center', 
	# 			# 		xycoords='figure fraction', rotation=90)
	# 			# fig.legend(['Technology-specific slope','Average technological slope'], loc='lower center', ncol=2)
	# 			# fig.suptitle('Number of technologies always available over the intervals considered: '+str(int(counttech[i,j])))
	# 			# plt.subplots_adjust(bottom=0.125, left=0.1, right=0.9, top=0.875, hspace=0.5)

	# 			fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6,8))
	# 			cmapp = matplotlib.colormaps['Purples']
	# 			cmapg = matplotlib.colormaps['Greens']
	# 			for i in range(len(frac)-1):
	# 				for j in range(len(frac)-1):
	# 					p1 = pct1.loc[(pct1['Predictor']==i) & (pct1['Prediction']==j)]
	# 					p2 = pct2.loc[(pct2['Predictor']==i) & (pct2['Prediction']==j)]
	# 					s1 , s2 = {}, {}
	# 					for c in p1.columns[2:]:
	# 						s1[c] = 10**p1[c].values[0]
	# 						s2[c] = 10**p2[c].values[0]
	# 					props1 = dict(color=cmapp(0.7), lw=2)
	# 					props2 = dict(color=cmapg(0.7), lw=2)
	# 					ax[i].bxp([s1], positions=[j*5-0.5], showfliers=False, 
	# 						boxprops = props1, whiskerprops = props1, capprops = props1)
	# 					ax[i].bxp([s2], positions=[j*5+0.5], showfliers=False, 
	# 						boxprops = props2, whiskerprops = props2, capprops = props2)
	# 				ax[i].set_ylim(8*1e-2,12)
	# 				ax[i].set_yscale('log', base=10)
	# 				ax[i].set_xticks([0,5,10],['From 10^'+str(np.log10(10**frac[x]-1).round(2))+' to 10^'+str(np.log10(10**frac[x+1]-1).round(2)) for x in range(len(frac)-1)])
	# 				ax[i].set_title('Predictor range (ratio of last to first point): from 10^'+str(np.log10(10**frac[i]-1).round(2))+' to 10^'+str(np.log10(10**frac[i+1]-1).round(2)))
	# 			ax[2].set_xlabel('Ratio of increase in cumulative production to current cumulative production')
	# 			ax[1].annotate('Prediction error (actual to predicted ratio)', 
	# 					xy=(0.025, 0.5), ha='center', va='center', 
	# 					xycoords='figure fraction', rotation=90)
	# 			legend_elements = [
	# 				matplotlib.lines.Line2D([0],[0], c=cmapp(0.7), lw=2, label='Technology-specific slope'),
	# 				matplotlib.lines.Line2D([0],[0], c=cmapg(0.7), lw=2, label='Average technological slope')
	# 			]
	# 			fig.legend(handles=legend_elements, loc='lower center', ncol=2)
	# 			fig.suptitle('Testing the '+str(int(counttech[i,j]))+' technologies always available'+\
	# 				' between 10^'+str(np.log10(10**rangess[maxrange][0]).round(2))+' and 10^'+str(np.log10(10**rangess[maxrange][-1]).round(2))+\
	# 					'\n on the interval from 10^'+str(np.log10(10**frac[0]-1).round(2))+' to 10^'+str(np.log10(10**frac[-1]-1).round(2)))
	# 			plt.subplots_adjust(bottom=0.125, left=0.1, right=0.9, top=0.875, hspace=0.5)


fig, ax = plt.subplots()
ax.bar([5 * i for i in range(len(maxsrange))], [len(v) for v in techsranges], edgecolor='k', facecolor='None')
ax.bar([1], len([x for x in techsranges[0] if x in techsranges[1]]), facecolor='blue')
# ax.bar([2], len([x for x in techsranges[0] if x in techsranges[2]]), facecolor='darkorange')
# ax.bar([3], len([x for x in techsranges[0] if x in techsranges[1] and x in techsranges[2]]), facecolor='forestgreen')
# ax.bar([6], len([x for x in techsranges[1] if x in techsranges[2]]), facecolor='darkmagenta')
# print([x for x in techsranges[0] if x in techsranges[2]])
# print([x for x in techsranges[0] if x in techsranges[1] and x in techsranges[2]])
ax.set_xticks([5 * i for i in range(len(maxsrange))],['From 10^'+str(rangess[x][0])+' to 10^'+str(rangess[x][-1]) for x in maxsrange])
ax.set_xlabel('Intervals examined')
ax.set_ylabel('Number of technologies always available in the interval')
legend_elements = [
	matplotlib.patches.Patch(facecolor='white', edgecolor='k', label='Total count' ),
	matplotlib.patches.Patch(facecolor='blue', label='Present in first and second interval' ),
	# matplotlib.patches.Patch(facecolor='darkorange', label='Present in first and third interval' ),
	# matplotlib.patches.Patch(facecolor='forestgreen', label='Present in first, second, and third interval' ),
	# matplotlib.patches.Patch(facecolor='darkmagenta', label='Present in second and third interval' ),
]
fig.legend(handles=legend_elements,loc='lower center')
fig.subplots_adjust(bottom=0.3)

plt.show()

