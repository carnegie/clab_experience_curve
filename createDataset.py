import pandas as pd
import os, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cmcrameri
import matplotlib
from matplotlib import rc
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cmcrameri.cm.batlowS(np.linspace(0,1,100)))
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# prepare figure for plots
fig, ax = plt.subplots(1, 1, figsize=(9,7))

# folder name
datafolder = 'expCurveData'
# for all files in folder
for file in os.listdir(datafolder):
	# read file and drop nan
	f = pd.read_csv(datafolder+os.path.sep+file)
	f = f.dropna()
	# compute cumulative sum of units produce and normalize price and cumulative units produced
	ax.plot(f[f.columns[3]], f[f.columns[0]], lw = 1,
	  alpha = 0.5, marker = 'o', markersize = 2)
	# ax[0].plot(f[f.columns[3]], f[f.columns[0]], lw = 0.75,
	#   alpha = 0.25, marker = 'o', markersize = 2)
	f['Normalized cumulative production'] = f[f.columns[3]] /\
		  f[f.columns[3]].values[0]
	f['Normalized unit cost'] = f[f.columns[0]] /\
		  f[f.columns[0]].values[0]
	# get the length of the observation series in years
	# f[f.columns[1]] = f[f.columns[1]] - f[f.columns[1]].values[0]
	# plot as step functions
	# ax[1].plot(f['Normalized cumulative production'],
	#      f['Normalized unit cost'], 'o-', 
	# 	 lw = 0.75, markersize=2, alpha=0.25)
	# save unit cost, cumulative production, and name of technology
	f['Tech'] = file[:-4]
	f = f[[f.columns[0], f.columns[1], f.columns[3],
		 'Normalized unit cost', 'Normalized cumulative production',
		 'Tech']].copy()
	f.columns = ['Unit cost','Year','Cumulative production',
		 'Normalized unit cost', 'Normalized cumulative production',
	     'Tech']
	# append to dataframe
	try:
		df = pd.concat([df, f])
	except NameError:
		df = f
# set axes styling
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)	
ax.set_ylabel('Unit cost')
ax.set_xlabel('Cumulative production')
ax.set_title('Raw data')
# ax[0].set_xscale('log', base=10)
# ax[0].set_yscale('log', base=10)	
# ax[0].set_ylabel('Unit cost')
# ax[0].set_xlabel('Cumulative production')
# ax[0].set_title('Raw data')
# ax[1].set_xscale('log', base=10)
# ax[1].set_yscale('log', base=10)	
# ax[1].set_ylabel('Unit cost')
# ax[1].set_xlabel('Cumulative production')
# ax[1].set_title('Data normalized using the first point')
plt.show()
# figure reporting the number of available technologies for number of units produced
fig, ax = plt.subplots()
nunits = [1]
for tech in df['Tech'].unique():
	nunits.append(df.loc[df['Tech']==tech,'Normalized cumulative production'].max())
nunits.sort()
ax.step(nunits,[x for x in range(len(nunits),0,-1)], where='post')
ax.set_xscale('log', base=10)
ax.set_ylabel('Number of technologies available')
ax.set_xlabel('Normalized cumulative production')

# # figure plotting number of years available against number of units
# fig, ax = plt.subplots(1, 3, figsize=(15,6))
# nyears =[]
# nunits = []
# for tech in df['Tech'].unique():
# 	nunits.append(df.loc[df['Tech']==tech, 
# 		      'Normalized cumulative production'].max())
# 	nyears.append(df.loc[df['Tech']==tech, df.columns[1]].max())
# ax[0].scatter(nunits,nyears, edgecolor = 'tab:blue',
# 	       facecolors = "None", alpha = 0.75)
# ax[0].set_xscale('log', base=10)
# ax[0].set_xlabel('Total number of units measured')
# ax[0].set_ylabel('Length of experience curve in years')
# ax[0].plot([0.99999,9.9e9],[10,10],'r--', 
# 	   zorder = -1, alpha = 0.5)
# ax[0].annotate('10 years',(1e8,13), 
# 	       ha = 'center', va = 'center')
# ax[0].plot([20,20],[1,130],'r--', zorder = -1, alpha = 0.5)
# ax[0].annotate('20 cumulative units',(25,100),
# 	        ha = 'left', va = 'center')

# # plotting number of points vs number of data vs number of cumulative production 
# # useful to check data quality
# nunits =[]
# npoints = []
# for tech in df['Tech'].unique():
# 	nunits.append(df.loc[df['Tech']==tech, 
# 		      'Normalized cumulative production'].max())
# 	npoints.append(df.loc[df['Tech']==tech, df.columns[1]].count())
# ax[1].scatter(nunits,npoints, edgecolor = 'tab:blue',
# 	       facecolors = "None", alpha = 0.75)
# ax[1].set_xscale('log', base=10)
# ax[1].set_xlabel('Measured increase in cumulative number of units')
# ax[1].set_ylabel('Number of data points available')
# ax[1].plot([0.99999,9.9e9],[10,10],'r--', 
# 	   zorder = -1, alpha = 0.5)
# ax[1].annotate('10 data points',(1e8,13), 
# 	       ha = 'center', va = 'center')
# ax[1].plot([20,20],[1,130],'r--', 
# 	   zorder = -1, alpha = 0.5)
# ax[1].annotate('20 cumulative units',(25,100),
# 	        ha = 'left', va = 'center')
# ax[2].scatter(nyears, npoints, 
# 	      edgecolor = 'tab:blue', 
# 		  facecolors = "None", alpha = 0.75)
# ax[2].set_xlabel('Length of experience curve in years')
# ax[2].set_ylabel('Number of data points available')
# ax[2].plot([1,130],[10,10],'r--', 
# 	   zorder = -1, alpha = 0.5)
# ax[2].annotate('10 years',(13,100), 
# 	       ha = 'left', va = 'center')
# ax[2].plot([10,10],[1,130],'r--', 
# 	   zorder = -1, alpha = 0.5)
# ax[2].annotate('10 data points',(100,13), ha = 'center', va = 'center')

plt.show()
print('Saving dataframes to csv files...')

# save dataframe
df_ = df[['Tech','Cumulative production','Year','Unit cost']].copy()
df_.to_csv('ExpCurves.csv', index=False)

# save normalized dataframe
df_ = df[['Tech','Normalized cumulative production','Year','Normalized unit cost']].copy()
df_.to_csv('NormalizedExpCurves.csv', index=False)


print('Running simple regression for prediction examples ...')
# predict for each technology using only available data 
f_values = [] # here the p values for the f test for each single regression are stored
slopes = []
for tech in df['Tech'].unique():
	select = df.loc[df['Tech']==tech].copy()
	plt.figure()
	plt.scatter(select['Cumulative production'],
	      select['Unit cost'], color='r', marker='o')
	x = np.log10(select['Cumulative production'].values)
	y = np.log10(select['Unit cost'].values)
	x = sm.add_constant(x)
	model = sm.OLS(endog=y, exog=x)
	res = model.fit()
	slopes.append(res.params[1])
	f_values.append(res.f_pvalue)
	# for npoints in range(2,len(select['Unit cost'].values)):
	# 	# estimate experience curve using simple linear regression
	# 	x = np.log10(select['Cumulative production'].values[:npoints])
	# 	y = np.log10(select['Unit cost'].values[:npoints])
	# 	x = sm.add_constant(x)
	# 	model = sm.OLS(endog=y, exog=x)
	# 	res = model.fit()
	# 	x_predict = np.log10(
	# 		select['Cumulative production'].values[npoints-1:])
	# 	x_predict = sm.add_constant(x_predict)
	# 	predictions = 10**(res.predict(x_predict))
	# 	# adjust by applying experience from last point onward
	# 	last_error = np.log10(
	# 		select['Unit cost'].values[npoints-1]) - \
	# 			res.predict(x_predict)[0]
	# 	adjusted_predictions = predictions * 10 ** last_error
	# 	plt.plot(
	# 		select['Cumulative production'].values[npoints-1:],
	# 		  predictions, 'k:')
	# 	plt.plot(
	# 		select['Cumulative production'].values[npoints-1:npoints+1],
	# 		  adjusted_predictions[:2], 'r:')
	# 	# predicting next value estimating experience curve based on last two points
	# 	x = np.log10(select['Cumulative production'].values[npoints-2:npoints])
	# 	x = sm.add_constant(x)
	# 	y = np.log10(select['Unit cost'].values[npoints-2:npoints])
	# 	model = sm.OLS(endog=y, exog=x)
	# 	res = model.fit()
	# 	x_predict = np.log10(select['Cumulative production'].values[npoints-1:npoints+1])
	# 	x_predict = sm.add_constant(x_predict)
	# 	predictions_ = 10**(res.predict(x_predict))
	# 	plt.plot(select['Cumulative production'].values[npoints-1:npoints+1], predictions_, 'g:')
	# legend_elements = [
	# 	matplotlib.lines.Line2D([0],[0], lw=0, marker='o', color='red', label='Obs'),
	# 	matplotlib.lines.Line2D([0],[0], lw=1, linestyle='--', color='k', label='Exp. curve estimate'),
	# 	matplotlib.lines.Line2D([0],[0], lw=1, linestyle='--', color='r', label='Exp. curve estimate from last point'),
	# 	matplotlib.lines.Line2D([0],[0], lw=1, linestyle='--', color='g', label='Exp. curve estimated from last two points'),
	# ]
	# plt.legend(handles=legend_elements)
	# plt.xscale('log', base=10)
	# plt.yscale('log', base=10)
	# plt.ylabel('Unit cost')
	# plt.xlabel('Cumulative production')
	# plt.title(select['Tech'].values[0])
	# plt.show()
print('Mean slope is ',np.mean(slopes))
print('Mean learning rate is ',1 - 2**np.mean(slopes))
print('The percentage of technologies for which'+
      ' the f-test statistic is significant is '+
	  str(round(100*sum(np.asarray(f_values)<0.05)/len(f_values),1))+
	  '%')

# using all data to compute statistics 
# x = np.log10(df['Cumulative production'].values)
# y = np.log10(df['Unit cost'].values)
# x = sm.add_constant(x)
# model = sm.OLS(endog=y, exog=x)
# res = model.fit()
# slopes.append(res.params[1])

# print(np.mean(slopes))
# print(1 - 2**np.mean(slopes))

# plt.show()


		

		