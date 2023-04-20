import pandas as pd
import os, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# prepare figure for plots
fig, ax = plt.subplots(1, 2, figsize=(15,6))

# folder name
datafolder = 'expCurveData'
# for all files in folder
for file in os.listdir(datafolder):
	# read file and drop nan
	f = pd.read_csv(datafolder+os.path.sep+file)
	f = f.dropna()
	# compute cumulative sum of units produce and normalize price and cumulative units produced
	f[f.columns[2]] = f[f.columns[2]].cumsum()
	ax[0].plot(f[f.columns[2]], f[f.columns[0]], lw = 0.75,
	  alpha = 0.25, marker = 'o', markersize = 2)
	f[f.columns[2]] = f[f.columns[2]] / f[f.columns[2]].values[0]
	f[f.columns[0]] = f[f.columns[0]] / f[f.columns[0]].values[0]
	# get the length of the observation series in years
	f[f.columns[1]] = f[f.columns[1]] - f[f.columns[1]].values[0]
	# plot as step functions
	ax[1].plot(f[f.columns[2]], f[f.columns[0]], 'o-', 
	 lw = 0.75, markersize=2, alpha=0.25)
	# ax[1].step(f[f.columns[2]], f[f.columns[0]], 'o-', 
	#  lw = 0.75, where='post', markersize=2, alpha=0.25)
	# save unit cost, cumulative production, and name of technology
	f['Tech'] = file
	f = f[[f.columns[0], f.columns[1], f.columns[2], 'Tech']].copy()
	f.columns = ['Unit cost','Year','Cumulative production','Tech']
	# append to dataframe
	try:
		df = pd.concat([df, f])
	except NameError:
		df = f
# set axes styling
ax[0].set_xscale('log', base=10)
ax[0].set_yscale('log', base=10)	
ax[0].set_ylabel('Unit cost')
ax[0].set_xlabel('Cumulative production')
ax[0].set_title('Raw data')
ax[1].set_xscale('log', base=10)
ax[1].set_yscale('log', base=10)	
ax[1].set_ylabel('Unit cost')
ax[1].set_xlabel('Cumulative production')
ax[1].set_title('Data normalized using the first point')

# figure reporting the number of available technologies for number of units produced
fig, ax = plt.subplots()
nunits =[1]
for tech in df['Tech'].unique():
	nunits.append(df.loc[df['Tech']==tech, df.columns[2]].max())
nunits.sort()
ax.step(nunits,[x for x in range(len(nunits),0,-1)], where='post')
ax.set_xscale('log', base=10)
ax.set_ylabel('Number of technologies available')
ax.set_xlabel('Normalized cumulative production')

# figure reporting the number of available technologies for number of years measured
fig, ax = plt.subplots()
nyears =[1]
for tech in df['Tech'].unique():
	nyears.append(df.loc[df['Tech']==tech, df.columns[1]].max())
nyears.sort()
ax.step(nyears,[x for x in range(len(nunits),0,-1)], where='post')
# ax.set_xscale('log', base=10)
ax.set_ylabel('Number of technologies available')
ax.set_xlabel('Length of experience curve in years')

# figure plotting number of years available against number of units
fig, ax = plt.subplots(1, 3, figsize=(15,6))
nyears =[]
nunits = []
for tech in df['Tech'].unique():
	nunits.append(df.loc[df['Tech']==tech, df.columns[2]].max())
	nyears.append(df.loc[df['Tech']==tech, df.columns[1]].max())
ax[0].scatter(nunits,nyears, edgecolor = 'tab:blue', facecolors = "None", alpha = 0.75)
ax[0].set_xscale('log', base=10)
ax[0].set_xlabel('Total number of units measured')
ax[0].set_ylabel('Length of experience curve in years')
ax[0].plot([0.99999,9.9e9],[10,10],'r--', zorder = -1, alpha = 0.5)
ax[0].annotate('10 years',(1e8,13), ha = 'center', va = 'center')
ax[0].plot([20,20],[1,130],'r--', zorder = -1, alpha = 0.5)
ax[0].annotate('20 cumulative units',(25,100), ha = 'left', va = 'center')

# plotting number of points vs number of data vs number of cumulative production 
# useful to check data quality
nunits =[]
npoints = []
for tech in df['Tech'].unique():
	nunits.append(df.loc[df['Tech']==tech, df.columns[2]].max())
	npoints.append(df.loc[df['Tech']==tech, df.columns[1]].count())
ax[1].scatter(nunits,npoints, edgecolor = 'tab:blue', facecolors = "None", alpha = 0.75)
ax[1].set_xscale('log', base=10)
ax[1].set_xlabel('Measured increase in cumulative number of units')
ax[1].set_ylabel('Number of data points available')
ax[1].plot([0.99999,9.9e9],[10,10],'r--', zorder = -1, alpha = 0.5)
ax[1].annotate('10 data points',(1e8,13), ha = 'center', va = 'center')
ax[1].plot([20,20],[1,130],'r--', zorder = -1, alpha = 0.5)
ax[1].annotate('20 cumulative units',(25,100), ha = 'left', va = 'center')
ax[2].scatter(nyears, npoints, edgecolor = 'tab:blue', facecolors = "None", alpha = 0.75)
ax[2].set_xlabel('Length of experience curve in years')
ax[2].set_ylabel('Number of data points available')
ax[2].plot([1,130],[10,10],'r--', zorder = -1, alpha = 0.5)
ax[2].annotate('10 years',(13,100), ha = 'left', va = 'center')
ax[2].plot([10,10],[1,130],'r--', zorder = -1, alpha = 0.5)
ax[2].annotate('10 data points',(100,13), ha = 'center', va = 'center')
plt.show()

df = df[['Unit cost','Year','Cumulative production','Tech']].copy()

# save dataframe
df.to_csv('ExpCurves.csv', index=False)

# predict for each technology using only available data 
for tech in df['Tech'].unique():
	select = df.loc[df['Tech']==tech].copy()
	plt.figure()
	plt.scatter(select['Cumulative production'], select['Unit cost'], color='r', marker='o')
	for npoints in range(2,len(select['Unit cost'].values)):
		# estimate experience curve using simple linear regression
		x = np.log10(select['Cumulative production'].values[:npoints])
		y = np.log10(select['Unit cost'].values[:npoints])
		x = sm.add_constant(x)
		model = sm.OLS(endog=y, exog=x)
		res = model.fit()
		x_predict = np.log10(select['Cumulative production'].values[npoints-1:])
		x_predict = sm.add_constant(x_predict)
		predictions = 10**(res.predict(x_predict))
		# adjust by applying experience from last point onward
		last_error = np.log10(select['Unit cost'].values[npoints-1]) - res.predict(x_predict)[0]
		adjusted_predictions = predictions * 10 ** last_error
		plt.plot(select['Cumulative production'].values[npoints-1:], predictions, 'k:')
		plt.plot(select['Cumulative production'].values[npoints-1:npoints+1], adjusted_predictions[:2], 'r:')
		# predicting next value estimating local experience curve based on last two points
		x = np.log10(select['Cumulative production'].values[npoints-2:npoints])
		x = sm.add_constant(x)
		y = np.log10(select['Unit cost'].values[npoints-2:npoints])
		model = sm.OLS(endog=y, exog=x)
		res = model.fit()
		x_predict = np.log10(select['Cumulative production'].values[npoints-1:npoints+1])
		x_predict = sm.add_constant(x_predict)
		predictions = 10**(res.predict(x_predict))
		plt.plot(select['Cumulative production'].values[npoints-1:npoints+1], predictions, 'g:')
	
	legend_elements = [
		matplotlib.lines.Line2D([0],[0], lw=0, marker='o', color='red', label='Obs'),
		matplotlib.lines.Line2D([0],[0], lw=1, linestyle='--', color='k', label='Exp. curve estimate'),
		matplotlib.lines.Line2D([0],[0], lw=1, linestyle='--', color='r', label='Exp. curve estimate from last point'),
		matplotlib.lines.Line2D([0],[0], lw=1, linestyle='--', color='g', label='Exp. curve estimated from last two points'),
	]
	plt.legend(handles=legend_elements)
	plt.xscale('log', base=10)
	plt.yscale('log', base=10)
	plt.ylabel('Unit cost')
	plt.xlabel('Cumulative production')
	plt.title(select['Tech'].values[0])
	plt.show()

plt.show()


		

		