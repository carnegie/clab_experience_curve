import pandas as pd
import os, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# prepare figure for plots
fig, ax = plt.subplots()
# folder name
datafolder = 'expCurveData'
# for all files in folder
for file in os.listdir(datafolder):
	# read file and drop nan
	f = pd.read_csv(datafolder+os.path.sep+file)
	f = f.dropna()
	# compute cumulative sum of units produce and normalize price and cumulative units produced
	f[f.columns[2]] = f[f.columns[2]].cumsum()
	f[f.columns[2]] = f[f.columns[2]] / f[f.columns[2]].values[0]
	f[f.columns[0]] = f[f.columns[0]] / f[f.columns[0]].values[0]
	# plot as step functions
	ax.step(f[f.columns[2]], f[f.columns[0]],'o-', where='post', markersize=2, alpha=0.5)
	# save unit cost, cumulative production, and name of technology
	f['Tech'] = file
	f = f[[f.columns[0], f.columns[2], 'Tech']].copy()
	f.columns = ['Unit cost','Cumulative production','Tech']
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

# save dataframe
df.to_csv('ExpCurves.csv', index=False)

# figure reporting the number of available technologies for number of units produced
fig, ax = plt.subplots()
nunits =[1]
for tech in df['Tech'].unique():
	nunits.append(df.loc[df['Tech']==tech, df.columns[1]].max())
nunits.sort()
ax.step(nunits,[x for x in range(len(nunits),0,-1)], where='post')
ax.set_xscale('log', base=10)
ax.set_ylabel('Number of technologies available')
ax.set_xlabel('Normalized cumulative production')
plt.show()

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


		

		