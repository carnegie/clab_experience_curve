import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib, os, analysisFunctions, seaborn

seaborn.set_palette('colorblind')
matplotlib.rc('font',**{'family':'sans-serif',
			'sans-serif':['Helvetica']})

# folder name
datafolder = 'expCurveData'

# for all files in folder
for file in os.listdir(datafolder):

	# read file and drop nan
	f = pd.read_csv(datafolder+os.path.sep+file)
	f = f.dropna()
	
	# compute cumulative sum of units produce 
	# and normalize price and cumulative units produced
	f['Normalized cumulative production'] = f[f.columns[3]] /\
		  f[f.columns[3]].values[0]
	f['Normalized unit cost'] = f[f.columns[0]] /\
		  f[f.columns[0]].values[0]
	
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

print('Saving dataframes to csv files...')

# save dataframe
df_ = df[['Tech','Cumulative production','Year','Unit cost']].copy()
# df_.to_csv('ExpCurves.csv', index=False)

# save normalized dataframe
df_ = df[['Tech','Normalized cumulative production',
	  'Year','Normalized unit cost']].copy()
# df_.to_csv('NormalizedExpCurves.csv', index=False)

df['Sector'] = [analysisFunctions.sectorsinv[tech] for tech in df['Tech']]

# uncomment for SM figure
df = df.loc[~(df['Tech'].str.contains('Nuclear'))].copy()

# plot normalized cost and cumulative production by sector
fig, ax = plt.subplots(2,1, sharex=True, 
		       height_ratios=[1,0.2],
			   figsize = (9,8))
last = []
for tech in df['Tech'].unique():
	s = df.loc[df['Tech']==tech].copy()
	ax[0].plot(s['Normalized cumulative production'], 
	 s['Normalized unit cost'],
	#  color=analysisFunctions.sectorsColor[s['Sector'].values[0]],
	 color='k',
	 alpha=0.7,
	 marker = 'o',
	 markersize=0.5,
	 lw=0.25,
	 zorder=-s['Normalized cumulative production'].values[-1])
	last.append(s['Normalized cumulative production'].values[-1])
last.sort()
# create total technologies availabilities
avail = [[1,df['Tech'].nunique()]]
for x in last:
	avail.append([x, avail[-1][1] - 1])
avail = np.array(avail).transpose()
# create sectoral availabilities
avails = {}
for sector in df['Sector'].unique():
	avails[sector] = [
		[1, df.loc[df['Sector']==sector, 'Tech'].nunique()]
					]
for x in last:
	s = df.loc[
		df['Normalized cumulative production']==x,'Sector'].values[0]
	for ss in df['Sector'].unique():
		if s == ss:
			avails[ss].append([x, avails[ss][-1][1] - 1])
		else:
			avails[ss].append([x, avails[ss][-1][1]])
availsy = [
	[avails[s][x][1] for x in range(len(avails[s])) ] 
	for s in df['Sector'].unique() ]
# ax[1].stackplot(avail[0],availsy, colors = [analysisFunctions.sectorsColor[s] for s in df['Sector'].unique()])
ax[1].plot(avail[0],avail[1], 'k', lw=1)
ax[0].set_xscale('log', base=10)
ax[0].set_yscale('log', base=10)
ax[1].set_xlabel('Normalized cumulative production')
ax[0].set_ylabel('Normalized unit cost')
ax[1].set_ylabel('Technologies available')

# legend_elements = [
# 	matplotlib.lines.Line2D([0],[0],lw=1, 
# 			color = analysisFunctions.sectorsColor[sector],
# 			label = sector)
# 			for sector in df['Sector'].unique() ]
# fig.legend(handles=legend_elements, title='Sector', loc='center right')
# fig.subplots_adjust(right=0.8, top=0.95, bottom=0.1, hspace=0.05)
fig.subplots_adjust(right=0.9, left=0.1, top=0.95, bottom=0.1, hspace=0.05)

plt.show()

exit()

### not relevant from now on

print('Running simple regression for prediction examples ...')
# predict for each technology using only available data 
f_values = [] # here the p values for the f test for each single regression are stored
slopes = []
for tech in df['Tech'].unique():
	select = df.loc[df['Tech']==tech].copy()
	plt.figure()
	plt.scatter(select['Normalized cumulative production'],
	      select['Normalized unit cost'], color='r', marker='o')
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

# plt.show()


		

		