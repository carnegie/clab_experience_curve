import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib, os, analysisFunctions, seaborn

matplotlib.rc('savefig', dpi=300)


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

	cols = f.columns.tolist()
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
	

	fig, ax = plt.subplots()
	ax.plot(f['Cumulative production'],f['Unit cost'], marker='o', lw=0)
	ax.set_xscale('log', base=10)
	ax.set_yscale('log', base=10)
	ax.set_ylabel(cols[0])
	ax.set_xlabel(cols[3])
	fig.savefig('figs/TechFigures'+os.path.sep+file[:-4]+'.pdf')
	model = sm.OLS(np.log10(f['Unit cost']), sm.add_constant(np.log10(f['Cumulative production'])))
	results = model.fit()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	ax.plot(f['Cumulative production'], 
		 10**(results.predict(
			 sm.add_constant(
				 np.log10(f['Cumulative production'])))), 'k')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	fig.savefig('figs/TechFigures'+os.path.sep+file[:-4]+'_fit.pdf')
	plt.close(fig)	

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
# df = df.loc[~(df['Tech'].str.contains('Nuclear'))].copy()
selTechs = ['Wind_Turbine_2_(Germany)', 'Fotovoltaica', 'Photovoltaics_2',
 'Onshore_Gas_Pipeline', 'Titanium_Sponge', 'Wind_Electricity', 'Milk_(US)',
 'Transistor', 'Primary_Aluminum', 'Photovoltaics_4', 'PolyesterFiber',
 'Geothermal_Electricity', 'Solar_Thermal', 'DRAM', 'Ethanol_2',
 'Monochrome_Television', 'Polyvinylchloride', 'PolyethyleneLD',
 'Solar_Thermal_Electricity', 'Polypropylene', 'Laser_Diode',
 'Electric_Power', 'Shotgun_Sanger_DNA_Sequencing',
 'Capillary_DNA_Sequencing', 'Paraxylene', 'SCGT', 'Automotive_(US)',
 'Photovoltaics', 'Solar_Water_Heaters', 'Wind_Turbine_(Denmark)',
 'Corn_(US)', 'PolyethyleneHD', 'Hard_Disk_Drive', 'Low_Density_Polyethylene',
 'Primary_Magnesium', 'Ethylene' ,'Wheat_(US)', 'Offshore_Gas_Pipeline',
 'Ethanol_(Brazil)', 'Wind_Power', 'Polystyrene', 'Beer_(Japan)']
selTechs = ['Wind_Turbine_2_(Germany)', 'Fotovoltaica', 'Photovoltaics_2',
 'Titanium_Sponge', 'Wind_Electricity', 'Transistor', 'Photovoltaics_4',
 'DRAM', 'Ethanol_2', 'Monochrome_Television', 'Laser_Diode',
 'Capillary_DNA_Sequencing', 'Photovoltaics', 'Solar_Water_Heaters',
 'Wind_Turbine_(Denmark)', 'Hard_Disk_Drive', 'Primary_Magnesium',
 'Wheat_(US)', 'Wind_Power', 'Polystyrene']
# df = df.loc[df['Tech'].isin(selTechs)]

# plot normalized cost and cumulative production by sector
fig, ax = plt.subplots(2,1, sharex=True, 
		       height_ratios=[1,0.4],
			   figsize = (10/2,8/2))
last = []
alpha=0.4
for tech in df['Tech'].unique():
	s = df.loc[df['Tech']==tech].copy()
	# if tech in selTechs:
	# 	color = 'C4'
	# 	alpha=0.8
	# 	zorder = 0
	# else:
	# 	color = 'C9'
	# 	alpha = 0.8
	# 	zorder = -s['Normalized cumulative production'].values[-1]
	ax[0].plot(s['Normalized cumulative production'], 
	 s['Normalized unit cost'],
	#  color=analysisFunctions.sectorsColor[s['Sector'].values[0]],
	#  color=color,
	#  alpha=alpha,
	 marker = 'o',
	 markersize=0.5,
	 lw=0.5,
	#  zorder=zorder
	 )
	last.append(s['Normalized cumulative production'].values[-1])
last.sort()
# create total technologies availabilities
avail = [[1,df['Tech'].nunique()]]
for x in last:
	avail.append([x, avail[-1][1] - 1])
avail = np.array(avail).transpose()
# create sectoral availabilities
avails = {}
# for sector in df['Sector'].unique():
# 	avails[sector] = [
# 		[1, df.loc[df['Sector']==sector, 'Tech'].nunique()]
# 					]
# for x in last:
# 	s = df.loc[
# 		df['Normalized cumulative production']==x,'Sector'].values[0]
# 	for ss in df['Sector'].unique():
# 		if s == ss:
# 			avails[ss].append([x, avails[ss][-1][1] - 1])
# 		else:
# 			avails[ss].append([x, avails[ss][-1][1]])
avails['sel'] = [[1, len(selTechs)]]
for x in last:
	s = df.loc[
		df['Normalized cumulative production']==x,'Tech'].values[0]
	if s in selTechs:
		avails['sel'].append([x, avails['sel'][-1][1] - 1])
	else:
		avails['sel'].append([x, avails['sel'][-1][1]])
avails['nonsel'] = [[1, df['Tech'].nunique() - len(selTechs)]]
for x in last:
	s = df.loc[
		df['Normalized cumulative production']==x,'Tech'].values[0]
	if s in selTechs:
		avails['nonsel'].append([x, avails['nonsel'][-1][1]])
	else:
		avails['nonsel'].append([x, avails['nonsel'][-1][1] - 1])			
availsy = [
	[avails[s][x][1] for x in range(len(avails[s])) ] 
	for s in ['sel','nonsel'] ]
# ax[1].stackplot(avail[0],availsy, colors = ['C4','C9'])
# ax[1].stackplot(avail[0],availsy, colors = [analysisFunctions.sectorsColor[s] for s in df['Sector'].unique()])
ax[1].plot(avail[0],avail[1], 'k', lw=1)
ax[0].set_xscale('log', base=10)
ax[0].set_yscale('log', base=10)
ax[1].set_xlabel('Cumulative production / Initial cumulative production')
ax[0].set_ylabel('Unit cost / Initial unit cost')
ax[1].set_ylabel('Technologies available')
# print(ax[1].get_yticks())
# print(ax[1].get_yticks().tolist())
# print(avail[1][0])
# ax[1].set_yticks(ax[1].get_yticks().tolist().append(avail[1][0]))
# ax[1].annotate('Selected technologies (20) are required to have at least one point that:'+\
# 	       '\n\t - is preceded by points for at least one order of magnitude of cumulative production'+\
# 			'\n\t - is followed by points for at least one order of magnitude or cumulative production',
# 			xy=(200,30),
# 			xycoords='data',
# 			ha='left',
# 			va='bottom',
# 			color='C4')
# ax[1].annotate('Selected technologies',
# 			xy=(10**6,10),
# 			xycoords='data',
# 			ha='center',
# 			va='center',
# 			color='C4',
# 			fontsize=11
# 			)
# ax[0].annotate('Selected technologies: \n'+\
# 				'cumulative production range covers\n'+\
# 				' at least two orders of magnitude',
# 			xy=(10**8,0.5),
# 			xycoords='data',
# 			ha='center',
# 			va='center',
# 			color='C4',
# 			fontsize=11
# 			)
# legend_elements = [
# 	matplotlib.lines.Line2D([0],[0],lw=1, 
# 			color = analysisFunctions.sectorsColor[sector],
# 			label = sector)
# 			for sector in df['Sector'].unique() ]
# fig.legend(handles=legend_elements, title='Sector', loc='center right')
# fig.subplots_adjust(right=0.8, top=0.95, bottom=0.1, hspace=0.05)
fig.subplots_adjust(right=0.9, left=0.15, top=0.95, bottom=0.15, hspace=0.05)
plt.show()

cmap = seaborn.color_palette("colorblind")
sectorsColor = {'Energy': cmap[0], 'Chemicals': cmap[1],
               'Hardware': cmap[2], 'Consumer goods': cmap[3],
               'Food': cmap[4], 'Genomics': cmap[8]}

plt.figure(figsize=(9,7.5))
for tech in df['Tech'].unique():
	plt.plot(df.loc[df['Tech']==tech,'Cumulative production'],
			df.loc[df['Tech']==tech,'Unit cost'],
			alpha=0.75,
			lw=0.5,
			marker='o',
			markersize=0.5,
			color=sectorsColor[df.loc[df['Tech']==tech,'Sector'].values[0]])
plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel('Cumulative production')
plt.ylabel('Unit cost')
legend_elements = [matplotlib.lines.Line2D([0],[0],lw=0.5, marker='o', 
			 color=sectorsColor['Energy'], label='Energy',
			 markersize=2.5),
			 matplotlib.lines.Line2D([0],[0],lw=0.5, marker='o',
			 color=sectorsColor['Chemicals'], label='Chemicals',
			 markersize=2.5),
			 matplotlib.lines.Line2D([0],[0],lw=0.5, marker='o',
			 color=sectorsColor['Hardware'], label='Hardware',
			 markersize=2.5),
			 matplotlib.lines.Line2D([0],[0],lw=0.5, marker='o',
			 color=sectorsColor['Consumer goods'], label='Consumer goods',
			 markersize=2.5),
			 matplotlib.lines.Line2D([0],[0],lw=0.5, marker='o',
			 color=sectorsColor['Food'], label='Food',
			 markersize=2.5),
			 matplotlib.lines.Line2D([0],[0],lw=0.5, marker='o',
			 color=sectorsColor['Genomics'], label='Genomics',
			 markersize=2.5),]
plt.gcf().legend(handles=legend_elements, title='Sectors', ncol=2, loc='lower center')
plt.subplots_adjust(bottom=0.2)
plt.figure(figsize=(9,7.5))
for tech in df['Tech'].unique():
	plt.plot(df.loc[df['Tech']==tech,'Year'],
			df.loc[df['Tech']==tech,'Unit cost'],
			alpha=0.75,
			lw=0.5,
			marker='o',
			markersize=0.5,
			color=sectorsColor[df.loc[df['Tech']==tech,'Sector'].values[0]])
# plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel('Cumulative production')
plt.ylabel('Unit cost')
plt.gcf().legend(handles=legend_elements, title='Sectors', ncol=2, loc='lower center')
plt.subplots_adjust(bottom=0.2)
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


		

		