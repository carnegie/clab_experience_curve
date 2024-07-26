import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import matplotlib, os, utils
import seaborn.objects as so

plt.rcParams['savefig.dpi'] = 300
sns.set_style('ticks')
sns.set_context('talk')
plt.rcParams['font.sans-serif'] = 'Helvetica'

## set to True to overwrite existing files
save_data = True

initial_correction = True

# folder name
datafolder = 'expCurveData'

# for all files in folder
for file in os.listdir(datafolder):

	# read file and drop nan
	f = pd.read_csv(datafolder+os.path.sep+file)
	f = f.dropna()

	# save original columns and redefine columns
	cols = f.columns
	f.columns = ['Unit cost',
				'Year',
				'Production',
				'Cumulative production']	
	
	# perform correction for initial cumulative production
	# following procedure described in lafond et al., 2018 
	if initial_correction:
		g_d = np.exp(\
			np.log(\
					(f['Production'].values[-1] \
						- f['Production'].values[0]))/\
					(f['Year'].values[-1] - f['Year'].values[0])) - 1
	if ~np.isnan(g_d) and g_d > 0:
		cumprod = np.array([f['Production'].values[0]/g_d])
		for i in range(f['Production'].shape[0] - 1):
			cumprod = np.append(cumprod, cumprod[-1] + f['Production'].values[i])
		f['Cumulative production'] = cumprod

	# compute cumulative sum of units produce 
	# and normalize price and cumulative units produced
	f['Normalized cumulative production'] = \
		f['Cumulative production'] /\
		  f['Cumulative production'].values[0]
	f['Normalized unit cost'] = \
		f['Unit cost'] /\
		  f['Unit cost'].values[0]
	
	# save unit cost, year, cumulative production,
	# normalized unit cost, normzlied cumulative production, 
	# and name of technology
	f['Tech'] = file[:-4]
	f = f[['Unit cost', 
			'Year', 
			'Cumulative production',
			'Normalized unit cost', 
			'Normalized cumulative production',
			'Tech']]

	# plot individual technology data 
	fig, ax = plt.subplots()
	ax.plot(f['Cumulative production'], 
			f['Unit cost'], 
			marker='o', lw=0)
	
	# set log scale and axes labels
	ax.set_xscale('log', base=10)
	ax.set_yscale('log', base=10)
	ax.set_ylabel(cols[0])
	ax.set_xlabel(cols[3])

	#save figure
	fig.tight_layout()
	if not os.path.exists('figs' + os.path.sep + 'SupplementaryFigures'):
		os.makedirs('figs' + os.path.sep + 'SupplementaryFigures')
	if not os.path.exists('figs' + os.path.sep + 
					   	'SupplementaryFigures' + os.path.sep + 'TechFigures'):
		os.makedirs('figs' + os.path.sep + 
			  		'SupplementaryFigures' + os.path.sep + 'TechFigures')
	fig.savefig('figs' + os.path.sep + 
			'SupplementaryFigures' + os.path.sep + 'TechFigures' + \
			os.path.sep + file[:-4] + '.pdf')

	# store position of axes limits
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	# add regression line to figure
	model = sm.OLS(np.log10(f['Unit cost']), 
				sm.add_constant(\
					np.log10(f['Cumulative production'])))
	results = model.fit()
	ax.plot(f['Cumulative production'], 
		 10**(results.predict(
			 sm.add_constant(
				 np.log10(f['Cumulative production'])))), 'k')

	# set position of axes limits
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	#save figure
	fig.savefig('figs' + os.path.sep + 
			'SupplementaryFigures' + os.path.sep + 'TechFigures' + \
			os.path.sep + file[:-4] + '_fit.pdf')
	plt.close(fig)	

	# append to dataframe
	try:
		df = pd.concat([df, f])
	except NameError:
		df = f

print('Saving dataframes to csv files...')

# save dataframe
df_ = df[['Tech',
		  'Cumulative production',
		  'Year',
		  'Unit cost',
		  'Normalized cumulative production',
		  'Normalized unit cost']]

if save_data:
	df_.to_csv('ExpCurves.csv', index=False)

df['Sector'] = [\
	utils.sectorsinv[tech] for tech in df['Tech']]

### plot normalized cost and cumulative production by sector

fig, ax = plt.subplots(1, 1, 
			   figsize=(8,8))

ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)


# iterate over all technologies
for tech in df['Tech'].unique():

	# select data for each technology
	s = df.loc[df['Tech']==tech].copy()

	# plot normalized unit cost vs
	# normalized cumulative production
	ax.plot(s['Normalized cumulative production'], 
		s['Normalized unit cost'],
		marker = '.',
		markersize=5,
		color = utils.sectors_colors[s['Sector'].values[0]],
		alpha=0.5,
		label = s['Sector'].values[0]
		)

	if tech in ['Solar_Water_Heaters','Fotovoltaica',
			 'DRAM','Transistor', 'Wind_Electricity',
			 'Nuclear_Electricity','Laser_Diode',
			 'Hard_Disk_Drive']:
		if tech == 'Hard_Disk_Drive':
			ha = 'right'
			va = 'top'
		else:
			ha = 'left'
			va = 'bottom'
		if tech == 'Fotovoltaica':
			tech = 'Solar PV Electricity'
		ax.annotate(tech.replace('_',' '),
						xy=(s['Normalized cumulative production'].values[-1],
						s['Normalized unit cost'].values[-1]),
						textcoords='data',
						ha=ha,
						va=va,
						fontsize=14,
						)
	
# set axes scales and labels
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
ax.set_xlim(1, 8e10)
ax.set_ylim(0.3*1e-8, 1e2)
ax.set_xlabel(
	'Cumulative production relative to initial')
ax.set_ylabel('Unit cost relative to initial')

# define legend handles and labels
legend_elements = [\
				matplotlib.lines.Line2D([0], [0], 
					marker='o', 
					color=utils.sectors_colors['Energy'], 
					label='Energy',
					markersize=5),
				matplotlib.lines.Line2D([0], [0], 
					marker='o',
					color=utils.sectors_colors['Chemicals'], 
					label='Chemicals',
					markersize=5),
				matplotlib.lines.Line2D([0], [0], 
					marker='o',
					color=utils.sectors_colors['Hardware'], 
					label='Hardware',
					markersize=5),
				matplotlib.lines.Line2D([0], [0], 
					marker='o',
					color=utils.sectors_colors['Consumer goods'], 
					label='Consumer goods',
					markersize=5),
				matplotlib.lines.Line2D([0], [0],
					marker='o',
					color=utils.sectors_colors['Food'], 
					label='Food',
					markersize=5),
				matplotlib.lines.Line2D([0], [0],
					marker='o',
					color=utils.sectors_colors['Genomics'], 
					label='Genomics',
					markersize=5),
				]

ax.legend(handles=legend_elements,
		   loc='upper right', ncol=1, title='Sectors')

plt.tight_layout()
fig.savefig('figs' + os.path.sep + 'Data.png')
fig.savefig('figs' + os.path.sep + 'Data.eps')

### Below some figures for the 
### Supplememntary Material are produced

### plot normalized cost vs cumulative production, 
### color by sector

# create figure
fig, ax = plt.subplots(figsize=(9,7.5))

# iterate over all technologies
for tech in df['Tech'].unique():

	# plot normalized unit cost vs
	# normalized cumulative production
	# color by sector
	ax.plot(df.loc[df['Tech']==tech,'Cumulative production'],
			df.loc[df['Tech']==tech,'Unit cost'],
			alpha=0.5,
			marker='.',
			markersize=5,
			color=utils.sectors_colors[\
				df.loc[df['Tech']==tech,'Sector'].values[0]])

# set axes scales and labels
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
ax.set_xlabel('Cumulative production')
ax.set_ylabel('Unit cost')

# add legend
fig.legend(handles=legend_elements, 
		   title='Sectors', 
		   ncol=2, 
		   loc='lower center')

plt.subplots_adjust(bottom=0.3, top=0.975)
fig.savefig('figs' + os.path.sep + 
			'SupplementaryFigures' + 
			 os.path.sep + 'data_raw.png')

### plot normalized cost and year
### color by sector

# create figure
fig, ax = plt.subplots(figsize=(9,7.5))

# iterate over all technologies
for tech in df['Tech'].unique():

	# plot normalized unit cost vs
	# year, color by sector
	plt.plot(df.loc[df['Tech']==tech,'Year'],
			df.loc[df['Tech']==tech,'Unit cost'],
			alpha=0.5,
			marker='.',
			markersize=5,
			color=utils.sectors_colors[\
				df.loc[df['Tech']==tech,'Sector'].values[0]])

# set axes scales and labels
ax.set_yscale('log', base=10)
ax.set_xlabel('Year')
ax.set_ylabel('Unit cost')

# add legend
fig.legend(handles=legend_elements, 
		   title='Sectors', 
		   ncol=2, loc='lower center')
plt.subplots_adjust(bottom=0.3, top=0.975)
fig.savefig('figs' + os.path.sep + 
		'SupplementaryFigures' + 
			os.path.sep + 'data_time.png')

## plot technologies available against time
dfy = df.copy()

for tech in dfy['Tech'].unique():
	dfy.loc[dfy['Tech']==tech, 'Year'] = \
		dfy.loc[dfy['Tech']==tech, 'Year'] - \
		dfy.loc[dfy['Tech']==tech, 'Year'].values[0]

dfy['Year'] = dfy['Year'].astype(int)

y_vs_sec = []
for y in dfy['Year'].unique():
	dfy_ = dfy.loc[dfy['Year']>=y].copy()
	for s in dfy['Sector'].unique():
		y_vs_sec.append([y, s, dfy_.loc[dfy_['Sector']==s,'Tech'].nunique()])
y_vs_sec = pd.DataFrame(y_vs_sec, columns=['Year', 'Sector', 'Count'])

fig, ax = plt.subplots(figsize=(9,7.5))
ax.stackplot(y_vs_sec['Year'].unique(),
			 [[y_vs_sec.loc[(y_vs_sec['Year']==y) & \
							(y_vs_sec['Sector']==s),'Count'].values[0] \
								for y in y_vs_sec['Year'].unique()]
								for s in y_vs_sec['Sector'].unique()],
			 colors=[utils.sectors_colors[s] \
					for s in y_vs_sec['Sector'].unique()],
					labels=[s for s in y_vs_sec['Sector'].unique()])
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Number of technologies')

## plot technologies available against cumulative production increase
dfp = df.copy()

maxp = []
for tech in dfp['Tech'].unique():
	dfp.loc[dfp['Tech']==tech, 'Cumulative production'] = \
		dfp.loc[dfp['Tech']==tech, 'Cumulative production'] / \
		dfp.loc[dfp['Tech']==tech, 'Cumulative production'].values[0]
	maxp.append(dfp.loc[dfp['Tech']==tech, 'Cumulative production'].values[-1])

maxp.sort()
maxp.insert(0, 1)

p_vs_sec = []
for p in maxp:
	dfp_ = dfp.loc[dfp['Cumulative production']>=p].copy()
	for s in dfp['Sector'].unique():
		p_vs_sec.append([p, s, dfp_.loc[dfp_['Sector']==s,'Tech'].nunique()])
pVsSec = pd.DataFrame(p_vs_sec, columns=['Cumulative production increase', 
									   'Sector', 'Count'])

fig, ax = plt.subplots(figsize=(9,7.5))
ax.stackplot(p_vs_sec['Cumulative production increase'].unique(),
			 [[p_vs_sec.loc[(p_vs_sec['Cumulative production increase']==p) & \
							(p_vs_sec['Sector']==s),'Count'].values[0] \
								for p in p_vs_sec['Cumulative production increase'].unique()]
								for s in p_vs_sec['Sector'].unique()],
			 colors=[utils.sectors_colors[s] \
					for s in p_vs_sec['Sector'].unique()],
					labels=[s for s in p_vs_sec['Sector'].unique()])
ax.set_xscale('log')
ax.legend()
ax.set_xlabel('Cumulative production - multiplicative increase')
ax.set_ylabel('Number of technologies')

plt.show()
