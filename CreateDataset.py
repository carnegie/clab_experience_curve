import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import matplotlib, os, analysisFunctions

matplotlib.rc('savefig', dpi=300)
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_palette('colorblind')
matplotlib.rc('font',**{'family':'sans-serif',
			'sans-serif':['Helvetica']})

## set to True to overwrite existing files
saveData = False

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
			'Tech']].copy()	

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
	if not os.path.exists('figs' + os.path.sep + 'TechFigures'):
		os.makedirs('figs' + os.path.sep + 'TechFigures')
	fig.savefig('figs' + os.path.sep + 'TechFigures' + \
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
	fig.savefig('figs' + os.path.sep + 'TechFigures' + \
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
		  'Unit cost']].copy()

if saveData:
	df_.to_csv('ExpCurves.csv', index=False)

# save normalized dataframe
df_ = df[['Tech',
			'Normalized cumulative production',
	  		'Year',
			'Normalized unit cost']].copy()
if saveData:
	df_.to_csv('NormalizedExpCurves.csv', index=False)

df['Sector'] = [\
	analysisFunctions.sectorsinv[tech] for tech in df['Tech']]

### plot normalized cost and cumulative production by sector

# create figure
fig, ax = plt.subplots(2, 1, sharex=True, 
		       height_ratios=[1,0.5],
			   figsize=(10,8))

# create list to store the range of cumulative production
# covered by each of the technologies
last = []

# iterate over all technologies
for tech in df['Tech'].unique():

	# select data for each technology
	s = df.loc[df['Tech']==tech].copy()

	# plot normalized unit cost vs
	# normalized cumulative production
	ax[0].plot(s['Normalized cumulative production'], 
		s['Normalized unit cost'],
		marker = '.',
		markersize=5,
		alpha=0.5
		)
	
	# append length of cumulative production range
	last.append(\
		s['Normalized cumulative production'].values[-1])

# sort list of cumulative production ranges
last.sort()

# create array contanining the number of technologies
# available at each order of magnitude of cumulative production
avail = [[1,df['Tech'].nunique()]]
for x in last:
	avail.append([x, avail[-1][1] - 1])
avail = np.array(avail).transpose()

# plot available technologies vs cumulative production range
ax[1].step(avail[0], avail[1], 
		   where='post', color='k', 
		   lw=2)

# set axes scales and labels
ax[0].set_xscale('log', base=10)
ax[0].set_yscale('log', base=10)
ax[0].set_xlim(1, 1.2e10)
ax[0].set_ylim(1e-8, 1e2)
ax[1].set_xlabel(
	'Cumulative production relative to initial')
ax[0].set_ylabel('Unit cost relative to initial')
ax[1].set_ylabel('Technologies available')
ax[0].annotate('a', (0.05, 1.05),
			   xycoords='axes fraction', 
			   ha='center', va='center')
ax[1].annotate('b', (0.05, 1.05),
			   xycoords='axes fraction', 
			   ha='center', va='center')

fig.subplots_adjust(right=0.9, left=0.15, 
					top=0.95, bottom=0.1, 
					hspace=0.2)

fig.savefig('figs' + os.path.sep + 'Data.png')

### Below some figures for the 
### Supplememntary Material are produced

### plot normalized cost vs cumulative production, 
### color by sector

# define sector colors
cmap = sns.color_palette("colorblind")
sectorsColor = {'Energy': cmap[0], 'Chemicals': cmap[1],
               'Hardware': cmap[2], 'Consumer goods': cmap[3],
               'Food': cmap[4], 'Genomics': cmap[8]}

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
			color=sectorsColor[\
				df.loc[df['Tech']==tech,'Sector'].values[0]])

# set axes scales and labels
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
ax.set_xlabel('Cumulative production')
ax.set_ylabel('Unit cost')

# define legend handles and labels
legend_elements = [\
				matplotlib.lines.Line2D([0], [0], 
					marker='o', 
					color=sectorsColor['Energy'], 
					label='Energy',
					markersize=5),
				matplotlib.lines.Line2D([0], [0], 
					marker='o',
					color=sectorsColor['Chemicals'], 
					label='Chemicals',
					markersize=5),
				matplotlib.lines.Line2D([0], [0], 
					marker='o',
					color=sectorsColor['Hardware'], 
					label='Hardware',
					markersize=5),
				matplotlib.lines.Line2D([0], [0], 
					marker='o',
					color=sectorsColor['Consumer goods'], 
					label='Consumer goods',
					markersize=5),
				matplotlib.lines.Line2D([0], [0],
					marker='o',
					color=sectorsColor['Food'], 
					label='Food',
					markersize=5),
				matplotlib.lines.Line2D([0], [0],
					marker='o',
					color=sectorsColor['Genomics'], 
					label='Genomics',
					markersize=5),
				]

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
			color=sectorsColor[\
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

plt.show()
