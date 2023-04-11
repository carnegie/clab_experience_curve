import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt

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
	ax.step(f[f.columns[2]], f[f.columns[0]],'o-',  where='post', markersize=2, alpha=0.5)
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
	select = df.loc[df['Tech']=='tech']
	for npoints in range(2,len(df['Unit cost'].values)):
		print(df['Unit cost'].values[:npoints])
		print(df['Unit cost'].values)
		exit()