import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

df = pd.read_csv('ExpCurves.csv')

method = 'regression'
method = 'slope'

# get slope for all technologies
slopes = []
for tech in df['Tech'].unique():
	sel = df.loc[df['Tech']==tech]
	x = np.log10(sel['Cumulative production'].values)
	y = np.log10(sel['Unit cost'].values)
	model = sm.OLS(y, sm.add_constant(x))
	result = model.fit()
	slopes.append([tech, result.params[1]])
slopes = pd.DataFrame(slopes, columns=['Tech', 'Slope'])

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
				model = sm.OLS(y[:i+1], sm.add_constant(x[:i+1]))
				result = model.fit()
				slope = result.params[1]
			# compute error associated using slope M points after midpoint
			for M in range(i+1, H):
				pred =  y[i] + slope * (x[M] - x[i])
				if method=='regression':
					pred = result.params[0] + slope * x[M]
				pred2 =  y[i] + slopeall * (x[M] - x[i])
				error = (y[M] - (pred)) 
				error2 = (y[M] - (pred2)) 
				dferr.append([x[i] - x[N],
								x[M] - x[i],
								error, tech])
				dferr2.append([x[i] - x[N],
								x[M] - x[i],
								error2, tech])
				if np.abs(error2) < np.abs(error):
					counterr += 1
				# if 'Offshore_Gas_Pi' in tech and i==7 and M==8 and N ==0:
				# 	print(tech, i, N, M)
				# 	print(x[:i], y[:i], pred, pred2, y[M])
				# 	print(x[i], x[N], y[i], y[N], slope, x[M], x[i])
				# 	exit()

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

# select ratios to be plotted
frac = []
for x in range(17):
	frac.append(0.001 * (10**(0.25*x)) )

mean1 = np.empty((len(frac)-1,len(frac)-1))
mean2 = np.empty((len(frac)-1,len(frac)-1))
median1 = np.empty((len(frac)-1,len(frac)-1))
median2 = np.empty((len(frac)-1,len(frac)-1))
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
			mean1[i-1, j-1] = 0.0
			mean2[i-1, j-1] = 0.0
			meandiff[i-1,j-1] = 0.0
			fracavg[i-1,j-1] = 0.0
			count[i-1,j-1] = 0.0
		else:
			mean1[i-1,j-1] = np.mean(select1['Error'].values**2)**0.5
			mean2[i-1,j-1] = np.mean(select2['Error'].values**2)**0.5
			# select1['Weights'] = [1/select1.loc[select1['Tech']==t,'Tech'].count() for t in select1['Tech'].values]
			# select2['Weights'] = [1/select2.loc[select2['Tech']==t,'Tech'].count() for t in select2['Tech'].values]
			# mean1[i-1, j-1] = np.average(select1['Error']**2, weights=select1['Weights'])**0.5
			# mean2[i-1, j-1] = np.average(select2['Error']**2, weights=select2['Weights'])**0.5
			mean1[i-1,j-1] = 0.0
			mean2[i-1,j-1] = 0.0
			ntechs = select1['Tech'].nunique()
			for tech in select1['Tech'].unique():
				sel1 = select1.loc[select1['Tech']==tech].copy()
				sel2 = select2.loc[select2['Tech']==tech].copy()
				mean1[i-1,j-1] += 1/(ntechs * sel1.count()[0]) * np.mean(sel1['Error'].values**2)**0.5
				mean2[i-1,j-1] += 1/(ntechs * sel2.count()[0]) * np.mean(sel2['Error'].values**2)**0.5
			meandiff[i-1,j-1] = mean2[i-1,j-1] - mean1[i-1,j-1]
			fracavg[i-1,j-1] = np.sum(select2['Error'].values**2 < 
									select1['Error'].values**2)/\
									select2['Error'].count() * 100
			count[i-1,j-1] = (select1['Error'].count())
			counttech[i-1,j-1] = (select1['Tech'].nunique())

print(mean2)

plt.figure()
mean = mean1[::-1,:]
im = plt.imshow(mean, aspect='auto', norm='log')
plt.gca().set_xticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('RMSE')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)
plt.suptitle('Technology-specific slope')

plt.figure()
mean = mean2[::-1,:]
im = plt.imshow(mean, aspect='auto', norm='log')
plt.gca().set_xticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('RMSE')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)
plt.suptitle('Average technology slope')

plt.figure()
mean = meandiff[::-1,:]
mean[mean<0] = -1
mean[mean>0] = 1
divnorm = matplotlib.colors.TwoSlopeNorm(vcenter=0)
im = plt.imshow(mean, aspect='auto', 
		norm=divnorm, 
		cmap='RdBu_r')
plt.gca().set_xticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('RMSE difference')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)
plt.suptitle('Average technology - Technology-specific')

plt.figure()
perc = fracavg[::-1,:]
divnorm = matplotlib.colors.TwoSlopeNorm(vcenter=50)
im = plt.imshow(perc, aspect='auto', norm=divnorm, cmap='RdBu')
plt.gca().set_xticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(16)], 
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
plt.gca().set_xticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('Bin size')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)

prob = 100*(perc > 50) * count/np.sum(count)
prob2 = -100*(perc < 50) * count/np.sum(count)
prob[prob==0] = prob2[prob==0]

plt.figure()
divnorm = matplotlib.colors.TwoSlopeNorm(vcenter=0)
im = plt.imshow(prob, aspect='auto', norm=divnorm, cmap='RdBu')
plt.gca().set_xticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])],
        rotation = 90)
plt.gca().set_yticks([x for x in range(16)], 
        [str(round(x,3))+' to '+str(round(y,3)) for x, y in zip(frac[:-1], frac[1:])][::-1])
plt.ylabel('Log of cumulative production ratios for prediction')
plt.xlabel('Log of cumulative production ratios for predictor')
cbar = plt.colorbar(im)
cbar.set_label('Porbability improvement of more accuracy with average technological slope')
plt.subplots_adjust(bottom=0.3, left=0.2, right=0.95, top=0.9)

plt.show()
