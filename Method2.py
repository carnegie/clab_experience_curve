import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

df = pd.read_csv('ExpCurves.csv')

# get slope for all technologies
x = np.log10(df['Cumulative production'].values)
y = np.log10(df['Unit cost'].values)
model = sm.OLS(y, sm.add_constant(x))
result = model.fit()
slopeall = result.params[1]

# get error using technology specific slope
dferr = []
dferr2 = []
counterr = 0
for tech in df['Tech'].unique():
	sel = df.loc[df['Tech']==tech]
	x = np.log10(sel['Cumulative production'].values)
	y = np.log10(sel['Unit cost'].values)
	H = len(x)
	i = round(H/2) - 1
    # select N points before midpoint and compute slope
	for i in range(H):
		for N in range(i-1, -1, -1):
			slope = (y[i] - y[N]) /\
				(x[i] - x[N])
			# compute error associated using slope M points after midpoint
			for M in range(i+1, H):
				pred =  y[i] + slope * (x[M] - x[i])
				pred2 =  y[i] + slopeall * (x[M] - x[i])
				error = (y[M] - (pred)) 
				error2 = (y[M] - (pred2)) 
				dferr.append([x[i] - x[N],
								x[M] - x[i],
								error])
				dferr2.append([x[i] - x[N],
								x[M] - x[i],
								error2])
				if np.abs(error2) < np.abs(error):
					counterr += 1

print('Percentage of cases where error', 
      ' is lower with average technological slope: ',
      100*counterr/len(dferr), '%')

dferr = pd.DataFrame(dferr, 
                     columns = ['Log of ratios for predictor',
                                'Log of ratios for prediction',
                                'Error'])
dferr2 = pd.DataFrame(dferr2, 
                     columns = ['Log of ratios for predictor',
                                'Log of ratios for prediction',
                                'Error'])

# select ratios to be plotted
# frac = np.log10([1, 2, 3, 5, 10, 1e2, 1e3, 1e9])
frac = []
for x in range(17):
	frac.append(0.001 * (10**(0.25*x)) )

mean1 = np.empty((len(frac)-1,len(frac)-1))
mean2 = np.empty((len(frac)-1,len(frac)-1))
meandiff = np.empty((len(frac)-1,len(frac)-1))
fracavg = np.empty((len(frac)-1,len(frac)-1))
count = np.empty((len(frac)-1,len(frac)-1))

for i in range(1, len(frac)):
    for j in range(1, len(frac)):
        select1 = dferr.loc[
            (dferr['Log of ratios for predictor']>frac[i-1]) &\
            (dferr['Log of ratios for predictor']<=frac[i]) & \
            (dferr['Log of ratios for prediction']>frac[j-1]) &\
            (dferr['Log of ratios for prediction']<=frac[j])]
        select2 = dferr2.loc[
            (dferr2['Log of ratios for predictor']>frac[i-1]) &\
            (dferr2['Log of ratios for predictor']<=frac[i]) & \
            (dferr2['Log of ratios for prediction']>frac[j-1]) &\
            (dferr2['Log of ratios for prediction']<=frac[j])]
        mean1[i-1,j-1] = np.mean(select1['Error'].values**2)**0.5
        mean2[i-1,j-1] = np.mean(select2['Error'].values**2)**0.5
        # mean1[i-1,j-1] = np.std(select1['Error'].values) 
        # mean2[i-1,j-1] = np.std(select2['Error'].values) 
        meandiff[i-1,j-1] = mean2[i-1,j-1] - mean1[i-1,j-1]
        fracavg[i-1,j-1] = np.sum(select2['Error'].values**2 < 
                                select1['Error'].values**2)/\
                                select2['Error'].count() * 100
        count[i-1,j-1] = (select1['Error'].count())

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
divnorm = matplotlib.colors.TwoSlopeNorm(vcenter=0)
im = plt.imshow(mean, aspect='auto', norm=divnorm, cmap='RdBu_r')
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
