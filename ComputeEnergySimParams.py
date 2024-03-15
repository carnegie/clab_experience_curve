import pandas as pd
import numpy as np
import statsmodels.api as sm
import analysisFunctions

# read data
df = pd.read_csv('ExpCurves.csv')

# create empty list to store slopes
slopes = []

# loop over all techs
for t in df['Tech'].unique():

    # get cumulative production and unit cost series
    x, y = np.log10(\
        df.loc[df['Tech'] == t,'Cumulative production'].values), \
        np.log10(\
            df.loc[df['Tech'] == t,'Unit cost'].values)
    
    # build linear regression model
    # and fit it to all data available
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()

    # store slope, sector, and technology
    slopes.append([result.params[1], result.bse[1],
                   analysisFunctions.sectorsinv[t], 
                   t])

# create dataframe from list
slopes = pd.DataFrame(slopes, columns=['Slope','Standard Error', 'Sector','Tech'])

# store information in list
expCurveParams = [['All techs', slopes['Slope'].mean(), 
      slopes['Standard Error'].mean()]]


# compute average slope and its standard error by sector
smean = slopes.groupby(['Sector']).mean(numeric_only=True)
print(smean.loc['Energy']['Slope'])
# store information in list
expCurveParams.append(\
    ['Energy sector', 
     smean.loc['Energy']['Slope'], 
      smean.loc['Energy']['Standard Error']])

# compute average slope and its standard error by sector
# after excluding nuclear technologies
slopes = slopes.loc[~(slopes['Tech'].str.contains('Nuclear'))]
smean = slopes.groupby(['Sector']).mean(numeric_only=True)

# store information in list
expCurveParams.append(\
    ['Energy sector without nuclear',
     smean.loc['Energy']['Slope'], 
      smean.loc['Energy']['Standard Error']])

# compute standard deviation of errors
# when using the average learning rates 
# obtained above
for p in expCurveParams:

    # create empty list to store errors
    errors = []

    # iterate over all technologies
    for t in df['Tech'].unique():

        # if only energy technologies are to be selected
        if 'Energy' in p[0] and \
            analysisFunctions.sectorsinv[t] != 'Energy':
            continue
        # if nuclear has to be excluded
        if 'Nuclear' in p[0] and 'Nuclear' in t:
            continue

        # get cumulative production and unit cost series
        x, y = np.log10(\
            df.loc[\
                df['Tech'] == t,'Cumulative production'].values), \
            np.log10(\
                df.loc[df['Tech'] == t,'Unit cost'].values)

        # iterate over all points and store 
        # error obtained using the average learning rate
        for i in range(len(x)-1):
            errors.append([y[i+1] - \
                          (y[i] + p[1] * (x[i+1] - x[i])),
                          t])

    # create dataframe from list
    errors = pd.DataFrame(errors, columns=['Error','Tech'])

    # create empty lists to store 
    # mean and variance for each technology
    m, v = [], []
    for t in errors['Tech'].unique():
        m.append(errors.loc[errors['Tech']==t,'Error'].mean())
        v.append(errors.loc[errors['Tech']==t,'Error'].var())

    # append standard deviation of errors to expCurveParams
    expCurveParams[expCurveParams.index(p)]\
        .append(np.sqrt(np.mean(v)))


expCurveParams = pd.DataFrame(expCurveParams,
            columns=['Aggregation','Mean slope', 
                     'Standard error of mean slope',
                     'Standard deviation of errors'])

print('Parameters used to estimate the '
       'cost of the energy transition:')
print(expCurveParams)

expCurveParams.to_csv('./energySim/ExpCurveParams.csv', 
                      index=False)
