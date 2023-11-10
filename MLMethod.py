import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib, analysisFunctions, plottingFunctions
import scipy, sklearn
from sklearn import tree


matplotlib.rc('font', 
              **{'family':'sans-serif','sans-serif':'Helvetica'})
matplotlib.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('ExpCurves.csv')

# print('Computing error dataset...')
# errorDataset = analysisFunctions.buildErrorDataset(df)
# print('Computing weights...')
# for tech in errorDataset['Tech'].unique():
#     errorDataset.loc[errorDataset['Tech']==tech,'Weight'] = 1/errorDataset['Tech'].nunique()/errorDataset.loc[errorDataset['Tech']==tech].count()[0]
# print('Saving file...')
# errorDataset.to_csv('ErrorDataset.csv', index=False)
# print('Done!')

errorDataset = pd.read_csv('ErrorDataset.csv')
errorDataset['Training horizon'] = 10**errorDataset['Training horizon']
errorDataset['Forecast horizon'] = 10**errorDataset['Forecast horizon']

ohcs = pd.get_dummies(errorDataset['Sector'], dtype='float64')

errorDataset = errorDataset.join(ohcs)

X = errorDataset[['Forecast horizon','Training horizon','F-test p-value','Chemicals','Consumer goods','Energy','Food','Genomics','Hardware']].copy()
X['Forecast-to-Training ratio'] = X['Forecast horizon']/X['Training horizon']
X = X[['Forecast-to-Training ratio','F-test p-value','Chemicals','Consumer goods','Energy','Food','Genomics','Hardware']].copy()
# X = X[['Forecast-to-Training ratio','F-test p-value']].copy()
# X = X[['Forecast-to-Training ratio']].copy()
# X = X[['Forecast horizon','Training horizon','F-test p-value']]
Y = errorDataset['Error'].copy()

reg = tree.DecisionTreeRegressor()
reg = reg.fit(X,Y, sample_weight=errorDataset['Weight'].values)
tree.plot_tree(reg, filled=True, fontsize=8, feature_names=X.columns,
               max_depth=3, proportion=True)

print('Feature importance for error prediction: ')
print(reg.feature_importances_)

X = errorDataset[['Forecast horizon','Training horizon','F-test p-value','Chemicals','Consumer goods','Energy','Food','Genomics','Hardware']].copy()
X['Forecast-to-Training ratio'] = X['Forecast horizon']/X['Training horizon']
# X = X[['Forecast-to-Training ratio','F-test p-value','Chemicals','Consumer goods','Energy','Food','Genomics','Hardware']].copy()
# X = X[['Forecast-to-Training ratio','F-test p-value']].copy()
# X = X[['Forecast-to-Training ratio']].copy()
X = X[['Forecast horizon','Training horizon','F-test p-value']]
Y = errorDataset['Best'].copy()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y, sample_weight=errorDataset['Weight'].values, )
plt.figure()
tree.plot_tree(clf, filled=True, fontsize=8, 
               feature_names=X.columns, class_names=clf.classes_,
               max_depth=3)

print('Feature importance for best method prediction: ')
print(clf.feature_importances_)

plt.show()