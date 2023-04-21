import pandas as pd
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel, Matern, ExpSineSquared
import matplotlib.pyplot as plt
import numpy as np

# read data and create class
df = pd.read_csv('ExpCurves.csv')

# kernels to be used for gaussian processes
kernel =  RBF() + WhiteKernel()

# for each technology
for tech in df['Tech'].unique():
    # get the data and process them
    sel = df.loc[df['Tech']==tech].copy().reset_index()
    X = sel['Cumulative production'].values.reshape(-1,1)
    y = sel['Unit cost'].values.reshape(-1,1)
    # adding one data point at each step
    fig, ax = plt.subplots()
    ax.scatter(sel['Cumulative production'], sel['Unit cost'], edgecolor="k",
            facecolor = "None", marker='o')
    for nobs in range(2,len(y)):
        # select training data
        X_train = X[:nobs]
        y_train = y[:nobs]
        # plot dataset
        # calibrate predictors
        gpr = GaussianProcessRegressor(kernel = kernel,
                    random_state=0).fit(np.log10(X_train),
                                        np.log10(y_train))
        reg = linear_model.BayesianRidge().fit(np.log10(X_train),
                                         np.log10(y_train))
        reg2 = linear_model.ARDRegression().fit(np.log10(X_train),
                                         np.log10(y_train))
        # select input for testing
        X_test = X[nobs]
        # predict and plot with uncertainty bounds
        mu, sigma = gpr.predict(np.log10(X_test), return_std = True)
        gpr_mu.append(mu)
        gpr_sigma.append(sigma)
        # ax.scatter(X_test, 10**mu, marker='P',
        #             color = 'g', alpha = 0.85, label='Gaussian Process')
        # for n in range(len(X_test)):
        #     ax.plot([X_test[n], X_test[n]] ,
        #             [10**(mu[n]-3*sigma[n]), 10**(mu[n]+3*sigma[n])],
        #             'g:', alpha = 0.85)
        # predict and plot with uncertainty bounds
        mu, sigma = reg.predict(np.log10(X_test), return_std = True)
        reg_mu.append(mu)
        reg_sigma.append(sigma)
        # ax.scatter(X_test, 10**mu, marker='D',
        #             color = 'm', alpha = 0.5, label='Bayesian Ridge')
        # for n in range(len(X_test)):
        #     ax.plot([X_test[n], X_test[n]] ,
        #             [10**(mu[n]-3*sigma[n]), 10**(mu[n]+3*sigma[n])],
        #             'm--', alpha = 0.5)
        # mu, sigma = reg2.predict(np.log10(X_test), return_std = True)
        # ax.scatter(X_test, 10**mu, marker='D',
        #             color = 'r', alpha = 0.5, label='ADR Reg')
        # for n in range(len(X_test)):
        #     ax.plot([X_test[n], X_test[n]] ,
        #             [10**(mu[n]-3*sigma[n]), 10**(mu[n]+3*sigma[n])],
        #             'r--', alpha = 0.5)
        # highlight the point that was to be predicted next
        # ax.scatter(X[nobs], y[nobs],marker='*', color = 'k', s = 100)
    ax.set_xscale('log',base=10)
    ax.set_yscale('log',base=10)
    ax.set_ylabel('Unit cost')
    ax.set_xlabel('Cumulative production')
    ax.set_title(sel['Tech'].values[0][:-4])
    # fig.legend(loc='lower center', ncol = 2)
    plt.subplots_adjust(bottom=0.2)
    plt.show()
