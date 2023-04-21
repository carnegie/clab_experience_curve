import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

# read data and create class
df = pd.read_csv('ExpCurves.csv')

# for each technology
for tech in df['Tech'].unique():
    # get the data and process them
    sel = df.loc[df['Tech']==tech].copy().reset_index()
    x = sel['Cumulative production'].values
    y = sel['Unit cost'].values
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('intercept', mu=0, sigma=10)
        beta = pm.Normal('slope', mu=0, sigma=10)
        sigma = pm.HalfNormal('error', sigma=10)
        pred = pm.Data('pred', np.log10(x[:-3]), mutable = True)
                
        obs = pm.Normal('obs', alpha + beta * pred,
                       sigma , shape = pred.shape, observed=np.log10(y[:-3]),)

        # Run the inference algorithm
        idata = pm.sample(1000, tune=2000, random_seed = 0, return_inferencedata=True)

    with model:
        pm.set_data({'pred': np.log10(x[-3:])})
        idata = pm.sample_posterior_predictive(
            idata, var_names = ['obs'], 
            return_inferencedata=True,
            predictions=True,
            extend_inferencedata=True,random_seed=0)
        model_preds = idata.predictions
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    ax.scatter(x[-3:], 10**(model_preds['obs'].mean(('chain','draw'))))
    ax.plot([x[-3:],x[-3:]],10**az.hdi(model_preds, hdi_prob=0.9)['obs'].transpose('hdi',...))
    ax.set_xscale('log', base = 10)
    ax.set_yscale('log', base = 10)
    az.plot_trace(idata)
    plt.show()

