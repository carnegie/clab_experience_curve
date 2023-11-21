import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from pyro.nn import PyroSample
import pyro 

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

# read data and create class
df = pd.read_csv('ExpCurves.csv')

# for each technology
for tech in df['Tech'].unique():
    # get the data and process them
    sel = df.loc[df['Tech']==tech].copy().reset_index()
    x = sel['Cumulative production'].values
    y = sel['Unit cost'].values
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    for nobs in range(2,len(y)):
        with pm.Model() as model:
            # Priors
            alpha = pm.Uniform('intercept', -10,10)
            # alpha = pm.Normal('intercept', mu=0, sigma=1)
            # sigma_slopes = pm.HalfNormal('error_rw', sigma = 10)
            # slope = pm.GaussianRandomWalk('rw', mu = 0, sigma=sigma_slopes,
            #                              shape = nobs-1)
            # beta = pm.Normal('beta', mu=0 , sigma=1)
            beta = pm.Uniform('beta', -1,1)
            sigma = pm.Uniform('error', -1, 1)
            # sigma = pm.HalfNormal('error', sigma=1)
            pred = pm.Data('pred', np.log10(x[:nobs]), mutable = True)
                    
            obs = pm.Normal('obs', alpha + beta * pred,
                        sigma , shape = pred.shape, observed=np.log10(y[:nobs]),)
            
            # Run the inference algorithm
            idata = pm.sample(1000, tune=1000, random_seed = 0, return_inferencedata=True)

        with model:
            pm.set_data({'pred': [np.log10(x[nobs])]})
            idata = pm.sample_posterior_predictive(
                idata, var_names = ['obs'], 
                return_inferencedata=True,
                predictions=True,
                extend_inferencedata=True,random_seed=0)
            model_preds = idata.predictions
        ax.scatter(x[nobs], 10**(model_preds['obs'].mean(('chain','draw'))),
                   marker = 'P', color = 'r')
        ax.plot([x[nobs],x[nobs]],
                10**az.hdi(model_preds, hdi_prob=0.96)['obs'].transpose('hdi',...),
                color = 'r')
        az.plot_trace(idata)

        with pm.Model() as model2:
            X_ = np.log10(x[:nobs]).reshape(-1,1)
            y_ = np.log10(y[:nobs])
            # Priors
            ls = pm.HalfNormal('ls', sigma = 10)
            cov_func = pm.gp.cov.ExpQuad(1, ls = ls)
            gp = pm.gp.Marginal(cov_func=cov_func)
            sigma = pm.HalfNormal('sigma', sigma=10)
                    
            obs = gp.marginal_likelihood('obs', X=X_, y=y_, sigma=sigma)
            # Run the inference algorithm
            mp = pm.find_MAP()
            # idata = pm.sample()
            # idata = pm.sample(1000, tune=1000, random_seed = 0, return_inferencedata=True)
        with model2:
            X_new = np.log10(x[nobs]).reshape(1,-1)
            # y_pred = gp.conditional('y_pred', Xnew = X_new, pred_noise=True)
            # y_samples = pm.sample_posterior_predictive([mp], vars=[y_pred], samples=2000)
            mu, var = gp.predict(X_new, point=mp, diag=True)
        ax.scatter(10**X_new[-1], 10**mu[-1], color='g')
        ax.plot([10**X_new[-1],10**X_new[-1]],
                [10**(mu[-1]-3*np.sqrt(var[-1])),10**(mu[-1]+3*np.sqrt(var[-1]))],
                alpha = 0.4, color='g')
    ax.set_xscale('log', base = 10)
    ax.set_yscale('log', base = 10)
    plt.show()

