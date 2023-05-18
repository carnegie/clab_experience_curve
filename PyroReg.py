import pyro
from pyro.nn import PyroModule
from torch import nn
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt 

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)
pyro.set_rng_seed(1)
df = pd.read_csv('ExpCurves.csv')

class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

for tech in df['Tech'].unique():
    sel = df[df['Tech'] == tech].copy()
    # data = torch.tensor(sel[['Cumulative production','Unit cost']].values, dtype=torch.float)
    x = np.log10(sel['Cumulative production'].values).reshape(-1,1)
    y = np.log10(sel['Unit cost'].values).reshape(-1,1) 

    # x_data, y_data = data[:,:-1], data[:,-1]
    x_data = torch.tensor(x, dtype=torch.float)
    y_data = torch.tensor(y, dtype=torch.float)
    linear_reg_model = PyroModule[nn.Linear](1,1)
    model = LinearRegressionModel()
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.SGD(linear_reg_model.parameters(), lr = 0.0001)
    num_iterations = 1500
    loss_list = []
    for i in range(num_iterations):
        y_pred = linear_reg_model(x_data).squeeze(-1)
        loss = loss_fn(y_pred, y_data)
        optim.zero_grad()
        loss.backward()
        loss_list.append(loss.item())
        optim.step()
        # print(i, ' : ', loss.item())
        # print([x for x in linear_reg_model.parameters()])

    print("Learned parameters:")
    for name, param in linear_reg_model.named_parameters():
        print(name, param.data.numpy())
    plt.plot(loss_list)
    plt.figure()
    plt.scatter(10**x, 10**y)
    plt.scatter(10**x, 10**(linear_reg_model.bias.item() + 
                            linear_reg_model.weight.item() * x))
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    for i in range(num_iterations):
        optimizer.zero_grad()
        y_hat = model(x_data)
        loss = criterion(y_hat, y_data)
        loss.backward()
        optimizer.step()
        # print(i, ' : ', loss.item())
    plt.figure()
    plt.scatter(10**x, 10**y)
    plt.scatter(10**x, 10**(model.linear.bias.data[0] + 
                            model.linear.weight.data[0] * x))
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)

    plt.show()



    
