import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# set figures' parameters
sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300
sns.color_palette('colorblind')
palette = 'crest'

# set to true to plot a figure for each technology
plot_fig_tech = False

# set to True if the regression dataset needs to be built
build_pwreg_dataset = False

# use half of the data for each technology
half_data = False

# use half of the technologies in the dataset
half_techs = False

# set the maximum number of breakpoints
max_breakpoints = 6

# set min distance between breakpoints
min_dist = np.log10(2)

# set random seed
np.random.seed(0)


# Load the data
df = pd.read_csv('ExpCurves.csv')


for data, techs, remove_sector in zip([True, False, True], 
                                     [False, True, False], 
                                     [None, None, 'Chemicals']):

    half_data = data
    half_techs = techs
    output_file = ("IC_half_data"*(half_data) 
                   + "_half_techs"*(half_techs))
    if remove_sector is not None:
        output_file += "_no" + remove_sector
    output_file += ".csv"

    
    # perform continuous piecewise regression 
    # on each technology in the dataset
    # and return a dataframe containing
    # the parameters of each fit
    # and the information criteria
    IC = utils.build_piecewise_regression_dataset(df,
                                                    max_breakpoints,
                                                    min_dist,
                                                    plot_fig_tech,
                                                    half_data,
                                                    half_techs,
                                                    remove_sector,
                                                    output_file)


