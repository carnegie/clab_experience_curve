# Overview

This repository reports the scripts used to obtain the results and figures reported in the manuscript "The predictive skill of technological experience curves".




# Clone the repository and create the environment

To clone the repository, use the following command in the terminal from the folder where you want to clone the repository:

~~~
git clone github.com/carnegie/clab_experience_curve
~~~

To install the required packages, we recommend using [conda](https://www.anaconda.com/). A list of the required packages is available in the YAML file (expCurveEnv.yml).

To create a virtual environment with the required packages using conda, use the following command in the terminal from the folder where the file is located:

~~~
conda env create -f expCurveEnv.yml
~~~

# Material

The folder `expCurveData` contains the files for 86 techhnologies as downloaded from the Santa Fe Institute [Performance Curve Database](https://pcdb.santafe.edu/). These are the technologies for which data without missing values is available.

The script `CreateDataset.py` reads in the CSV files for each technology, prepares the data in two CSV files (`ExpCurves.csv` and `NormalizedExpCurves.csv`) and produces figures 1, S1, and S2.

The script `IntroGifs.py` can be used to generate gifs showing how learning rate changed over time by observing how observed and future learning rate evolved over time.

The script `ExplainedVariance.py` examined the fraction of variance of future learning rates that can be explained based on observed learning rates. It produces figures 2 and S3.

The script `CostErrors.py` compares the magnitude of errors made assuming that all technologies decline with the same learning rate with the standard used of experience curves, i.e., assuming that each technology has a specific learning rate. This script produces figures 3, S4, S5.

The script `ErrorsStatisticalSignificance.py` examines whether the error differences between the two methods discussed above are significant. This script produces CSV tables with p-values for paired t test and Wilcoxon signed-ranks test which are stored in the folder `StatisticalTests` and figure S6.

The script `ComputeEnergySimParams.py` is used to derive alternative assumptions for computing the cost of technologies that follow experience curves whose values are stored in the folder `energySim` in the file `ExpCurveParams.csv`. 

These alternatives parameters assumptions are used in the script `EnergyTransitionCosts.py` to evaluate the importance of these assumptions in determining the cost of the energy transition. This script produces figures 4 and 5.

The scripts `analysisFunctions.py` and `plottingFunctions.py` contain functions used to analyze data and produce figures.

Fhe folder `energySim` contains the script `EnergySim.py` used to simulate the energy system transition costs as in *"Empirically grounded technology forecast and the costs of the energy transition"*, Way et al., Joule (2022). The parameters used in the model together with the alternative cost assumptions are reported in the script `EnergySimParams.py`. The script `checkEnergySim.py` simulates the 5 scenarios used in the previously cited paper and reproduces figures coherently with that papers' supplementary material.









