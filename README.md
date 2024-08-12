# Overview

This repository reports the scripts used to obtain the results and figures reported in the manuscript "Learning rate variability and implications for technology forecasts".


# Clone the repository and create the environment

To clone the repository, use the following command in the terminal from the folder where you want to clone the repository:

~~~
git clone https://github.com/carnegie/clab_experience_curve.git
~~~

To install the required packages, we recommend using [conda](https://www.anaconda.com/). A list of the required packages is available in the YAML file (expCurveEnv.yml).

To create a virtual environment with the required packages using conda, use the following command in the terminal from the folder where the file is located:

~~~
conda env create -f expCurveEnv.yml
~~~

# Replicating figures

Run the scripts:
- `CreateDataset.py` (Fig. 1a, S1-2)
- `LRanalysis.py` (Fig. 1b)
- `LearningRateDyamicsAllTechs.py` (Fig. 2, S7-S12)
- `LearningRateErrorAggregate.py` (Fig. S3)
- `PiecewiseRegression.py` (Fig. 3a, S4)
- `AnalyzePiecewise.py` (Fig. 3b, S5-6)
- `SolarProjection.py` (Fig. 4a)
- `WindProjection.py` (Fig. 4b)
- `BatteryProjection.py` (Fig. 4c)


# Material

The folder `expCurveData` contains the files for 87 techhnologies as downloaded from the Santa Fe Institute [Performance Curve Database](https://pcdb.santafe.edu/). These are the technologies for which data without missing values is available.

The script `CreateDataset.py` reads in the CSV files for each technology, prepares the data in two CSV files (`ExpCurves.csv` and `NormalizedExpCurves.csv`) and produces Figs. 1a, S1, and S2.

The script `LearningRateAnalysis.py` examines the variability of learning rates dividing each technological data series in two parts with equal data points. This script produces Fig.1b.

The script `LearningRateDynamicsAllTechs.py` is used to examine the learning rate variability for solar PV, wind, and lithium-ion batteries among other technologies. This script produce Fig.2 as well as Figures S7-12.

The script `LearningRateErrorAggregate.py` examines the distance between learning rate estimated from all the data and learning rate at a certain time or cumulative production. This produces Fig. S3.

The piecewise regression fits are computed in the script `PiecewiseRegression.py` suing a function inside the scipt `utils.py`. These two scripts produce Figs. 3a, S4.

The script `AnalyzePiecewise.py` examines the results of piecewise regression to fit probability distributions to data and use them in the forecasting model. This script produces Figs. 3b, S5-6.

The scripts `SolarProjection.py`, `WindProjection.py`, `BatteryProjection.py` build piecewise regressions for solar photovoltaics, wind power, and lithium-ion batteries. They use the parameters estimated from the Performance Curve Database to produce estimates of future cost until 2050 for the three technologies examined. These scripts are used to produce Fig.4.

The scripts `utils.py` contains functions used to analyze data and produce figures.

The script `IntroGifs.py` can be used to generate gifs showing how learning rate changed over time by observing how observed and future learning rate evolved over time.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13308841.svg)](https://doi.org/10.5281/zenodo.13308841)
