# Overview

This repository reports the scripts used to obtain the results and figures reported in the manuscript "Learning rate variability and implications for technology forecasts".


# Clone the repository and create the environment

To clone the repository, use the following command in the terminal from the folder where you want to clone the repository:

~~~
git clone https://github.com/carnegie/clab_experience_curve.git
~~~

To create the python environment, use the following commands in the terminal replacing <<PATHTOENV>> with the name of the folder where you want to place your environment:

```bash
python -m venv <<PATHTOENV>>
source <<PATHTOENV>>/bin/activate
pip install -r requirements.txt
```

# Replicating figures

Run the scripts:
- `CreateDataset.py` (Fig. 1a, S1-2)
- `LearningRateAnalysis.py` (Fig. 1b)
- `LearningRateDynamicsAllTechs.py` (Fig. 2)
- `LearningRateErrorAggregate.py` (Fig. S6)
- `PiecewiseRegression.py` (Fig. 3a, S7)
- `AnalyzePiecewise.py` (Fig. S3, S4, S5, Tables S1, S2)
- `PiecewiseRobustness.py` (Fig. S8, S9)
- `AnalyzePiecewiseValidation.py` (Fig. 3b)
- `SolarProjection.py` (Fig. 4a)
- `WindProjection.py` (Fig. 4b)
- `BatteryProjection.py` (Fig. 4c)

# Data summary table
To obtain the summary table run the script:
- `TechInfo.py` (Table 1)

# Additional data
To run all the scripts additional data is required from previous publications:
- https://ars.els-cdn.com/content/image/1-s2.0-S2589004222006496-mmc2.zip
    + (whose content is expected to be placed inside a folder named BolingerEtAl2022 inside the AdditionalData folder)
- https://doi.org/10.7910/DVN/9FEJ7C
    + (whose content is expected to be placed in a folder named ZieglerTrancik inside the AdditionalData folder)

# Material

The folder `expCurveData` contains the files for 87 techhnologies as downloaded from the Santa Fe Institute [Performance Curve Database](https://pcdb.santafe.edu/). These are the technologies for which data without missing values is available.

The script `CreateDataset.py` reads in the CSV files for each technology, prepares the data in two CSV files (`ExpCurves.csv` and `NormalizedExpCurves.csv`) and produces Figs. 1a, S1, and S2.

The script `LearningRateAnalysis.py` examines the variability of learning rates dividing each technological data series in two parts with equal data points. This script produces Fig.1b.

The script `LearningRateDynamicsAllTechs.py` is used to examine the learning rate variability for solar PV, wind, and lithium-ion batteries among other technologies. This script produce Fig.2 as well as Figures S7-12.

The script `LearningRateErrorAggregate.py` examines the distance between learning rate estimated from all the data and learning rate at a certain time or cumulative production. This produces Fig. S6.

The piecewise regression fits are computed in the script `PiecewiseRegression.py` using a function inside the scipt `utils.py`. These two scripts produce Figs. 3a, S7.

The script `AnalyzePiecewise.py` examines the results of piecewise regression to fit probability distributions to data and use them in the forecasting model. This script produces Figs. S3, S4, S5. 

The script `PiecewiseRobustness.py` performs standard non-parametric boostraping to evaluate the stability of breakpoint detection across the technologies in the Performance Curve Database. This script produces the figures S8, S9.

The script `ComparePiecewiseLafondValidation.py`is used to compare the Continuous Ranked Probability Forecast for using half of the data to calibrate models and the remaining half of data for each data series for validation. This scripts uses results obtained from `BuildValidationDataset.py`, analyzed further in `AnalyzePiecewiseValidation.py`.

The scripts `SolarProjection.py`, `WindProjection.py`, `BatteryProjection.py` build piecewise regressions for solar photovoltaics, wind power, and lithium-ion batteries. They use the parameters estimated from the Performance Curve Database to produce estimates of future cost until 2050 for the three technologies examined. These scripts are used to produce Fig.4.

The scripts `utils.py` contains functions used to analyze data and produce figures.

The script `IntroGifs.py` can be used to generate gifs showing how learning rate changed over time by observing how observed and future learning rate evolved over time.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10823145.svg)](https://doi.org/10.5281/zenodo.10823145)
