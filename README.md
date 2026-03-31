# MicroGrowth and PioGrowth

Repository for Streamlit apps based on growthcurves package.

The MicroGrowth app is designed for analysis of optical density (OD) measurements.

The PioGrowth app is designed for analysis of optical density (OD) measurements
produced mainly with a [PioReactor](https://pioreactor.com/)

The code translated to Python was based on the original R code from the Shiny apps.

## MicroGrowth: Install and start app

In a new environment you can install the app using the app's
[`requirements.txt`](MicroGrowth/requirements.txt)

```bash
pip install -r MicroGrowth/requirements.txt
```

Start the app from the root of the repository (as it's done on Streamlit Cloud):

```bash
streamlit run MicroGrowth/app.py
```

## PioGrowth: Install and start app

In a new environment you can install the app using

```bash
pip install .
```

Start the app from the root of the repository (as it's done on Streamlit Cloud):

```bash
streamlit run PioGrowth/main.py
```

## Development environment for shared ui code

Install package so that new code is picked up in a restared python interpreter:

```bash
pip install -e ".[dev]"
```

## History

Growthcurves, PioGrowth and the MicroGrowth app were developed in parallel. The 
growthcurves package severes as the backend for both apps, and the apps were developed 
and aligned to feature similar design.

The joint app combining three Shiny apps for PioReactor tools was started as the
three individual apps mentioned above using shiny, but never completed. You can find the
last development version in the original repository of this fork
(see: [milnus/PioGrowth](https://github.com/milnus/PioGrowth)).

It was created on the basis of three individual apps

- [milnus/pioreactor_turbidostat_shiny](https://github.com/milnus/pioreactor_turbidostat_shiny)
- [milnus/Batch_analysis](https://github.com/milnus/Batch_analysis)
- [milnus/OD_calibration_bioreactor](https://github.com/milnus/OD_calibration_bioreactor)

Then it was moved from [biosustain/PioGrowth](https://github.com/biosustain/PioGrowth) 
to this repository, to start sharing functionality with similary designed MicroGrowth 
app.

The original code for the MicroGrowth app was developed in the repository
[sambra95/TheGrowthAnalysisApp](https://github.com/sambra95/TheGrowthAnalysisApp)
