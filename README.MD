# Cognitive load detection

This repository contains the code to reproduce the ML analysis in 

Tervonen, Pettersson, Mäntyjärvi. Ultra-short window length and feature importance analysis for cognitive load detection from wearable sensors. Submitted to Electronics.

Please cite the paper above if you use any part of this code in a publication.

The code was developed on Xubuntu 20.04 LTS and Python 3.6.9. The required packages are listed in `requirements.txt`, you can install them with `pip install -r requirements.txt`.

The dataset used is the CogLoad dataset available [here](https://github.com/MartinGjoreski/martingjoreski.github.io/blob/master/files/CogDatasets.rar). To use default paths, place the data of each individual user under `./data/train/`

Usage: 
To extract features run `python data.py` and to run Bayesian hyperparameter optimization for XGBoost model, run `python xgb_hyperopt.py`.

