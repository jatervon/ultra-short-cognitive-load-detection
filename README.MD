# Cognitive load detection

This repository contains the code to reproduce the ML analysis in 

```
@Article{electronics10050613,
AUTHOR = {Tervonen, Jaakko and Pettersson, Kati and Mäntyjärvi, Jani},
TITLE = {Ultra-Short Window Length and Feature Importance Analysis for Cognitive Load Detection from Wearable Sensors},
JOURNAL = {Electronics},
VOLUME = {10},
YEAR = {2021},
NUMBER = {5},
ARTICLE-NUMBER = {613},
URL = {https://www.mdpi.com/2079-9292/10/5/613},
ISSN = {2079-9292},
DOI = {10.3390/electronics10050613}
}
```

Please cite the paper above if you use any part of this code in a publication.

The code was developed on Xubuntu 20.04 LTS and Python 3.6.9. The required packages are listed in `requirements.txt`, you can install them with `pip install -r requirements.txt`.

The dataset used is the CogLoad dataset available [here](https://github.com/MartinGjoreski/martingjoreski.github.io/blob/master/files/CogDatasets.rar). To use default paths, place the data of each individual user under `./data/train/`

Usage: 

To extract features run `python data.py` and to run Bayesian hyperparameter optimization for XGBoost model, run `python xgb_hyperopt.py`.

