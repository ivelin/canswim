# canswim
Toolkit for CANSLIM investment style practitioners


# Setup

## Install canswim package in dev mode

```
pip install -e ./
```

## Explore jupyter notebooks

  * [prepare_data.ipynb](prepare_data.ipynb) - Download raw data from third party sources and save to local storage.

  * [model_sandbox.ipynb](model_sandbox.ipynb) - Prepare timeseries, find optimal model hyper parameters, train and backtest model

## Command line interface

```
$ python -m canswim -h
usage: canswim [-h] {dashboard,train}

CANSWIM is a toolkit for CANSLIM style investors. Humbly complements the Simple Moving Average and other technical indicators.

positional arguments:
  {dashboard,train}  Which canswim module to run: `dashboard` for stock scans and charting of uploaded forecasts or `train` for continuous training and forecast uploads

options:
  -h, --help         show this help message and exit

NOTE: NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.
```