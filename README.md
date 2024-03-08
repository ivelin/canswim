# canswim
Developer toolkit for CANSLIM investment style practitioners


# Setup


```
pip install canswim
```


## Install canswim package in dev mode

```
pip install -e ./
```

## Command line interface

```
$ python -m canswim -h
usage: canswim [-h] {dashboard,train}

CANSWIM is a toolkit for CANSLIM style investors. Aims to complement the Simple Moving Average and other technical indicators.

positional arguments:
  {dashboard,gatherdata,uploaddata,modelsearch,train,forecast}
                        Which canswim task to run:
                        `dashboard` - start Web App service with stock charting and forecast scans.
                        'gatherdata` to gather 3rd party stock market data and save to HF Hub.
                        'uploaddata` upload to HF Hub any interim changes to local data storage.
                        `modelsearch` to find and save optimal hyperparameters for model training.
                        `train` for continuous model training.

options:
  -h, --help            show this help message and exit
  --forecast_start_date FORECAST_START_DATE
                        Optional argument for the `forecast` task. Indicate forecast start date in YYYY-MM-DD format. If not specified, forecast will start from the end of the target series.

NOTE: NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.
```
