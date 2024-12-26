# canswim
Developer toolkit for CANSLIM investment style practitioners

For a brief introduction read [this blog post](https://medium.com/@ivelin.atanasoff.ivanov/canswim-a-deep-learning-tool-for-canslim-practitioners-2c9740bb0d3d).

[Here is also a video recording](https://www.youtube.com/watch?v=GfC-H0uxXvk&ab_channel=AustinPythonMeetup) of a CANSWIM presentation for the [Python Austin Meetup](https://www.meetup.com/austinpython/).

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
$  python -m canswim -h
usage: canswim [-h] [--forecast_start_date FORECAST_START_DATE] [--new_model NEW_MODEL] [--same_data SAME_DATA] {dashboard,gatherdata,downloaddata,uploaddata,modelsearch,train,forecast}

CANSWIM is a toolkit for CANSLIM style investors. Aims to complement the Simple Moving Average and other technical indicators.

positional arguments:
  {dashboard,gatherdata,downloaddata,uploaddata,modelsearch,train,forecast}
                        Which canswim task to run: `dashboard` for stock charting and scans of recorded forecasts. 'gatherdata` to gather 3rd party stock market data and save to HF Hub. 'downloaddata` download model training and forecast
                        data from HF Hub to local data storage. 'uploaddata` upload to HF Hub any interim changes to local train and forecast data. `modelsearch` to find and save optimal hyperparameters for model training. `train` for
                        continuous model training. `forecast` to run forecast on stocks and upload dataset to HF Hub.

options:
  -h, --help            show this help message and exit
  --forecast_start_date FORECAST_START_DATE
                        Optional argument for the `forecast` task. Indicate forecast start date in YYYY-MM-DD format. If not specified, forecast will start from the end of the target series.
  --new_model NEW_MODEL
                        Optional argument for the `train` task. Whether to train a newly created model or continue training an existing pre-trained model.
  --same_data SAME_DATA
                        Optional argument for the `dashboard` task. Whether to reuse previously created search database (faster start time) or update with new forecast data (slower start time).

NOTE: NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.
```
