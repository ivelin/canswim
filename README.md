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
  {dashboard,gatherdata,uploaddata,modelsearch,train,finetune,forecast}
                        Which canswim task to run:
                        `dashboard` for stock charting and scans of recorded forecasts.
                        'gatherdata` to gather 3rd party stock market data and save to HF Hub.
                        'uploaddata` upload to HF Hub any interim changes to local data storage.
                        `modelsearch` to find and save optimal hyperparameters for model training.
                        `train` for continuous model training.
                        `finetune` to fine tune pretrained model on new stock market data. `forecast` to run forecast on stocks and upload dataset to HF Hub.

options:
  -h, --help            show this help message and exit
  --forecast_start_date FORECAST_START_DATE
                        Optional argument for the `forecast` task. Indicate forecast start date in YYYY-MM-DD format. If not specified, forecast will start from the end of the target series.

NOTE: NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.
```

## Interactive Dashboard hosted on Hugging Face Hub


https://huggingface.co/spaces/ivelin/canswim_playground

![canswim playground](https://github.com/ivelin/canswim/assets/2234901/26fb1a5d-49e0-4888-bc1b-042800bcef8f)
