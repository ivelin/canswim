"""
   Copyright 2024 Cocoonhive LLC

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from loguru import logger
from dotenv import load_dotenv
import os
import argparse
from canswim import dashboard, gather_data, model_search, train, forecast
from canswim.hfhub import HFHub

# Instantiate the parser
parser = argparse.ArgumentParser(
    prog="canswim",
    description="""CANSWIM is a toolkit for CANSLIM style investors.
        Aims to complement the Simple Moving Average and other technical indicators.
        """,
    epilog="NOTE: NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.",
)

parser.add_argument(
    "task",
    type=str,
    help="""Which %(prog)s task to run:
        `dashboard` for stock charting and scans of recorded forecasts.
        'gatherdata` to gather 3rd party stock market data and save to HF Hub.
        'uploaddata` upload to HF Hub any interim changes to local data storage.
        `modelsearch` to find and save optimal hyperparameters for model training.
        `train` for continuous model training.
        `finetune` to fine tune pretrained model on new stock market data.
        `forecast` to run forecast on stocks and upload dataset to HF Hub.
        """,
    choices=[
        "dashboard",
        "gatherdata",
        "uploaddata",
        "modelsearch",
        "train",
        "finetune",
        "forecast",
    ],
)

parser.add_argument(
    '--forecast_start_date',
    type=str,
    required=False,
    help="""Optional argument for the `forecast` task. Indicate forecast start date in YYYY-MM-DD format. If not specified, forecast will start from the end of the target series.""")

args = parser.parse_args()

load_dotenv(override=True)
logging_dir = os.getenv("logging_dir", "tmp")
logging_path = f"{logging_dir}/canswim.log"
rot = "24 hours"
ret = "30 days"
logger.add(logging_path, rotation=rot, retention=ret)

logger.info(
    "Logging to: {p} with rotation {rot} and retention {ret}",
    p=logging_path,
    rot=rot,
    ret=ret,
)

logger.info("command line args: {args}", args=args)

hfhub = HFHub()
load_dotenv(override=True)
repo_id = os.getenv("repo_id", "ivelin/canswim")


match args.task:
    case "dashboard":
        dashboard.main()
    case "modelsearch":
        model_search.main()
    case "gatherdata":
        gather_data.main()
    case "uploaddata":
        hfhub.upload_data()
    case "train":
        train.main()
    case "finetune":
        raise NotImplementedError("finetune task not implemented yet")
    case "forecast":
        forecast.main(forecast_start_date=args.forecast_start_date)
    case _:
        logger.error("Unrecognized task argument {m} ", m=args.module)
