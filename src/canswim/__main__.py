#!/usr/bin/env python
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

import signal
import sys
from dotenv import load_dotenv
from importlib import metadata

# load os env vars before loguru import
# otherwise it won't pickup LOGURU_LEVEL
load_dotenv(override=True)

from loguru import logger
import os
import argparse

from canswim.run_triggers import (
    FORECAST_START_HELP,
    TICKERS_HELP,
    RUNS_OPT_IN_HELP,
)

# Instantiate the parser
parser = argparse.ArgumentParser(
    prog="canswim",
    description="""CANSWIM is a toolkit for CANSLIM style investors.
        Aims to complement the Simple Moving Average and other technical indicators.
        """,
    epilog=(
        "NOTE: NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.\n"
        "Get market data: gatherdata --tickers …\n"
        "Run a forecast: forecast --tickers … [--forecast_start_date] [--dry_run]\n"
        "See docs/run_triggers.md for start-date rules and operator detail.\n"
        f"{RUNS_OPT_IN_HELP}"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument(
    "task",
    type=str,
    help="""Which %(prog)s task to run:
        `dashboard` — charts, scans, get market data, run forecasts.
        `gatherdata` — get market data (all symbols, or --tickers for a short list).
        `downloaddata` — download model training and forecast data from HF Hub.
        `uploaddata` — upload local train/forecast data to HF Hub.
        `modelsearch` — find and save optimal hyperparameters.
        `train` — continuous model training (full history).
        `forecast` — run forecasts (all symbols, or --tickers for a short list).
        `mcp` — MCP server (read-only by default; write tools need MCP_ALLOW_RUNS=1).
        `resolve_start` — show which forecast start date would be used.
        """,
    choices=[
        "dashboard",
        "gatherdata",
        "downloaddata",
        "uploaddata",
        "modelsearch",
        "train",
        "forecast",
        "mcp",
        "resolve_start",
    ],
)

parser.add_argument(
    "--tickers",
    type=str,
    required=False,
    default=None,
    help=(
        f"Limit get-market-data / forecast to these symbols. {TICKERS_HELP} "
        "Matches the dashboard and MCP tools."
    ),
)

parser.add_argument(
    "--forecast_start_date",
    type=str,
    required=False,
    help=(
        f"For `forecast` and `resolve_start`. {FORECAST_START_HELP}"
    ),
)

parser.add_argument(
    "--dry_run",
    action="store_true",
    default=False,
    help=(
        "With `forecast --tickers`: only check symbols and start date (no model run)."
    ),
)

parser.add_argument(
    "--no_covariates",
    action="store_true",
    default=False,
    help="With `gatherdata --tickers`: prices only (skip earnings, dividends, etc.).",
)

parser.add_argument(
    "--new_model",
    type=bool,
    required=False,
    default=False,
    help="""Optional argument for the `train` task. Whether to train a newly created model or continue training an existing pre-trained model.""",
)

parser.add_argument(
    "--same_data",
    type=bool,
    required=False,
    default=False,
    help="""Optional argument for the `dashboard` task. Whether to reuse previously created search database (faster start time) or update with new forecast data (slower start time).""",
)

log_level = os.getenv("LOG_LEVEL", os.getenv("LOGURU_LEVEL", "INFO")).upper()
logger.remove()
logger.add(sys.stderr, level=log_level)

logging_dir = os.getenv("logging_dir", "tmp")
logging_path = f"{logging_dir}/canswim.log"
rot = "24 hours"
ret = "30 days"
logger.add(logging_path, level=log_level, rotation=rot, retention=ret)

logger.info(
    "Logging to: {p} with rotation {rot} and retention {ret}",
    p=logging_path,
    rot=rot,
    ret=ret,
)

version = metadata.version("canswim")
logger.info(f"canswim version: {version}")

args = parser.parse_args()

logger.info("command line args: {args}", args=args)

load_dotenv(override=True)


def signal_handler(sig, frame):
    print("Ctrl+C - Exit")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Lazy imports keep the MCP path free of torch / Gradio / HF Hub.
match args.task:
    case "dashboard":
        from canswim import dashboard

        dashboard.main(same_data=args.same_data)
    case "modelsearch":
        from canswim import model_search

        model_search.main()
    case "gatherdata":
        if args.tickers:
            from canswim.cli_run import run_gather_tickers

            sys.exit(
                run_gather_tickers(
                    args.tickers,
                    include_covariates=not args.no_covariates,
                )
            )
        from canswim import gather_data

        gather_data.main()
    case "downloaddata":
        from canswim.hfhub import HFHub

        HFHub().download_data()
    case "uploaddata":
        from canswim.hfhub import HFHub

        HFHub().upload_data()
    case "train":
        from canswim import train

        train.main(new_model=args.new_model)
    case "forecast":
        if args.tickers:
            from canswim.cli_run import run_forecast_tickers

            sys.exit(
                run_forecast_tickers(
                    args.tickers,
                    forecast_start_date=args.forecast_start_date,
                    dry_run=args.dry_run,
                )
            )
        if args.dry_run:
            logger.error("--dry_run requires --tickers (scoped forecast path)")
            sys.exit(2)
        from canswim import forecast

        forecast.main(forecast_start_date=args.forecast_start_date)
    case "resolve_start":
        from canswim.cli_run import run_resolve_start

        sys.exit(run_resolve_start(args.forecast_start_date))
    case "mcp":
        from canswim.mcp.server import main as mcp_main

        mcp_main()
    case _:
        logger.error("Unrecognized task argument {m} ", m=args.task)
