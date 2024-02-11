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
from canswim import train, dashboard

load_dotenv(override=True)
logging_dir = os.getenv("logging_dir", "tmp")
logger.add(f"{logging_dir}/canswim.log", rotation="24 hours", retention="30 days")

# Instantiate the parser
parser = argparse.ArgumentParser(
    prog="canswim",
    description="""CANSWIM is a toolkit for CANSLIM style investors.
        Humbly complements the Simple Moving Average and other technical indicators.
        """,
    epilog="NOTE: NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.",
)

parser.add_argument(
    "module",
    type=str,
    help="""Which %(prog)s module to run: 
        `dashboard` for stock scans and charting of uploaded forecasts or 
        `train` for continuous training and forecast uploads""",
    choices=["dashboard", "train"],
)


# Required positional argument
# parser.add_argument('pos_arg', type=int,
#                    help='A required integer positional argument')
# Optional positional argument
# parser.add_argument('opt_pos_arg', type=int, nargs='?',
#                    help='An optional integer positional argument')
# Optional argument
# parser.add_argument('--opt_arg', type=int,
#                    help='An optional integer argument')
# Switch
# parser.add_argument('--switch', action='store_true',
#                    help='A boolean switch')

args = parser.parse_args()

logger.info("argyment module:", module=args.module)

match args.module:
    case "train":
        train.main()
    case "dashboard":
        dashboard.main()
    case _:
        logger.error("Unrecognized module argument {m} ", m=args.module)
