#!/usr/bin/bash
#args=("$@")

# This script is intended to be executed by cron each weekend.
# The goal is to pull latest market data for the week 
# and run forecast for the following week.

# stop on first error
# print verbose messages
set -ex

echo "CANSWIM Weekend Forecast Routine: Starting..."

conda activate canswim
pip install -e ./

#python -m canswim "${args[@]}"

python -m canswim 

# gather up to date market data
##./canswim.sh gatherdata

# run forecast and upload to hf hub
##./canswim.sh forecast

echo "CANSWIM Weekend Forecast Routine: Finished."


