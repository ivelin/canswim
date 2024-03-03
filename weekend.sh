#!/usr/bin/bash

# stop on first error
set -e

# gather up to date market data
./canswim gatherdata

# run forecast and upload to hf hub
./canswim forecast

