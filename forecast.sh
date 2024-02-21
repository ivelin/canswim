#!/usr/bin/bash
echo "Running forecast for multiple periods"
set exv


./canswim.sh forecast
./canswim.sh forecast --forecast_start_date "2023-11-18"
./canswim.sh forecast --forecast_start_date "2023-12-02"
./canswim.sh forecast --forecast_start_date "2023-12-16"
./canswim.sh forecast --forecast_start_date "2024-01-13"
./canswim.sh forecast --forecast_start_date "2024-01-27"
