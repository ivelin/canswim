#!/usr/bin/bash
echo "Starting Optuna Dashboard"
optuna-dashboard sqlite:///data/optuna_study.db

