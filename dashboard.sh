#!/bin/bash
# activate conda with required native OS and python package dependencies
conda activate canswim
# install local canswim package in dev mode
pip install -e ./
gradio src/canswim/dashboard.py
