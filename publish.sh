#!/usr/bin/bash
echo "Packaging canswim and publishing to PyPI repo"
set -exv

rm -r dist/

# install build deps
python3 -m pip install build
python3 -m pip install twine


# test repo publish
python3 -m build
python3 -m twine upload --verbose --repository testpypi dist/*
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps canswim

# proper repo publish
python3 -m twine upload --verbose dist/*
