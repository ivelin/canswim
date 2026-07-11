#!/usr/bin/env bash
# Build and publish canswim to TestPyPI and/or PyPI.
# Usage:
#   ./publish.sh test     # TestPyPI only
#   ./publish.sh prod     # PyPI (after version bump on main)
#   ./publish.sh both     # TestPyPI then PyPI
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:-test}"

python3 -m pip install -q build twine
rm -rf dist/ build/ *.egg-info src/*.egg-info
python3 -m build

version="$(python3 -c "import configparser; c=configparser.ConfigParser(); c.read('setup.cfg'); print(c['metadata']['version'])")"
echo "Built canswim==${version}"

case "$MODE" in
  test)
    python3 -m twine upload --verbose --repository testpypi dist/*
    echo "Install: pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ canswim==${version}"
    ;;
  prod)
    python3 -m twine upload --verbose dist/*
    echo "Install: pip install canswim==${version}"
    ;;
  both)
    python3 -m twine upload --verbose --repository testpypi dist/*
    python3 -m twine upload --verbose dist/*
    ;;
  *)
    echo "Usage: $0 test|prod|both" >&2
    exit 1
    ;;
esac
