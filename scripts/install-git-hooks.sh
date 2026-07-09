#!/usr/bin/env bash
# Point this repo at versioned hooks under .githooks/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
chmod +x scripts/ci-local.sh scripts/gh-pr-merge-safe.sh .githooks/* 2>/dev/null || true
git config core.hooksPath .githooks
echo "Installed core.hooksPath=.githooks"
echo "  pre-push        → scripts/ci-local.sh"
echo "  pre-merge-commit → scripts/ci-local.sh (local git merge only)"
echo "Safe remote merge: ./scripts/gh-pr-merge-safe.sh <pr#>"
echo "Bypass: git push --no-verify   or   SKIP_LOCAL_CI=1 git push"
