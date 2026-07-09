#!/usr/bin/env bash
# Merge a PR only after local CI and remote checks are green.
# Usage: ./scripts/gh-pr-merge-safe.sh <pr-number> [--squash|--merge|--rebase]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pr-number> [--squash|--merge|--rebase]" >&2
  exit 2
fi

PR="$1"
shift || true
MERGE_FLAG="--merge"
if [[ "${1:-}" == "--squash" || "${1:-}" == "--merge" || "${1:-}" == "--rebase" ]]; then
  MERGE_FLAG="$1"
fi

echo "==> local CI before merge"
"$ROOT/scripts/ci-local.sh"

echo "==> remote checks for PR #$PR"
if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found" >&2
  exit 1
fi

# Fail if any check is not pass/skip
checks_out="$(gh pr checks "$PR" 2>&1)" || true
echo "$checks_out"

if echo "$checks_out" | grep -qiE $'\t(fail|pending|queued|in_progress)\t|^fail|^pending'; then
  # parse carefully: gh pr checks format is NAME\tSTATUS\t...
  bad=0
  while IFS=$'\t' read -r name status rest; do
    [[ -z "${name:-}" ]] && continue
    case "${status,,}" in
      pass|success|skipping|skip|neutral) ;;
      *)
        echo "Blocking merge: check '$name' status='$status'" >&2
        bad=1
        ;;
    esac
  done <<< "$checks_out"
  if [[ "$bad" -eq 1 ]]; then
    echo "Remote CI not green — refusing merge (see AGENTS.md)" >&2
    exit 1
  fi
fi

# Also require overall success via JSON when available
state="$(gh pr view "$PR" --json state -q .state 2>/dev/null || echo OPEN)"
if [[ "$state" != "OPEN" ]]; then
  echo "PR #$PR state is $state — nothing to merge" >&2
  exit 1
fi

echo "==> merging PR #$PR ($MERGE_FLAG)"
gh pr merge "$PR" "$MERGE_FLAG"
echo "Merged PR #$PR"
