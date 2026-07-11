#!/usr/bin/env bash
# Capture dashboard screenshots for docs/images/{charts,scans,run}.png
# Requires: dashboard on http://127.0.0.1:7860 and Node with playwright.
# Usage:
#   python -m canswim dashboard --same_data True   # other terminal
#   ./scripts/capture-gui-docs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/docs/images"
BASE="${CANSWIM_GUI_URL:-http://127.0.0.1:7860/}"
mkdir -p "$OUT"

if ! curl -sf -o /dev/null "$BASE"; then
  echo "Dashboard not reachable at $BASE" >&2
  echo "Start: python -m canswim dashboard --same_data True" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
npm init -y >/dev/null 2>&1
npm install playwright@1.61.1 --no-save >/dev/null 2>&1

cat > capture.mjs <<EOF
import { chromium } from 'playwright';
import path from 'path';
import fs from 'fs';

const outDir = ${JSON.stringify(OUT)};
const base = ${JSON.stringify(BASE)};
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
await page.goto(base, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForTimeout(8000);

async function shotTab(label, file) {
  const tab = page.getByRole('tab', { name: label, exact: true });
  if (await tab.count()) await tab.first().click();
  else await page.getByText(label, { exact: true }).first().click();
  await page.waitForTimeout(2500);
  const p = path.join(outDir, file);
  await page.screenshot({ path: p, fullPage: false });
  console.log('saved', p, fs.statSync(p).size);
}

await shotTab('Charts', 'charts.png');
await shotTab('Scans', 'scans.png');
await shotTab('Run', 'run.png');
await browser.close();
EOF

NODE_PATH="$TMP/node_modules" node capture.mjs
echo "OK: updated $OUT/{charts,scans,run}.png"
