#!/usr/bin/env bash
# Stage the browser viewer into docs/ for GitHub Pages.
#
#   scripts/build-pages.sh
#
# viewer/ stays the single source and keeps importing chromapakz from
# node_modules so local development works with `npm install`. The hosted copy
# has no node_modules, so the import is rewritten to a pinned CDN build —
# pinned, not floating, because a silent chromapakz major would break the
# published demo with no commit to blame.
#
# Output under docs/viewer is generated; edit viewer/ and re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHROMAPAKZ_VERSION="0.7.0"
CDN="https://cdn.jsdelivr.net/npm/chromapakz@${CHROMAPAKZ_VERSION}/src/chromapakz.js"

OUT="$ROOT/docs/viewer"
mkdir -p "$OUT" "$ROOT/docs/samples"

for f in index.html live.html; do
  sed "s#\.\./node_modules/chromapakz/src/chromapakz\.js#${CDN}#g" \
    "$ROOT/viewer/$f" > "$OUT/$f"
  if grep -q "node_modules" "$OUT/$f"; then
    echo "error: $f still references node_modules after rewrite" >&2
    exit 1
  fi
done
cp "$ROOT/viewer/wurld.js" "$OUT/wurld.js"

echo "staged:"
ls -la "$OUT"
echo
echo "chromapakz pinned to ${CHROMAPAKZ_VERSION} via jsDelivr"
