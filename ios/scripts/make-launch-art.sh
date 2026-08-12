#!/usr/bin/env bash
# Regenerate the full-bleed launch art with OpenAI's image API and install it
# into the asset catalog.
#
#   OPENAI_API_KEY=sk-... ios/scripts/make-launch-art.sh [prompt-override]
#
# The key is read from the environment only — never passed as an argument, so it
# cannot land in shell history or `ps` output.
#
# The shipped scene is a living room rendered entirely as a luminous point
# cloud, filling the frame — the thing the app produces, as the first thing you
# see. Each run redraws the scene from scratch (the model never produces the
# same image twice), so only replace the committed art when the new draw
# actually beats it.
#
# Composition contract, encoded in the prompt: near-black navy background with
# every edge vignetting into it. The storyboard aspect-fills this image, so each
# device crops it differently; art that fades to the background makes every crop
# look deliberate, and the storyboard's matching background colour covers any
# sliver the image does not reach.
set -euo pipefail

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "error: OPENAI_API_KEY is not set" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Pillow is needed for the sanity checks; the repo venv has it.
PYTHON="${PYTHON:-$ROOT/../.venv/bin/python}"
command -v "$PYTHON" >/dev/null || PYTHON=python3
DEST="$ROOT/WurldCam/Sources/Assets.xcassets/LaunchArt.imageset/LaunchArt.png"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

DEFAULT_PROMPT='Vertical smartphone launch screen, near-black deep navy
background. HIGH CONTRAST: the scene glows intensely against the dark,
instantly readable. Absolutely no text, no logos, no UI. A glowing room
rendered as thousands of luminous cyan and electric-blue points, like embers
of blue fire: sofa, floor lamp, side table, rug and the suggestion of walls,
seen in a wide immersive perspective that FILLS THE ENTIRE FRAME top to bottom
— the ceiling dissolves into sparse drifting points at the very top, the rug
and floor points run off the bottom edge, furniture large and close. Dense,
rich, luminous, electric. No rings, no orbit lines, no circles, no halos.'

PROMPT="${1:-$DEFAULT_PROMPT}"

echo "requesting image…"
curl -sS https://api.openai.com/v1/images/generations \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$("$PYTHON" - "$PROMPT" <<'PY'
import json, sys
print(json.dumps({
    "model": "gpt-image-1",
    "prompt": " ".join(sys.argv[1].split()),
    "size": "1024x1536",
    "output_format": "png",
    "quality": "high",
}))
PY
)" > "$TMP/resp.json"

"$PYTHON" - "$TMP/resp.json" "$TMP/raw.png" <<'PY'
import base64, json, sys
resp = json.load(open(sys.argv[1]))
if "error" in resp:
    raise SystemExit(f"OpenAI error: {resp['error'].get('message', resp['error'])}")
open(sys.argv[2], "wb").write(base64.b64decode(resp["data"][0]["b64_json"]))
PY

"$PYTHON" - "$TMP/raw.png" "$DEST" <<'PY'
import sys
from PIL import Image
img = Image.open(sys.argv[1]).convert("RGB")
w, h = img.size
if h <= w:
    raise SystemExit(f"got {w}x{h}; launch art must be portrait")
img.save(sys.argv[2], "PNG", optimize=True)
print(f"installed {sys.argv[2]} ({w}x{h})")
PY

echo "done. Rebuild (ios/scripts/install-device.sh) to see it. iOS caches launch"
echo "screens aggressively: delete the app once if the old art keeps appearing."
