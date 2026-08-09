#!/usr/bin/env bash
# Fetch the real TUM RGB-D sequence that tests/test_real_tum.py checks against.
#
# The sequence is CC BY 4.0 (Sturm et al., "A Benchmark for the Evaluation of
# RGB-D SLAM Systems", IROS 2012) and is deliberately *not* vendored: 328 MB of
# someone else's data does not belong in this repository, and the licence's
# attribution requirement is easier to honour by pointing at the source.
#
#   scripts/fetch_tum.sh              # into $TMPDIR/wurld-data
#   DEST=~/wurld-data scripts/fetch_tum.sh
#
# Then:  pytest tests/test_real_tum.py
# Or point at an existing copy:  WURLD_TUM_DIR=/path/to/rgbd_dataset_freiburg1_desk
set -euo pipefail

SEQ="rgbd_dataset_freiburg1_desk"
DEST="${DEST:-${TMPDIR:-/tmp}/wurld-data}"
URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/${SEQ}.tgz"

mkdir -p "$DEST"
if [ -f "$DEST/$SEQ/groundtruth.txt" ]; then
  echo "already present: $DEST/$SEQ"
  exit 0
fi

echo "fetching $SEQ (~328 MB) from cvg.cit.tum.de"
curl -fL --progress-bar -o "$DEST/$SEQ.tgz" "$URL"
tar xzf "$DEST/$SEQ.tgz" -C "$DEST"
rm -f "$DEST/$SEQ.tgz"
echo "extracted to $DEST/$SEQ"
echo
echo "run:  pytest tests/test_real_tum.py"
echo "      (or WURLD_TUM_DIR=$DEST/$SEQ pytest tests/test_real_tum.py)"
