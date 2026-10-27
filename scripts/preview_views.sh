#!/bin/bash
# Preview-quality close-ups of named views (see VIEWS in render_closeup.py) for checking a change.
#
# Usage:
#     scripts/preview_views.sh TAG FRAME [--cycles] [--lit] [--nofur] VIEW...
#
# Renders each view in EEVEE, and with --cycles also in Cycles, into
# build/preview/TAG_VIEW_{eevee,cycles}_FRAME.png, and a montage of all of them into
# build/preview/TAG.jpg (needs ImageMagick). Logs go to build/preview/TAG_VIEW_ENGINE.log.
set -u
cd "$(dirname "$0")/.."
tag=$1; frame=$2; shift 2
engines=(eevee); extra=(preview); views=()
for arg in "$@"; do
  case $arg in
    --cycles) engines+=(cycles) ;;
    --lit) extra+=(lit) ;;
    --nofur) extra+=(nofur) ;;
    *) views+=("$arg") ;;
  esac
done
f=$(printf %04d "$frame")
outs=()
for view in "${views[@]}"; do
  for engine in "${engines[@]}"; do
    pre=(); [ "$engine" = cycles ] && pre=(--python scripts/render_cycles.py)
    log=build/preview/${tag}_${view}_${engine}.log
    # --python-exit-code applies only to the --python options after it.
    timeout 300 blender -b build/cat_scooter.blend --python-exit-code 1 "${pre[@]}" --python scripts/render_closeup.py \
      -o "//preview/${tag}_${view}_${engine}_####" -F PNG -f "$frame" -- "$view" "${extra[@]}" > "$log" 2>&1
    status=$?
    out=build/preview/${tag}_${view}_${engine}_$f.png
    if [ $status -ne 0 ]; then
      echo "$view $engine: failed with exit status $status, see $log"
    else
      echo "$out"
      outs+=("$out")
    fi
  done
done
if [ ${#outs[@]} -gt 1 ] && command -v magick > /dev/null; then
  magick montage "${outs[@]}" -tile "${#engines[@]}x" -geometry +2+2 "build/preview/$tag.jpg" && echo "build/preview/$tag.jpg"
fi
