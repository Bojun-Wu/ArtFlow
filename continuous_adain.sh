#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Paths below are relative to the project folder.
cd "$(dirname "$0")"

# Usage: bash continuous_adain.sh "content.png" "style.png" 20
# Change these defaults, or pass content, style, and rounds as arguments.
CONTENT_IMAGE="${1:-data/content/example.jpg}"
STYLE_IMAGE="${2:-data/style/example.jpg}"
ROUNDS="${3:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-output_continuous_adain}"
MODEL="${MODEL:-experiments/ArtFlow-AdaIN/glow.pth}"
# These architecture settings must match the selected checkpoint.
N_FLOW="${N_FLOW:-8}"
N_BLOCK="${N_BLOCK:-2}"
SAVE_EXT="${SAVE_EXT:-.png}"
# Leave SIZE unset to use the Python default (256); omit --size entirely.
# Explicit overrides: SIZE=512 resizes the shorter side; SIZE=0 keeps original dimensions.
# Example: SIZE=512 bash continuous_adain.sh content.png style.png 20
size_args=()
if [[ -n "${SIZE:-}" ]]; then
    size_args=(--size "$SIZE")
fi

# Use the project environment on Windows; override PYTHON to use another one.
PYTHON="${PYTHON:-./env/python.exe}"

# Use the original continuous_transfer.py pipeline with single-image arguments.
# Load the model once and pass full floating-point outputs to the next round.
# PNGs are display copies only: reloading them would clip and quantize the data.
# Filenames include microsecond timestamps, method, current round, and total rounds.
"$PYTHON" -u continuous_transfer.py \
    --content "$CONTENT_IMAGE" --style "$STYLE_IMAGE" --rounds "$ROUNDS" \
    --decoder "$MODEL" --output "$OUTPUT_DIR" "${size_args[@]}" \
    --operator adain --n_flow "$N_FLOW" --n_block "$N_BLOCK" --save_ext "$SAVE_EXT"
