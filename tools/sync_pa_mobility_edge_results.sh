#!/bin/bash
set -euo pipefail

: "${PA_REMOTE_HOST:=hoffman}"
: "${PA_REMOTE_OUTPUT_ROOT:?set the remote mobility_edge output root}"
: "${PA_LOCAL_OUTPUT_ROOT:?set the local outputs/preferential_attachment/mobility_edge directory}"

mkdir -p "$PA_LOCAL_OUTPUT_ROOT"
rsync -av --prune-empty-dirs \
  --exclude='partials/' \
  --exclude='*.pyc' \
  "$PA_REMOTE_HOST:$PA_REMOTE_OUTPUT_ROOT/" \
  "$PA_LOCAL_OUTPUT_ROOT/"
