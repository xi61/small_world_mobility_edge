#!/bin/bash
set -euo pipefail

: "${PA_REMOTE_HOST:=hoffman}"
: "${PA_REMOTE_MICROSCOPE_ROOT:?set remote abnormal_band_microscope root}"
: "${PA_LOCAL_MICROSCOPE_ROOT:?set local abnormal_band_microscope root}"

mkdir -p "$PA_LOCAL_MICROSCOPE_ROOT"
rsync -av --prune-empty-dirs --exclude='*.pyc'   "$PA_REMOTE_HOST:$PA_REMOTE_MICROSCOPE_ROOT/"   "$PA_LOCAL_MICROSCOPE_ROOT/"
