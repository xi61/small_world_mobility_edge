#!/bin/bash
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${RS_REMOTE_HOST:?set RS_REMOTE_HOST to the SSH host name}"

: "${RS_REMOTE_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_20260921}"
: "${RS_LOCAL_ROOT:=outputs/random_schrodinger_20260921}"

mkdir -p "$RS_LOCAL_ROOT"
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='*.json' \
  --include='*.csv' \
  --include='*.csv.gz' \
  --include='*.png' \
  --include='*.log' \
  --exclude='*' \
  "${RS_REMOTE_HOST}:$RS_REMOTE_ROOT/" "$RS_LOCAL_ROOT/"
