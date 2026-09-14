#!/bin/bash
set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"

: "${PA_REMOTE_HOST:=hoffman}"
: "${PA_REMOTE_SINGLE_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_single_graph_20260901/outputs}"
: "${PA_REMOTE_SINGLE_LOG_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_single_graph_20260901/logs}"
: "${PA_LOCAL_SINGLE_ROOT:=outputs/preferential_attachment/single_graph_edge_bulk}"

mkdir -p "$PA_LOCAL_SINGLE_ROOT" "$PA_LOCAL_SINGLE_ROOT/logs"
rsync -av --prune-empty-dirs --exclude='*.pyc' \
  "$PA_REMOTE_HOST:$PA_REMOTE_SINGLE_ROOT/" \
  "$PA_LOCAL_SINGLE_ROOT/"
rsync -av --exclude='*.pyc' \
  "$PA_REMOTE_HOST:$PA_REMOTE_SINGLE_LOG_ROOT/" \
  "$PA_LOCAL_SINGLE_ROOT/logs/"
