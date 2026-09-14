#!/bin/bash
set -euo pipefail

: "${PA_REMOTE_HOST:=hoffman}"
: "${PA_REMOTE_SSH:=ssh}"
: "${PA_REMOTE_OUTPUT_ROOT:?set remote outputs/preferential_attachment/mobility_edge}"
: "${PA_LOCAL_OUTPUT_ROOT:?set canonical local outputs/preferential_attachment/mobility_edge}"

mkdir -p "$PA_LOCAL_OUTPUT_ROOT/nonlinear" "$PA_LOCAL_OUTPUT_ROOT/extreme_positive_edge_quantiles"

rsync -av -e "$PA_REMOTE_SSH" --prune-empty-dirs \
  --exclude='partials/' \
  --exclude='spacing_q_*.npz' \
  --exclude='*.pyc' \
  "$PA_REMOTE_HOST:$PA_REMOTE_OUTPUT_ROOT/nonlinear/sublinear/" \
  "$PA_LOCAL_OUTPUT_ROOT/nonlinear/sublinear/"

rsync -av -e "$PA_REMOTE_SSH" --prune-empty-dirs \
  --exclude='augmentation_caches/' \
  --exclude='edge_gap_and_ratio_records.npz' \
  --exclude='*.pyc' \
  "$PA_REMOTE_HOST:$PA_REMOTE_OUTPUT_ROOT/extreme_positive_edge_quantiles/" \
  "$PA_LOCAL_OUTPUT_ROOT/extreme_positive_edge_quantiles/"

for zeta_slug in 1p05 1p1 1p2 1p3 1p5; do
  remote="$PA_REMOTE_OUTPUT_ROOT/nonlinear/n_5000_m_10/attachment_zeta_$zeta_slug/global_spectrum/nonlinear_hub_edge/"
  local_dir="$PA_LOCAL_OUTPUT_ROOT/nonlinear/n_5000_m_10/attachment_zeta_$zeta_slug/global_spectrum/nonlinear_hub_edge/"
  mkdir -p "$local_dir"
  rsync -av -e "$PA_REMOTE_SSH" "$PA_REMOTE_HOST:$remote" "$local_dir"
done

cross_regime_remote="$PA_REMOTE_OUTPUT_ROOT/nonlinear/n_5000_m_10/nonlinear_hub_edge_cross_regime_summary/"
cross_regime_local="$PA_LOCAL_OUTPUT_ROOT/nonlinear/n_5000_m_10/nonlinear_hub_edge_cross_regime_summary/"
mkdir -p "$cross_regime_local"
rsync -av -e "$PA_REMOTE_SSH" "$PA_REMOTE_HOST:$cross_regime_remote" "$cross_regime_local"
