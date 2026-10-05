#!/bin/bash
set -euo pipefail
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
if [[ $# -ne 3 ]]; then
  echo "usage: $0 part1|part2 N alpha_index" >&2
  exit 2
fi
STAGE="$1"
N="$2"
ALPHA_INDEX="$3"
case "$STAGE:$N" in
  part1:500) SCRIPT=tools/hoffman_linear_pa_two_part_part1_n500.sh ;;
  part1:10000) SCRIPT=tools/hoffman_linear_pa_two_part_part1_n10000.sh ;;
  part2:500) SCRIPT=tools/hoffman_linear_pa_two_part_part2_n500.sh ;;
  part2:10000) SCRIPT=tools/hoffman_linear_pa_two_part_part2_n10000.sh ;;
  *) echo "unsupported stage/N combination: $STAGE/$N" >&2; exit 2 ;;
esac
JOB=$(qsub -terse -t "$ALPHA_INDEX" "$SCRIPT")
echo "$JOB"
