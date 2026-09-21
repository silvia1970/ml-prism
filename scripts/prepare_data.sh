#!/usr/bin/env bash
# =============================================================================
# PRISM — data preparation
# Creates the expected datasets/ layout and fetches / points to the raw data.
#
#   SepsisExp : public (CC-BY 4.0)  -> downloaded automatically when possible
#   MIMIC-III : credentialed (PhysioNet) -> manual download, instructions printed
#
# Usage:
#   bash scripts/prepare_data.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/datasets"

# --- expected directory layout ---------------------------------------------
mkdir -p \
  "$DATA/MIMIC/carry_forward/mean/train" \
  "$DATA/MIMIC/carry_forward/mean/val" \
  "$DATA/MIMIC/carry_forward/mean/test" \
  "$DATA/MIMIC/original" \
  "$DATA/SepsisExp"

echo "==> datasets/ layout created under: $DATA"

# --- SepsisExp (public) ------------------------------------------------------
SEPSIS_EXP_URL="https://www.cl.uni-heidelberg.de/statnlpgroup/sepsisexp/"
echo
echo "==> SepsisExp (public, CC-BY 4.0)"
echo "    Download the 4 time-series partitions (A/B/C/D) from:"
echo "      $SEPSIS_EXP_URL"
echo "    and place them as:"
echo "      $DATA/SepsisExp/sepsisexp_timeseries_partition-{A,B,C,D}.{csv,tsv}"
if command -v wget >/dev/null 2>&1; then
  echo "    (wget available — if the site exposes direct links, download them here)"
fi

# --- MIMIC-III (credentialed) ------------------------------------------------
echo
echo "==> MIMIC-III (credentialed access required)"
echo "    1. Complete the PhysioNet credentialed-data course and accept the license:"
echo "         https://physionet.org/content/mimiciii/1.4/"
echo "    2. Download the MIMIC-III files into a local folder."
echo "    3. Run the loader/preprocessor to build the carry_forward/mean arrays:"
echo "         python statistics/load_mimic.py   (see statistics/README.md)"
echo
echo "    Expected outputs under $DATA/MIMIC/carry_forward/mean/{train,val,test}/:"
echo "      x_ml.npy, x_dl.npy, X_{train,val,test}.tsv, y.npy, y_{train,val,test}.tsv"

echo
echo "==> Done. See datasets/README.md for the full expected layout."
