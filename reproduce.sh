#!/usr/bin/env bash
# reproduce.sh — regenerate every paper artefact from the banked result CSVs.
#
#   1. tables  <- results/*.csv        (code/gen_tables_n100.py)
#   2. figures <- results/*.csv        (code/make_figures_n100.py)
#   3. paper PDF <- paper/*.tex       (pdflatex, twice for references)
#   4. copies the PDF and a working-tree ZIP to the goal files directory
#
# This does NOT rerun the experiments themselves (the 190-seed synthetic
# reruns take hours; the CIFAR-10H study needs its raw CSV -- see the
# CIFAR10H_CSV lookup in code/real_world_cifar10h.py).
# Rerun drivers: code/upgrade_n100.py (synthetic, --resume is safe),
# code/run_all_n10.py (ten-seed suite), code/real_world_cifar10h.py
# (needs the CIFAR-10H raw CSV; see its docstring).
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
PDFLATEX=${PDFLATEX:-pdflatex}

echo "==> [1/4] regenerating tables from results/*.csv"
$PY code/gen_tables_n100.py >/dev/null
echo "==> [2/4] regenerating figures from results/*.csv"
$PY code/make_figures_n100.py >/dev/null
echo "==> [3/4] rebuilding paper/main.pdf"
(cd paper && $PDFLATEX -interaction=nonstopmode main.tex >/dev/null \
  && $PDFLATEX -interaction=nonstopmode main.tex >/dev/null)
echo "==> [4/4] packaging PDF + ZIP"
OUT="$HOME/workspace/goals/provenance-aware-belief-aggregation-research-paper/files"
if [ -d "$OUT" ]; then
  cp paper/main.pdf "$OUT/PVI_Paper_Jayant_Dubey.pdf"
  zip -qr /tmp/pvi-paper-tree.zip . \
    -x ".git/*" "*/__pycache__/*" "*.pyc" ".venv/*" "*.aux" "*.log" "*.out" \
    "results/*smoke*.csv"
  cp /tmp/pvi-paper-tree.zip "$OUT/provenance-aware-belief-aggregation.zip"
  rm -f /tmp/pvi-paper-tree.zip
  echo "done."
  echo "  PDF: $OUT/PVI_Paper_Jayant_Dubey.pdf"
  echo "  ZIP: $OUT/provenance-aware-belief-aggregation.zip"
else
  echo "done. (local packaging skipped: $OUT does not exist on this machine)"
  echo "  PDF: paper/main.pdf"
fi
