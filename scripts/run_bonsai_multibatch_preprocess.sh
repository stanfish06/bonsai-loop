#!/usr/bin/env bash
# Bonsai-scout preprocess after a multibatch Bonsai run (see micropattern/.../run_bonsai_multibatch.sh).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv run "${REPO_ROOT}/Bonsai-data-representation/bonsai_scout/bonsai_scout_preprocess.py" \
    --results_folder /home/stan/Git/micropattern/data/multibatch/bonsai_out \
    --annotation_path /home/stan/Git/micropattern/data/multibatch/cell_metadata.csv \
    --take_all_genes False \
    --config_filepath '' \
    --perform_annot_guided_clustering False
