#!/usr/bin/env bash
uv run bonsai_scout/bonsai_scout_preprocess.py \
    --results_folder /home/stan/Git/micropattern/notebooks/heemskerk_data/Bonsai_48h-1/ \
    --annotation_path '/home/stan/Git/micropattern/notebooks/heemskerk_data/adata_timeseries_old_48h-1_meta.csv' \
    --take_all_genes False \
    --config_filepath '' \
    --perform_annot_guided_clustering False
