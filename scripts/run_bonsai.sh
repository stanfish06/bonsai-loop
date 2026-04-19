#!/usr/bin/env bash
nix-shell -p openmpi --command 'LD_LIBRARY_PATH=$(dirname $(which mpiexec))/../lib mpiexec -n 8 uv run bonsai/bonsai_main.py --config_filepath /home/stan/Git/micropattern/Bonsai-data-representation/48h-1.yaml --step all'
