"""Build a tree with the Python Bonsai, starting from cellstates.

Every cell stays a leaf, but Bonsai starts from a tree in which the cells of one
cellstate already share an ancestor, instead of all cells hanging off the root. This is
the Bonsai authors' "cs_premerged" route:

1. create_config_file.py writes a Bonsai config that names the starting-tree folder.
2. create_cellstates_premerged_tree.py (third_party/pipeline) builds that starting tree.
3. bonsai_main.py runs the preprocess, core_calc and metadata steps.

Inputs: a Sanity folder for all cells (scripts/run_sanity.py) and the cellstate labels
of the same cells in the same order (scripts/run_cellstates.py).

Usage: uv run python scripts/run_bonsai_old_premerged.py [sanity_dir] [clusters_file] [out_dir]
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pybonsai import BonsaiOutput

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- options -------------------------------------------------------------
# Sanity folder of all cells, text format
SANITY_DIR = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/stan/Git/micropattern/notebooks/heemskerk_data/adata_timeseries_old_48h-1_raw_expr_hvg_sanity"
)
# one cellstate label per line, in the cell order of SANITY_DIR/cellID.txt
CLUSTERS_FILE = Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else SANITY_DIR.with_name(f"{SANITY_DIR.name.removesuffix('_sanity')}_cellstates")
    / "optimized_clusters.txt"
)
# output folder; default sits next to the Sanity folder
OUT_DIR = Path(
    sys.argv[3]
    if len(sys.argv) > 3
    else SANITY_DIR.with_name(f"{SANITY_DIR.name}_bonsai_premerged")
)
DATASET = OUT_DIR.name  # dataset name written into the config
ZSCORE = 1.0  # keep genes whose z-score exceeds this
# Bonsai steps to run, in order; drop the finished ones to continue an interrupted run
STEPS = ["preprocess", "core_calc", "metadata"]
# MPI processes for the Bonsai steps; more than 1 needs mpiexec on PATH and mpi4py installed
MPI_PROCESSES = 1
OVERWRITE = False  # rebuild the config and starting tree even if OUT_DIR has them
# --------------------------------------------------------------------------

BONSAI_REPO = REPO_ROOT / "third_party" / "Bonsai-data-representation"
PREMERGE_SCRIPT = (
    REPO_ROOT
    / "third_party"
    / "pipeline"
    / "workflow"
    / "scripts"
    / "create_cellstates_premerged_tree.py"
)
config_path = OUT_DIR / "bonsai_config.yaml"
premerged_folder = OUT_DIR / "premerged_tree"

if not PREMERGE_SCRIPT.is_file():
    raise SystemExit(
        f"{PREMERGE_SCRIPT} not found; run `git submodule update --init third_party/pipeline`"
    )
n_cells = len((SANITY_DIR / "cellID.txt").read_text().split("\n")) - 1
n_labels = len(CLUSTERS_FILE.read_text().split())
if n_labels != n_cells:
    raise SystemExit(
        f"{CLUSTERS_FILE} has {n_labels} labels, {SANITY_DIR} has {n_cells} cells"
    )
mpi_prefix: list[str] = []
if MPI_PROCESSES > 1:
    # without mpi4py every process would run the whole job on its own and write the same files
    if importlib.util.find_spec("mpi4py") is None or shutil.which("mpiexec") is None:
        raise SystemExit("MPI_PROCESSES > 1 needs mpi4py installed and mpiexec on PATH")
    mpi_prefix = ["mpiexec", "-n", str(MPI_PROCESSES)]

OUT_DIR.mkdir(parents=True, exist_ok=True)
log_dir = OUT_DIR / "logs"
log_dir.mkdir(exist_ok=True)
# the Bonsai scripts import their own package from the repository root
environment = {**os.environ, "PYTHONPATH": str(BONSAI_REPO)}


def run(name: str, command: list[str]) -> None:
    """Run one stage from the Bonsai repository, showing its output and storing it in logs/."""
    print(f"== {name}: {' '.join(command)}")
    start = time.time()
    with open(log_dir / f"{name}.log", "w") as log:
        process = subprocess.Popen(
            command,
            cwd=BONSAI_REPO,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        returncode = process.wait()
    minutes = (time.time() - start) / 60
    if returncode != 0:
        raise SystemExit(
            f"{name} exited with code {returncode} after {minutes:.1f} min"
        )
    print(f"== {name} done in {minutes:.1f} min")


if OVERWRITE or not config_path.exists():
    run(
        "create_config",
        [
            sys.executable,
            str(BONSAI_REPO / "bonsai" / "create_config_file.py"),
            "--new_yaml_path",
            str(config_path),
            "--dataset",
            DATASET,
            "--data_folder",
            str(SANITY_DIR.resolve()),
            "--results_folder",
            str(OUT_DIR.resolve()),
            "--filenames_data",
            "delta.txt,d_delta.txt",
            "--input_is_sanity_output",
            "true",
            "--zscore_cutoff",
            str(ZSCORE),
            "--tmp_folder",
            str(premerged_folder.resolve()),
        ],  # fmt: skip
    )
if OVERWRITE or not (premerged_folder / "tree.nwk").exists():
    run(
        "create_premerged_tree",
        [
            sys.executable,
            str(PREMERGE_SCRIPT),
            "--config_filepath",
            str(config_path.resolve()),
            "--cellstates_file",
            str(CLUSTERS_FILE.resolve()),
            "--premerged_folder",
            str(premerged_folder.resolve()),
        ],  # fmt: skip
    )
for step in STEPS:
    run(
        step,
        [
            *mpi_prefix,
            sys.executable,
            "-u",
            str(BONSAI_REPO / "bonsai" / "bonsai_main.py"),
            "--config_filepath",
            str(config_path.resolve()),
            "--step",
            step,
        ],  # fmt: skip
    )

final_trees = sorted(OUT_DIR.glob("final_bonsai*"))
if not final_trees:
    raise SystemExit(f"no final_bonsai* folder in {OUT_DIR}; see {log_dir}")
# load the result with pybonsai to confirm the analysis side can read it
tree = BonsaiOutput(final_trees[-1], topology_only=True)
print(
    f"{final_trees[-1]}\n"
    f"{int(tree.is_sample.sum())} cells, {tree.n_nodes} nodes, {tree.n_features} genes"
)
