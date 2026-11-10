"""Build a Bonsai tree with the C++ Bonsai.

Build the executable first with `make bonsai` (on NixOS: `nix develop -c make bonsai`).
Input is a Sanity folder from scripts/run_sanity.py, or a merged multi-batch folder from
scripts/run_sanity_multibatch.py; which one it is gets detected from the files.

Usage: uv run python scripts/run_bonsai.py [data_dir] [out_dir]
"""

import subprocess
import sys
import time
from pathlib import Path

from pybonsai import BonsaiOutput

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- options -------------------------------------------------------------
# input folder: Sanity output in text format (scripts/run_sanity.py), or features.txt and
# standard_deviations.txt as merge_batches.py writes them
DATA_DIR = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/stan/Git/micropattern/notebooks/heemskerk_data/adata_timeseries_old_48h-1_raw_expr_hvg_sanity"
)
# output folder; default sits next to the input folder
OUT_DIR = Path(
    sys.argv[2] if len(sys.argv) > 2 else DATA_DIR.with_name(f"{DATA_DIR.name}_bonsai")
)
BONSAI = REPO_ROOT / "build" / "bonsai" / "bonsai"  # built by `make bonsai`
# True: DATA_DIR is Sanity output, and Bonsai divides Sanity's prior back out.
# False: DATA_DIR holds plain means and standard deviations (merged batches).
# None: Sanity output if DATA_DIR has a sanity_command.txt, otherwise plain.
INPUT_IS_SANITY_OUTPUT = None
# keep genes whose z-score exceeds this; the Bonsai authors use 1, the Python-Bonsai
# tree of this dataset used 0.75
ZSCORE = 1.0
THREADS = 12  # worker threads
VERBOSITY = 3  # 0=critical .. 4=debug; 3 prints progress per stage
# None starts from scratch; "auto" continues an interrupted run from its last checkpoint;
# a stage name such as "spr_moves" reruns from that stage
PICKUP = None
OVERWRITE = False  # start from scratch even if OUT_DIR already holds a finished tree
EXTRA_ARGS: list[str] = []  # further options, e.g. ["--num-neighbours", "20"]
# --------------------------------------------------------------------------

if not BONSAI.is_file():
    raise SystemExit(f"{BONSAI} not found; run `make bonsai` first")
sanity_input = INPUT_IS_SANITY_OUTPUT
if sanity_input is None:
    sanity_input = (DATA_DIR / "sanity_command.txt").exists()
    if not sanity_input and not (DATA_DIR / "features.txt").exists():
        raise SystemExit(
            f"{DATA_DIR} holds neither sanity_command.txt nor features.txt"
        )
final_tree = OUT_DIR / "final_bonsai_tree"
if final_tree.exists() and PICKUP is None and not OVERWRITE:
    raise SystemExit(
        f"{final_tree} already exists; set OVERWRITE = True or pick another OUT_DIR"
    )

command = [
    str(BONSAI),
    "--out", str(OUT_DIR),
    "--threads", str(THREADS),
    "--verbosity", str(VERBOSITY),
]  # fmt: skip
if PICKUP is None:
    command += ["--data", str(DATA_DIR), "--zscore", str(ZSCORE)]
    if sanity_input:
        command.append("--input-is-sanity-output")
else:
    # preprocessing is skipped on pickup, so the input folder and z-score are not passed
    command += ["--pickup", PICKUP]
command += EXTRA_ARGS

OUT_DIR.mkdir(parents=True, exist_ok=True)
stdout_log = OUT_DIR / "run.stdout.log"
stderr_log = OUT_DIR / "run.stderr.log"
print(" ".join(command))
print(f"tree-change log: {stdout_log}\nprogress log:    {stderr_log}")

start = time.time()
# stdout is a record of every tree change and can be large, so it goes to a file only;
# stderr carries progress and is shown as well as stored
with open(stdout_log, "a") as out, open(stderr_log, "a") as err:
    process = subprocess.Popen(command, stdout=out, stderr=subprocess.PIPE, text=True)
    assert process.stderr is not None
    for line in process.stderr:
        sys.stderr.write(line)
        err.write(line)
        err.flush()
    returncode = process.wait()
minutes = (time.time() - start) / 60
if returncode != 0:
    raise SystemExit(
        f"bonsai exited with code {returncode} after {minutes:.1f} min; "
        'set PICKUP = "auto" to continue from the last checkpoint'
    )

# load the result with pybonsai to confirm the analysis side can read it
tree = BonsaiOutput(final_tree, topology_only=True)
print(
    f"done in {minutes:.1f} min: {final_tree}\n"
    f"{int(tree.is_sample.sum())} cells, {tree.n_nodes} nodes, {tree.n_features} genes"
)
