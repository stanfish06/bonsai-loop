The `.py` scripts are current.

- `run_sanity.py`: fit Sanity on the raw counts of a `.h5ad` with sanity-py and write a Sanity folder that
  Bonsai reads. `--counts X|layer|obsm` and `--key NAME` say where the counts are; a tab-separated
  count table also works. `--help` lists the options.
- `run_sanity_multibatch.py`: fit Sanity per batch and merge the batches with the Python Bonsai's
  `merge_batches.py`; writes a merged `bonsai_in` folder. Takes a `.h5ad` with `--batch-key OBS_COLUMN`
  and the same `--counts`/`--key` options, or a count table with `--batch-annotation FILE`.
- `run_bonsai.py`: build a tree from either folder with the C++ Bonsai. Build it once with `make bonsai`.
- `run_cellstates.py`: cluster cells into cellstates; writes `optimized_clusters.txt` and the aggregated
  `Gene_table.txt`. Takes the same input and `--counts`/`--key` options as `run_sanity.py`. With
  `--batch-key OBS_COLUMN` it clusters each batch separately and writes `batch_annotation.tsv` for
  `run_sanity_multibatch.py`. `--n-target N` merges the cellstates further until N are left, for data
  where cellstates alone merges almost nothing. Build the executables once with `make cellstates`.
- `run_bonsai_old_premerged.py`: build a tree with the Python Bonsai, starting from the cellstates of
  `optimized_clusters.txt`; every cell stays a leaf.
- `check_posterior_pdists.py`: check `bonsai_lib.get_pdists_posterior` on a Bonsai output folder.

The `.sh` scripts use the Python Bonsai. `run_bonsai_old.sh` builds a tree with it, and the
others run Bonsai Scout, which reads only trees built by the Python Bonsai. They are old and
wont work without updating paths and/or installing deps
