`run_sanity.py`, `run_bonsai.py` and `check_posterior_pdists.py` are current.

- `run_sanity.py`: fit Sanity on a count table with sanity-py and write a Sanity folder that Bonsai reads.
- `run_bonsai.py`: build a tree from that folder with the C++ Bonsai. Build it once with `make bonsai`.
- `check_posterior_pdists.py`: check `bonsai_lib.get_pdists_posterior` on a Bonsai output folder.

The `.sh` scripts use the Python Bonsai. `run_bonsai_old.sh` builds a tree with it, and the
others run Bonsai Scout, which reads only trees built by the Python Bonsai. They are old and
wont work without updating paths and/or installing deps
