from pathlib import Path

import pytest

from pybonsai import BonsaiOutput

# 64-cell, 1131-feature Bonsai outputs vendored in penjing, one per implementation.
FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "third_party" / "penjing" / "tests" / "data"
)
FIXTURES = {
    "python": FIXTURE_ROOT / "bonsai_output_python" / "final_bonsai_tree",
    "cpp": FIXTURE_ROOT / "bonsai_output" / "final_bonsai_tree",
}


@pytest.fixture(scope="session", params=sorted(FIXTURES))
def tree_folder(request):
    folder = FIXTURES[request.param]
    if not folder.is_dir():
        pytest.skip(f"{folder} missing; run `git submodule update --init`")
    return folder


@pytest.fixture(scope="session")
def output(tree_folder):
    return BonsaiOutput(tree_folder)
