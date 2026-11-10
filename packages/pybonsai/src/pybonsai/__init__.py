"""Python bindings for the C++ Bonsai stack (bonsai_port and penjing)."""

try:
    from ._core import BonsaiOutput, PairDistances
except ModuleNotFoundError as error:
    # the package installs without its extension when the upstream submodules are absent
    raise ImportError(
        "pybonsai was installed without its C++ extension: the penjing and "
        "bonsai_port submodules under packages/pybonsai/third_party were not "
        "checked out when it was built. See packages/pybonsai/README.md."
    ) from error

__all__ = ["BonsaiOutput", "PairDistances"]
