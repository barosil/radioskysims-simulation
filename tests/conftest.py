"""Conftest do pacote radioskysims-simulation."""

from __future__ import annotations

import warnings

import pytest


def pytest_configure(config):  # noqa: ARG001
    warnings.filterwarnings("ignore", category=pytest.PytestUnraisableExceptionWarning)
    warnings.filterwarnings("ignore", message=r"numpy\.core\.", category=DeprecationWarning)
    # Aviso benigno de compatibilidade binária do netCDF4/h5py (numpy size changed)
    warnings.filterwarnings(
        "ignore", message=r"numpy\.ndarray size changed", category=RuntimeWarning
    )
