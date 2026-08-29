"""
Testes para o SimulationContext (domínio simulation).

Cobre:
- sort_sources_by_ra: ordena fontes por RA ascendente (integração com sources).

O teste original vivia em sources (`tests/domain/source/test_source.py`), mas a
classe pertence ao domínio de simulation — movido aqui na Fase 7 (remoção de
shims de compatibilidade).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from radioskysims_simulation.domain.simulation.context import SimulationContext
from radioskysims_sources.domain.source import SourceFactory


@pytest.fixture
def sample_tracks():
    """Dataset sintético com 10 timesteps."""
    n = 10
    time = np.linspace(2460000.0, 2460001.0, n)
    return xr.Dataset(
        data_vars={
            "theta": ("time", np.full(n, 0.5)),
            "az": ("time", np.linspace(0, 2 * np.pi, n)),
            "el": ("time", np.full(n, 0.8)),
            "range": ("time", np.full(n, 4e7)),
            "ra": ("time", np.linspace(0, 2 * np.pi, n)),
            "dec": ("time", np.full(n, 0.5)),
        },
        coords={"time": time},
    )


@pytest.fixture
def sample_psd():
    """PSD sintético em W/Hz."""
    freq = np.linspace(1.2e9, 1.6e9, 100)
    return xr.DataArray(
        np.ones_like(freq) * 1e-20,
        dims="frequency",
        coords={"frequency": freq},
        attrs={"units": "W/Hz"},
    )


class TestSimulationContext:
    """Testes do SimulationContext."""

    def test_sort_sources_by_ra(self, sample_tracks, sample_psd):
        """sort_sources_by_ra ordena fontes por RA ascendente."""
        # Cria 3 fontes com RA crescente mas em ordem bagunçada
        t0 = sample_tracks.copy()
        t1 = sample_tracks.copy()
        t2 = sample_tracks.copy()
        t0["ra"] = ("time", np.full(len(t0.time), np.deg2rad(200.0)))
        t1["ra"] = ("time", np.full(len(t1.time), np.deg2rad(10.0)))
        t2["ra"] = ("time", np.full(len(t2.time), np.deg2rad(100.0)))

        s0 = SourceFactory.from_satellite(name="ra200", tracks=t0, psd=sample_psd)
        s1 = SourceFactory.from_satellite(name="ra10", tracks=t1, psd=sample_psd)
        s2 = SourceFactory.from_satellite(name="ra100", tracks=t2, psd=sample_psd)

        ctx = SimulationContext(
            sources=[s0, s1, s2],
            time_range=None,
            radiotelescope=None,
        )
        ctx.sort_sources_by_ra()
        names = [s.name for s in ctx.sources]
        assert names == ["ra10", "ra100", "ra200"], f"Esperado ordenado por RA, obtido {names}"
