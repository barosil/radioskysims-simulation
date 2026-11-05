import cProfile
import pstats
import sys
import time

import numpy as np
import pytest
import xarray as xr

from radioskysims_simulation.domain.tod.simulator import TODSimulator


@pytest.mark.performance
def test_tod_simulator_performance():
    """Mede performance da simulação TOD com ~1.000.000 amostras.

    Requisitos:
    - usa cProfile + pstats
    - não persiste arquivos
    - determinístico (seed fixa)
    """

    np.random.seed(0)

    N_SAMPLES = 1_000_000
    FREQS = np.array([1e9])

    class SimpleSat:
        def __init__(self, n_time, freqs):
            self._freq = xr.DataArray(freqs, dims=("frequency",), coords={"frequency": freqs})
            times = np.arange(n_time).astype(float)
            # angle in radians
            self._angle = xr.DataArray(
                np.linspace(0.0, np.pi, n_time), dims=("time",), coords={"time": times}
            )
            self._range = xr.DataArray(np.full(n_time, 1e6), dims=("time",), coords={"time": times})

        def psd(self):
            da = xr.DataArray(
                np.ones_like(self._freq.values),
                dims=("frequency",),
                coords={"frequency": self._freq.values},
            )
            da.attrs["units"] = "W/Hz"
            return da

        def angle(self):
            return self._angle

        def range(self):
            return self._range

    class SimpleBeam:
        def normalized(self, kind: str = "integral"):  # noqa: ARG002
            return self

        def evaluate(self, *, frequency: xr.DataArray, angle: xr.DataArray, polarization=None):  # noqa: ARG002
            # fast, vectorized amplitude: shape (time, freq)
            freq_vals = np.asarray(frequency.values)
            ang_vals = np.asarray(angle.values)
            amp = np.exp(-ang_vals / np.pi)[:, None] * np.ones((1, freq_vals.size))
            return xr.DataArray(
                amp,
                dims=("time", "frequency"),
                coords={"time": angle.coords["time"], "frequency": frequency.values},
            )

    sat = SimpleSat(N_SAMPLES, FREQS)
    beam = SimpleBeam()

    sim = TODSimulator(
        beam=beam,
        gain=1.0,
        effective_area_m2=1.0,
    )

    # Usar prof.runcall para evitar conflito com pytest-cov
    prof = cProfile.Profile()
    start = time.perf_counter()
    ds = prof.runcall(sim.simulate, sat=sat)  # type: ignore
    elapsed = time.perf_counter() - start

    # montar dataset['data'] conforme requisito de checagem
    ds_check = ds.copy()
    ds_check["data"] = ds_check["power"]

    # Relatório: ordenar por tempo cumulativo e imprimir top20
    ps = pstats.Stats(prof, stream=sys.stdout)
    ps.strip_dirs().sort_stats("cumtime").print_stats(20)

    assert ds_check["data"].size == 1_000_000
    assert elapsed < 5.0
