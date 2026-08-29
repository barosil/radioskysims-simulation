"""\
Módulo: tests/domain/simulation/test_transits.py

Testes da extração de trânsitos por chunks (``transits.extract_transits``),
do ``SimulationEvents.from_dataset`` em datasets lazy (dask) e da persistência
streaming (``to_hdf5``/``to_zarr``) que respeita a estrutura de chunks sem
materializar o dataset completo na memória.

Copyright BINGO Collaboration
Last modified: 2026-08-04
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_core.domain.common import TimeRange
from radioskysims_simulation.domain.simulation import (
    SimulationOrchestratorResult,
    SimulationResult,
)
from radioskysims_simulation.domain.simulation.simulation import SimulationEvents
from radioskysims_simulation.domain.simulation.transits import (
    chunk_time,
    chunk_uniform,
    estimate_time_chunk,
    extract_transits,
    time_slices,
)


def _time_grid(n_time: int, step_s: int = 600) -> np.ndarray:
    start = np.datetime64("2026-07-26T00:00:00", "ns")
    return np.arange(
        start, start + n_time * np.timedelta64(step_s, "s"), np.timedelta64(step_s, "s")
    )


def _transit_dataset(n_time: int = 200, n_freq: int = 8, n_src: int = 3) -> xr.Dataset:
    """Dataset sintético com SAT-1 e Sun em trânsito e (opcional) SAT-2 fora."""
    time = _time_grid(n_time)
    freq = np.linspace(900e6, 1700e6, n_freq)
    T = np.zeros((n_time, n_freq, n_src, 1, 1))
    T[10:40, :, 0, 0, 0] = 5.0  # SAT-1 trânsito
    if n_src >= 2:
        T[120:160, :, 1, 0, 0] = 8.0  # Sun trânsito
    angles = np.full((n_time, n_src, 1), np.deg2rad(10.0))
    angles[10:40, 0, 0] = np.deg2rad(1.0)
    if n_src >= 2:
        angles[120:160, 1, 0] = np.deg2rad(1.0)
    src_names = ["SAT-1", "Sun", "SAT-2"][:n_src]
    src_types = ["satellite", "localsource", "satellite"][:n_src]
    ds = xr.Dataset(
        {
            "T_antenna": (("time", "frequency", "source", "polarization", "pointing"), T),
            "power": (("time", "frequency", "source", "polarization", "pointing"), T),
        },
        coords={
            "time": time,
            "frequency": freq,
            "source": src_names,
            "source_type": ("source", src_types),
            "polarization": ["V"],
            "pointing": [0],
            "angle_boresight": (("time", "source", "pointing"), angles),
        },
    )
    ds.attrs.update(
        {
            "beam_fwhm_deg": 1.0,
            "fwhm_factor": 2.0,
            "time_start": "2026-07-26T00:00:00",
            "time_step_s": 600.0,
            "pipeline": "tod",
        }
    )
    return ds


def _time_range() -> TimeRange:
    return TimeRange.from_iso(
        "2026-07-26T00:00:00",
        "2026-07-27T09:10:00",
        resolution=600,
    )


# ── Chunk helpers ───────────────────────────────────────────────


def test_chunk_time_respects_requested_size() -> None:
    ds = _transit_dataset()
    chunked = chunk_time(ds, chunk_s=25)
    assert chunked["T_antenna"].chunks == (
        (25,) * 8,
        (8,),
        (3,),
        (1,),
        (1,),
    )


def test_time_slices_covers_full_axis() -> None:
    ds = _transit_dataset()
    slices = time_slices(chunk_time(ds, chunk_s=25))
    assert [sl.stop - sl.start for sl in slices] == [25] * 8
    assert slices[-1].stop == 200


def test_estimate_time_chunk_positive() -> None:
    ds = _transit_dataset()
    assert estimate_time_chunk(ds, target_mb=64.0) >= 1


def test_chunk_time_caps_large_chunk_s_by_memory_budget() -> None:
    ds = _transit_dataset()  # time=200
    # chunk_s maior que o orçamento de memória é reduzido — nunca vira um
    # chunk único que materialize o dataset inteiro.
    chunked = chunk_time(ds, chunk_s=200, target_mb=0.001)
    n_time_chunks = len(chunked["T_antenna"].chunks[0])
    assert n_time_chunks > 1
    assert max(chunked["T_antenna"].chunks[0]) <= 2


def test_chunk_uniform_makes_non_time_dims_single_chunk() -> None:
    ds = _transit_dataset().chunk(
        {"time": 25, "frequency": 5, "source": 2}  # chunks irregulares
    )
    uniform = chunk_uniform(ds, chunk_s=25)
    # dims nao-temporais viram um unico chunk (full) — requisito do Zarr
    assert uniform["T_antenna"].chunks[1] == (8,)  # frequency
    assert uniform["T_antenna"].chunks[2] == (3,)  # source
    assert uniform["T_antenna"].chunks[3] == (1,)  # polarization


def test_to_zarr_with_irregular_chunks_roundtrip(tmp_path) -> None:
    """Regressao: chunks irregulares em frequency/source quebravam o Zarr."""
    ds = _transit_dataset().chunk({"frequency": 3, "source": 1})
    res = SimulationResult(
        dataset=ds,
        sources=[],
        satellites=[],
        time_range=_time_range(),
    )
    path = tmp_path / "tod.zarr"
    res.to_zarr(str(path), chunk_s=25)
    loaded = SimulationResult.load(str(path))
    np.testing.assert_allclose(
        loaded.dataset["T_antenna"].values,
        ds["T_antenna"].values,
    )


def test_extract_transits_with_large_chunk_s_is_capped() -> None:
    # Simula o caso real: dataset pequeno (~145 amostras) com chunk_s alto.
    ds = _transit_dataset()
    res = extract_transits(ds, chunk_s=1000)
    assert res.objects_in_beam == ["SAT-1", "Sun"]
    assert res.transits.sizes["transit"] == 2
    assert res.beam_events.sizes["time"] == 70


# ── Extração por chunks ─────────────────────────────────────────


def test_extract_transits_chunkwise_matches_known_transits() -> None:
    ds = _transit_dataset()
    # chunk_s=25 força 8 chunks — trânsitos cruzam fronteiras de chunk.
    res = extract_transits(ds, chunk_s=25)

    assert res.objects_in_beam == ["SAT-1", "Sun"]
    assert res.transits.sizes["transit"] == 2

    sat = res.transits.sel(transit=0)
    assert str(sat["source"].values) == "SAT-1"
    assert sat["duration_s"].values == 29 * 600.0
    assert float(sat["t_antenna_mean"].values) == pytest.approx(5.0)
    # SAT-2 (fora do feixe) não vira trânsito
    assert "SAT-2" not in res.objects_in_beam

    # beam_events mantém apenas os instantes em trânsito (30 + 40 amostras)
    assert res.beam_events.sizes["time"] == 70


def test_extract_transits_zeros_out_of_beam_sources() -> None:
    ds = _transit_dataset()
    res = extract_transits(ds, chunk_s=25)
    be = res.beam_events["T_antenna"]
    # primeiro instante de trânsito (t=10): SAT-1 ativo, Sun/SAT-2 zerados
    assert float(be.isel(time=0, source=0).sum()) > 0
    assert float(be.isel(time=0, source=1).sum()) == 0
    assert float(be.isel(time=0, source=2).sum()) == 0
    # instante do trânsito do Sol (t=120): Sun ativo, SAT-1 zerado
    idx = np.flatnonzero(be.coords["time"].values == ds.coords["time"].values[120])[0]
    assert float(be.isel(time=idx, source=1).sum()) > 0
    assert float(be.isel(time=idx, source=0).sum()) == 0


def test_extract_transits_on_lazy_dask_dataset() -> None:
    ds = _transit_dataset().chunk({"time": 25})
    res = extract_transits(ds, chunk_s=25)
    assert res.objects_in_beam == ["SAT-1", "Sun"]
    assert res.transits.sizes["transit"] == 2
    assert res.beam_events.sizes["time"] == 70


def test_extract_transits_never_computes_full_dataset_and_respects_chunk_s(monkeypatch) -> None:
    expected = extract_transits(_transit_dataset(), chunk_s=25)
    ds = _transit_dataset().chunk({"time": 25})

    full_n_time = int(ds.sizes["time"])
    seen_time_sizes: list[int] = []
    original_compute = xr.Dataset.compute

    def _spy_compute(self, *args, **kwargs):
        n_time = int(self.sizes.get("time", 0))
        if n_time == full_n_time:
            raise AssertionError("extract_transits must not compute the full dataset")
        seen_time_sizes.append(n_time)
        return original_compute(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "compute", _spy_compute)

    got = extract_transits(ds, chunk_s=25)

    assert seen_time_sizes == [25] * 8
    assert got.objects_in_beam == expected.objects_in_beam
    assert got.transits.sizes["transit"] == expected.transits.sizes["transit"]
    np.testing.assert_allclose(
        got.transits["t_antenna_mean"].values,
        expected.transits["t_antenna_mean"].values,
    )
    np.testing.assert_allclose(
        got.transits["duration_s"].values,
        expected.transits["duration_s"].values,
    )
    assert list(got.transits.coords["source"].values) == list(
        expected.transits.coords["source"].values
    )


def test_orchestrator_result_events_uses_chunk_compute_and_caches(monkeypatch) -> None:
    ds = _transit_dataset().chunk({"time": 25})
    sim_result = SimulationResult(
        dataset=ds,
        sources=[],
        satellites=[],
        time_range=_time_range(),
    )
    orchestrator_result = SimulationOrchestratorResult(
        simulation_result=sim_result,
        config={},
    )

    full_n_time = int(ds.sizes["time"])
    seen_time_sizes: list[int] = []
    original_compute = xr.Dataset.compute

    def _spy_compute(self, *args, **kwargs):
        n_time = int(self.sizes.get("time", 0))
        if n_time == full_n_time:
            raise AssertionError("events must not compute the full dataset")
        seen_time_sizes.append(n_time)
        return original_compute(self, *args, **kwargs)

    monkeypatch.setattr(
        "radioskysims_simulation.domain.simulation.transits.estimate_time_chunk",
        lambda dataset, target_mb=64.0: 25,
    )
    monkeypatch.setattr(xr.Dataset, "compute", _spy_compute)

    events_first = orchestrator_result.events
    seen_after_first = list(seen_time_sizes)
    events_second = orchestrator_result.events

    assert seen_after_first == [25] * 8
    assert seen_time_sizes == seen_after_first
    assert events_second is events_first
    assert set(events_first.objects_in_beam) == {"SAT-1", "Sun"}
    assert events_first.transits is not None
    assert events_first.transits.sizes["transit"] == 2


def test_source_types_filter() -> None:
    ds = _transit_dataset()
    res = extract_transits(ds, chunk_s=25, source_types=["satellite"])
    assert res.objects_in_beam == ["SAT-1"]


def test_simulation_events_from_dataset_lazy() -> None:
    ds = _transit_dataset().chunk({"time": 25})
    events = SimulationEvents.from_dataset(
        dataset=ds,
        time_range=_time_range(),
        fwhm_threshold_deg=2.0,
    )
    assert set(events.objects_in_beam) == {"SAT-1", "Sun"}
    assert events.transits is not None
    assert events.transits.sizes["transit"] == 2
    assert len(events.beam_events.coords["time"]) == 70
    assert events.duration_s == pytest.approx(33 * 3600 + 600)


# ── Persistência streaming (respeita chunks) ────────────────────


def _simulation_result() -> SimulationResult:
    ds = _transit_dataset(20, 4, 2)
    return SimulationResult(
        dataset=ds,
        sources=[],
        satellites=[],
        time_range=TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T03:10:00", resolution=600),
    )


def test_to_hdf5_streaming_roundtrip(tmp_path) -> None:
    res = _simulation_result()
    path = tmp_path / "tod.nc"
    # chunk_s=5 → múltiplos chunks de tempo (escrita por chunks)
    res.to_hdf5(str(path), chunk_s=5)
    loaded = SimulationResult.load(str(path))
    assert dict(loaded.dataset.sizes) == {
        "time": 20,
        "frequency": 4,
        "source": 2,
        "polarization": 1,
        "pointing": 1,
    }
    np.testing.assert_allclose(
        loaded.dataset["T_antenna"].values,
        res.dataset["T_antenna"].values,
    )
    assert loaded.time_range is not None


def test_to_hdf5_streaming_keeps_attrs(tmp_path) -> None:
    res = _simulation_result()
    path = tmp_path / "tod.nc"
    res.to_hdf5(str(path), chunk_s=5)
    loaded = SimulationResult.load(str(path))
    assert loaded.dataset.attrs.get("pipeline") == "tod"
    assert loaded.dataset.attrs.get("radioskysim_provenance") is not None


def test_to_zarr_streaming_roundtrip(tmp_path) -> None:
    res = _simulation_result()
    path = tmp_path / "tod.zarr"
    res.to_zarr(str(path), chunk_s=5)
    loaded = SimulationResult.load(str(path))
    assert dict(loaded.dataset.sizes)["time"] == 20
    np.testing.assert_allclose(
        loaded.dataset["T_antenna"].values,
        res.dataset["T_antenna"].values,
    )
    assert loaded.time_range is not None


def test_to_zarr_compute_false_returns_delayed(tmp_path) -> None:
    res = _simulation_result()
    path = tmp_path / "tod.zarr"
    delayed = res.to_zarr(str(path), chunk_s=5, compute=False)
    assert delayed is not None
    assert hasattr(delayed, "compute")


def test_to_zarr_no_encoding_warnings_and_strings_preserved(tmp_path) -> None:
    """Regressão: Zarr v3 emitia warnings de encoding (vlen-utf8/<U) nas coords."""
    import warnings

    res = _simulation_result()
    path = tmp_path / "tod.zarr"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res.to_zarr(str(path), chunk_s=5)

    enc_warnings = [
        str(x.message)
        for x in w
        if "not part in the Zarr format" in str(x.message)
        or "vlen-utf8" in str(x.message)
        or "dtype `<U" in str(x.message)
    ]
    assert not enc_warnings, enc_warnings

    loaded = SimulationResult.load(str(path))
    assert list(loaded.dataset.coords["source"].values) == ["SAT-1", "Sun"]
    assert list(loaded.dataset.coords["source_type"].values) == ["satellite", "localsource"]
