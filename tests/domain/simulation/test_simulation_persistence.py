"""\
Módulo: tests/domain/simulation/test_simulation_persistence.py

Testes de persistência do ``SimulationResult`` (HDF5/NetCDF e Zarr):
round-trip save/load, materialização in-place do ``compute()`` e
reconstrução do resultado composto (``save_result``/``load_result``).

Copyright BINGO Collaboration
Last modified: 2026-08-03
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from radioskysims_core.domain.common import FrequencyRange, TimeRange
from radioskysims_simulation.domain.simulation import (
    SimulationOrchestratorResult,
    SimulationResult,
)
from radioskysims_simulation.domain.simulation.summary import assemble_config, format_summary


def _dataset() -> xr.Dataset:
    n_time, n_freq, n_src = 12, 8, 2
    time = np.arange(
        np.datetime64("2026-07-26T00:00:00", "ns"),
        np.datetime64("2026-07-26T00:00:00", "ns") + n_time * np.timedelta64(600, "s"),
        np.timedelta64(600, "s"),
    )
    freq = np.linspace(900e6, 1700e6, n_freq)
    T = np.zeros((n_time, n_freq, n_src, 1, 1))
    T[:, :, 0, 0, 0] = 5.0
    T[2:5, :, 1, 0, 0] = 20.0
    return xr.Dataset(
        {
            "T_antenna": (("time", "frequency", "source", "polarization", "pointing"), T),
            "power": (("time", "frequency", "source", "polarization", "pointing"), T),
        },
        coords={
            "time": time,
            "frequency": freq,
            "source": ["SAT-1", "Sun"],
            "source_type": ("source", ["satellite", "localsource"]),
            "polarization": ["V"],
            "pointing": [0],
        },
    )


def _dataset_three_types() -> xr.Dataset:
    n_time, n_freq, n_src = 12, 8, 3
    time = np.arange(
        np.datetime64("2026-07-26T00:00:00", "ns"),
        np.datetime64("2026-07-26T00:00:00", "ns") + n_time * np.timedelta64(600, "s"),
        np.timedelta64(600, "s"),
    )
    freq = np.linspace(900e6, 1700e6, n_freq)
    T = np.zeros((n_time, n_freq, n_src, 1, 1))
    T[:, :, 0, 0, 0] = 5.0
    T[2:5, :, 1, 0, 0] = 20.0
    T[6:9, :, 2, 0, 0] = 30.0
    return xr.Dataset(
        {
            "T_antenna": (("time", "frequency", "source", "polarization", "pointing"), T),
            "power": (("time", "frequency", "source", "polarization", "pointing"), T),
        },
        coords={
            "time": time,
            "frequency": freq,
            "source": ["SAT-1", "Sun", "NVSS-1"],
            "source_type": ("source", ["satellite", "localsource", "radiosource"]),
            "polarization": ["V"],
            "pointing": [0],
        },
    )


def _time_range() -> TimeRange:
    return TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T02:00:00", resolution=600)


def _result() -> SimulationResult:
    return SimulationResult(
        dataset=_dataset(),
        sources=[],
        satellites=[],
        time_range=_time_range(),
        provenance={
            "pipeline": "tod",
            "created_at": "2026-07-26T00:00:00+00:00",
            "n_sources": 2,
        },
    )


def _result_three_types() -> SimulationResult:
    return SimulationResult(
        dataset=_dataset_three_types(),
        sources=[],
        satellites=[],
        time_range=_time_range(),
        provenance={
            "pipeline": "tod",
            "created_at": "2026-07-26T00:00:00+00:00",
            "n_sources": 3,
        },
    )


def _config() -> dict:
    tr = _time_range()
    fr = FrequencyRange(min_mhz=900.0, max_mhz=1700.0, resolution_mhz=0.2)
    from datetime import UTC, datetime

    from radioskysims_core.domain.common import Observer
    from radioskysims_sources.domain.satellite.models import OrbitalElement, Satellite

    obs = Observer(lat=-7.23, lon=-35.72, height=600.0, timeframe=tr)
    sats = [
        Satellite(
            norad_id=1,
            name="GPS-1",
            constellation="GPS",
            epoch=datetime(2026, 7, 25, tzinfo=UTC),
            element=OrbitalElement(
                epoch=datetime(2026, 7, 25, tzinfo=UTC),
                tle_line1="1 00001",
                tle_line2="2 00001",
            ),
        )
    ]
    return assemble_config(
        telescope_name="bingo",
        observer=obs,
        frequency_range=fr,
        antennas=[_Antenna()],
        num_channels=4096,
        time_range=tr,
        satellites=sats,
        include_local_sources=True,
        include_radio_sources=True,
        start_utc="2026-07-26T00:00:00+00:00",
        computing_time_s=12.5,
        delayed=True,
    )


class _Antenna:
    def __init__(self) -> None:
        self.name = "horn01"
        self.pointing = type("P", (), {"alt_deg": 90.0, "az_deg": 0.0})()
        self.beam = type("B", (), {"polarizations": ["V", "H"]})()


# ── HDF5 / NetCDF ─────────────────────────────────────────────


def test_hdf5_roundtrip(tmp_path) -> None:
    result = _result()
    path = tmp_path / "sim.nc"
    result.to_hdf5(str(path))
    assert path.exists()

    loaded = SimulationResult.from_hdf5(str(path))
    assert loaded.dataset.sizes == result.dataset.sizes
    assert "T_antenna" in loaded.dataset
    # TimeRange reconstruído dos attrs
    assert loaded.time_range is not None
    assert loaded.time_range.start_jd == pytest.approx(result.time_range.start_jd)
    assert loaded.time_range.end_jd == pytest.approx(result.time_range.end_jd)
    assert loaded.time_range.resolution_sec == pytest.approx(result.time_range.resolution_sec)
    # FrequencyRange derivado da coordenada
    assert loaded.freq_range is not None
    assert loaded.freq_range.min_mhz == pytest.approx(900.0)
    assert loaded.freq_range.max_mhz == pytest.approx(1700.0)
    # Proveniência persistida
    assert loaded.provenance.get("pipeline") == "tod"
    assert loaded.provenance.get("n_sources") == 2


def test_save_load_hdf5_auto_detect(tmp_path) -> None:
    result = _result()
    path = tmp_path / "sim_result.h5"
    result.save(str(path))
    loaded = SimulationResult.load(str(path))
    assert loaded.dataset.sizes == result.dataset.sizes


# ── Zarr ──────────────────────────────────────────────────────


def test_zarr_roundtrip(tmp_path) -> None:
    pytest.importorskip("zarr")
    result = _result()
    path = tmp_path / "sim.zarr"
    result.to_zarr(str(path))
    assert path.is_dir()

    loaded = SimulationResult.from_zarr(str(path))
    assert loaded.dataset.sizes == result.dataset.sizes
    assert "T_antenna" in loaded.dataset
    assert loaded.time_range.start_jd == pytest.approx(result.time_range.start_jd)
    assert loaded.freq_range is not None
    assert loaded.provenance.get("pipeline") == "tod"


def test_save_load_zarr_auto_detect(tmp_path) -> None:
    pytest.importorskip("zarr")
    result = _result()
    path = tmp_path / "sim.zarr"
    result.save(str(path))
    loaded = SimulationResult.load(str(path))
    assert loaded.dataset.sizes == result.dataset.sizes


def test_select_source_type_filters_satellite_and_preserves_lazy_state() -> None:
    import dask.array as da

    result = SimulationResult(
        dataset=_dataset_three_types().chunk({"time": 4, "frequency": 4}),
        sources=[
            type("S", (), {"name": "SAT-1", "source_type": "satellite"})(),
            type("S", (), {"name": "Sun", "source_type": "localsource"})(),
            type("S", (), {"name": "NVSS-1", "source_type": "radiosource"})(),
        ],
        satellites=[type("S", (), {"name": "SAT-1", "source_type": "satellite"})()],
        time_range=_time_range(),
        provenance={"pipeline": "tod", "created_at": "2026-07-26T00:00:00+00:00"},
    )

    selected = result.select_source_type("satellite")

    assert list(selected.dataset.coords["source"].values) == ["SAT-1"]
    assert list(selected.dataset.coords["source_type"].values) == ["satellite"]
    assert list(selected.dataset.coords["source_type"].dims) == ["source"]
    assert list(selected.dataset.sizes.keys()) == [
        "time",
        "frequency",
        "source",
        "polarization",
        "pointing",
    ]
    assert selected.dataset.sizes["source"] == 1
    assert selected.time_range.start_jd == pytest.approx(result.time_range.start_jd)
    assert selected.time_range.end_jd == pytest.approx(result.time_range.end_jd)
    assert selected.provenance == result.provenance
    assert list(result.dataset.coords["source"].values) == ["SAT-1", "Sun", "NVSS-1"]
    assert isinstance(selected.dataset["T_antenna"].data, da.Array)
    assert isinstance(selected.dataset["power"].data, da.Array)


def test_select_source_type_filters_localsource_and_radiosource() -> None:
    result = _result_three_types()

    local = result.select_source_type("localsource")
    radio = result.select_source_type("radiosource")

    assert list(local.dataset.coords["source"].values) == ["Sun"]
    assert list(local.dataset.coords["source_type"].values) == ["localsource"]
    assert local.dataset.sizes["source"] == 1

    assert list(radio.dataset.coords["source"].values) == ["NVSS-1"]
    assert list(radio.dataset.coords["source_type"].values) == ["radiosource"]
    assert radio.dataset.sizes["source"] == 1


def test_select_source_type_rejects_invalid_and_missing_types() -> None:
    result = _result()

    with pytest.raises(ValueError, match="source_type inválido"):
        result.select_source_type("galaxy")

    with pytest.raises(ValueError, match="Nenhuma fonte com source_type='radiosource'"):
        result.select_source_type("radiosource")


def test_to_zarr_with_source_type_roundtrip(tmp_path) -> None:
    pytest.importorskip("zarr")
    result = _result_three_types()
    before_sources = list(result.dataset.coords["source"].values)
    before_types = list(result.dataset.coords["source_type"].values)
    path = tmp_path / "filtered.zarr"

    result.to_zarr(str(path), source_type="localsource")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        loaded = SimulationResult.load(str(path))

    assert list(result.dataset.coords["source"].values) == before_sources
    assert list(result.dataset.coords["source_type"].values) == before_types
    assert list(loaded.dataset.coords["source"].values) == ["Sun"]
    assert list(loaded.dataset.coords["source_type"].values) == ["localsource"]
    assert loaded.dataset.sizes["source"] == 1
    assert loaded.provenance.get("pipeline") == "tod"


def test_to_hdf5_with_source_type_roundtrip(tmp_path) -> None:
    result = _result_three_types()
    before_sources = list(result.dataset.coords["source"].values)
    before_types = list(result.dataset.coords["source_type"].values)
    path = tmp_path / "filtered.nc"

    result.to_hdf5(str(path), source_type="radiosource")
    loaded = SimulationResult.load(str(path))

    assert list(result.dataset.coords["source"].values) == before_sources
    assert list(result.dataset.coords["source_type"].values) == before_types
    assert list(loaded.dataset.coords["source"].values) == ["NVSS-1"]
    assert list(loaded.dataset.coords["source_type"].values) == ["radiosource"]
    assert loaded.dataset.sizes["source"] == 1
    assert loaded.provenance.get("pipeline") == "tod"


# ── compute() materializa in-place ────────────────────────────


def test_compute_materializes_in_place() -> None:
    import dask.array as da

    ds_lazy = _dataset().chunk({"time": 4, "frequency": 4})
    result = SimulationResult(
        dataset=ds_lazy,
        sources=[],
        satellites=[],
        time_range=_time_range(),
    )
    assert isinstance(result.dataset["T_antenna"].data, da.Array)

    computed = result.compute()
    assert isinstance(computed, xr.Dataset)
    # In-place: o dataset do objeto agora é numpy → summary preenche stats
    assert not isinstance(result.dataset["T_antenna"].data, da.Array)
    assert float(result.dataset["T_antenna"].max().item()) == pytest.approx(20.0)


# ── Resultado composto (SimulationOrchestratorResult) ─────────


def test_composite_save_result_load_result(tmp_path) -> None:
    result = _result()
    orch = SimulationOrchestratorResult(simulation_result=result, config=_config())
    path = tmp_path / "sim_result.zarr"

    sidecar = orch.save_result(str(path))
    assert (tmp_path / "sim_result.zarr.json").exists()
    assert sidecar.endswith(".json")

    loaded = SimulationOrchestratorResult.load_result(str(path))
    assert loaded.simulation_result is not None
    assert loaded.config["observer"]["name"] == "bingo"
    assert loaded.config["computation"]["saved"] is True
    assert loaded.config["computation"]["filepath"] == str(path)
    assert "T_antenna" in loaded.simulation_result.dataset


def test_format_summary_renders_flux() -> None:
    cfg = _config()
    cfg["simulation"]["sky"]["localsources"]["Sun"]["flux_jy"] = {"value_jy": 123.4}
    txt = format_summary(cfg)
    assert "123.4 Jy" in txt
