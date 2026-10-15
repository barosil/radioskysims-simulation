"""\
Módulo: tests/domain/simulation/test_simulation_summary.py

Testes do resumo estruturado de simulação (Observer/Simulation/Computation)
e do ``SimulationOrchestratorResult.summary()``/``format_summary()``.

Copyright BINGO Collaboration
Last modified: 2026-08-03
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr

from radioskysims_core.domain.common import FrequencyRange, Observer, TimeRange
from radioskysims_sources.domain.satellite.models import OrbitalElement, Satellite
from radioskysims_simulation.domain.simulation import (
    SimulationOrchestratorResult,
    SimulationResult,
)
from radioskysims_simulation.domain.simulation.summary import (
    assemble_config,
    format_summary,
    populate_computation,
    populate_sky_stats,
    to_json,
)


def _satellites() -> list[Satellite]:
    return [
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
        ),
        Satellite(
            norad_id=2,
            name="GAL-1",
            constellation="GALILEO",
            epoch=datetime(2026, 7, 24, tzinfo=UTC),
            element=OrbitalElement(
                epoch=datetime(2026, 7, 24, tzinfo=UTC),
                tle_line1="1 00002",
                tle_line2="2 00002",
            ),
        ),
    ]


class _Antenna:
    def __init__(self, name: str = "horn01") -> None:
        self.name = name
        self.pointing = type("P", (), {"alt_deg": 90.0, "az_deg": 0.0})()
        self.beam = type("B", (), {"polarizations": ["V", "H"]})()


def _config() -> dict:
    tr = TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-27T00:00:00", resolution=600)
    fr = FrequencyRange(min_mhz=900.0, max_mhz=1700.0, resolution_mhz=0.2)
    obs = Observer(lat=-7.23, lon=-35.72, height=600.0, timeframe=tr)
    return assemble_config(
        telescope_name="bingo",
        observer=obs,
        frequency_range=fr,
        antennas=[_Antenna()],
        num_channels=4096,
        time_range=tr,
        satellites=_satellites(),
        include_local_sources=True,
        include_radio_sources=True,
        start_utc="2026-07-26T00:00:00+00:00",
        computing_time_s=12.5,
        delayed=True,
    )


def test_assemble_config_observer_block() -> None:
    cfg = _config()
    obs = cfg["observer"]
    assert obs["name"] == "bingo"
    assert obs["location"]["lat_deg"] == pytest.approx(-7.23)
    assert obs["frequency_range"]["min_mhz"] == 900.0
    assert obs["frequency_range"]["max_mhz"] == 1700.0
    assert obs["frequency_range"]["n_channels"] == 4096
    assert obs["frequency_range"]["fc_mhz"] == pytest.approx(1300.0)
    assert obs["antennas"][0]["polarizations"] == ["V", "H"]


def test_assemble_config_satellites_block() -> None:
    sat = _config()["simulation"]["sky"]["satellites"]
    assert set(sat.keys()) == {"GPS", "GALILEO"}
    assert sat["GPS"]["n_satellites"] == 1
    assert sat["GPS"]["ids"] == [1]
    assert len(sat["GPS"]["tles"]) == 1
    assert sat["GPS"]["tle_mean_epoch"] is not None


def test_format_summary_sections() -> None:
    txt = format_summary(_config())
    assert "OBSERVER" in txt
    assert "SIMULATION" in txt
    assert "COMPUTATION" in txt
    assert "Central Frequency" in txt
    assert "Timerange" in txt


def test_to_json_serializable() -> None:
    payload = to_json(_config())
    assert "observer" in payload


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


def _dataset_with_angle_boresight() -> xr.Dataset:
    n_time, n_freq, n_src = 12, 8, 2
    time = np.arange(
        np.datetime64("2026-07-26T00:00:00", "ns"),
        np.datetime64("2026-07-26T00:00:00", "ns") + n_time * np.timedelta64(600, "s"),
        np.timedelta64(600, "s"),
    )
    freq = np.linspace(900e6, 1700e6, n_freq)
    T = np.ones((n_time, n_freq, n_src, 1, 1)) * 5.0
    angles = np.full((n_time, n_src, 1), np.deg2rad(10.0))
    angles[2:5, 0, 0] = np.deg2rad(1.0)  # SAT-1 no feixe
    angles[6:9, 1, 0] = np.deg2rad(1.0)  # Sun no feixe
    ds = xr.Dataset(
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
            "angle_boresight": (("time", "source", "pointing"), angles),
        },
    )
    return ds


def test_from_dataset_uses_angle_boresight_threshold() -> None:
    from radioskysims_simulation.domain.simulation.simulation import SimulationEvents

    ds = _dataset_with_angle_boresight()
    events = SimulationEvents.from_dataset(
        dataset=ds,
        time_range=TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T02:00:00", resolution=600),
        fwhm_threshold_deg=2.0,
    )

    assert set(events.objects_in_beam) == {"SAT-1", "Sun"}
    assert len(events.beam_events.coords["time"]) == 6
    assert events.beam_events["T_antenna"].isel(time=0).sum() > 0
    assert events.beam_events["T_antenna"].isel(time=0, source=1).sum() == 0
    assert events.beam_events["T_antenna"].isel(time=5, source=0).sum() == 0


def test_populate_computation_materialized() -> None:
    cfg = _config()
    result = SimulationResult(
        dataset=_dataset(),
        sources=[],
        satellites=[],
        time_range=TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T02:00:00", resolution=600),
    )
    populate_computation(result, cfg)
    comp = cfg["computation"]
    assert comp["delayed"] is False
    assert comp["dimensions"]["time"] == 12
    assert comp["dimensions"]["source"] == 2
    assert comp["size_mb"] is not None
    assert comp["memory"]["loaded_mb"] is not None


def test_populate_sky_stats() -> None:
    cfg = _config()
    result = SimulationResult(
        dataset=_dataset(),
        sources=[],
        satellites=[],
        time_range=TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T02:00:00", resolution=600),
    )
    populate_sky_stats(result, cfg)
    sun = cfg["simulation"]["sky"]["localsources"]["Sun"]
    assert sun["t_antenna_k"]["max"] == pytest.approx(20.0)
    # Sem fontes de rádio no dataset → bloco permanece sem estatística
    radio = cfg["simulation"]["sky"]["radiosources"]
    assert radio["t_antenna_k"] is None


def test_result_summary_and_repr() -> None:
    tr = TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T02:00:00", resolution=600)
    result = SimulationResult(
        dataset=_dataset(),
        sources=[],
        satellites=[],
        time_range=tr,
    )
    orch_result = SimulationOrchestratorResult(simulation_result=result, config=_config())
    summary = orch_result.summary()
    assert summary["observer"]["name"] == "bingo"
    # Dataset materializado → estatísticas de céu preenchidas
    assert summary["simulation"]["sky"]["localsources"]["Sun"]["t_antenna_k"][
        "max"
    ] == pytest.approx(20.0)
    assert "OBSERVER" in repr(orch_result)


def test_simulation_result_repr_is_short_and_contains_dims() -> None:
    tr = TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T02:00:00", resolution=600)
    result = SimulationResult(
        dataset=_dataset(),
        sources=[],
        satellites=[],
        time_range=tr,
    )

    txt = repr(result)
    assert txt.startswith("SimulationResult(")
    assert "dims={" in txt
    assert "backend='numpy'" in txt
    assert "Data variables:" not in txt
    assert "<xarray.Dataset>" not in txt


def test_simulation_result_repr_detects_dask_without_compute(monkeypatch) -> None:
    import dask.array as da

    tr = TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T02:00:00", resolution=600)
    ds_lazy = _dataset().chunk({"time": 4, "frequency": 4})
    result = SimulationResult(
        dataset=ds_lazy,
        sources=[],
        satellites=[],
        time_range=tr,
    )

    called = {"compute": False}

    def _fail_compute(_self, *_args, **_kwargs):
        called["compute"] = True
        msg = "repr must not call compute"
        raise RuntimeError(msg)

    monkeypatch.setattr(xr.Dataset, "compute", _fail_compute)

    txt = repr(result)
    assert "backend='dask'" in txt
    assert called["compute"] is False
    assert isinstance(result.dataset["T_antenna"].data, da.Array)


def test_simulation_result_repr_handles_missing_optional_coords() -> None:
    tr = TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T00:10:00", resolution=600)
    ds = xr.Dataset(
        data_vars={"x": ("sample", np.array([1.0, 2.0], dtype=float))},
        coords={"sample": [0, 1]},
    )

    result = SimulationResult(
        dataset=ds,
        sources=[],
        satellites=[],
        time_range=tr,
    )

    txt = repr(result)
    assert "SimulationResult(" in txt
    assert "dims={'sample': 2}" in txt
    assert "time='" not in txt
    assert "freq='" not in txt


def _truncated_source_type_dataset() -> xr.Dataset:
    """Reproduz a coord ``source_type`` truncada (<U9) de arquivos salvos."""
    ds = _dataset()
    return ds.assign_coords(
        source_type=("source", np.asarray(["satellite", "localsour"], dtype="<U9"))
    )


def test_populate_sky_stats_tolerates_truncated_source_type() -> None:
    cfg = _config()
    result = SimulationResult(
        dataset=_truncated_source_type_dataset(),
        sources=[],
        satellites=[],
        time_range=TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T02:00:00", resolution=600),
    )
    populate_sky_stats(result, cfg)
    sun = cfg["simulation"]["sky"]["localsources"]["Sun"]
    assert sun["t_antenna_k"]["max"] == pytest.approx(20.0)


def test_populate_sky_stats_sun_fallback_without_source_type() -> None:
    cfg = _config()
    result = SimulationResult(
        dataset=_dataset().drop_vars("source_type"),
        sources=[],
        satellites=[],
        time_range=TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T02:00:00", resolution=600),
    )
    populate_sky_stats(result, cfg)
    sun = cfg["simulation"]["sky"]["localsources"]["Sun"]
    assert sun["t_antenna_k"]["max"] == pytest.approx(20.0)
