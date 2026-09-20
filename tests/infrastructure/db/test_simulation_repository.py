"""\
Módulo: tests/infrastructure/db/test_simulation_repository.py

Testes da conversão de resumo → SimulationRecord (lógica pura, sem DB)
e do ciclo de vida do repositório (save/load/flush/delete by_id) com
engine SQLite em memória (StaticPool).

Copyright BINGO Collaboration
Last modified: 2026-08-03
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from radioskysims_core.domain.common import FrequencyRange, Observer, TimeRange
from radioskysims_sources.domain.satellite.models import OrbitalElement, Satellite
from radioskysims_simulation.domain.simulation import (
    SimulationOrchestratorResult,
    SimulationResult,
)
from radioskysims_simulation.domain.simulation.summary import assemble_config
from radioskysims_simulation.infrastructure.db.simulation_repository import (
    SimulationRepository,
    _row_from_summary,
)


class _Antenna:
    def __init__(self) -> None:
        self.name = "horn01"
        self.pointing = type("P", (), {"alt_deg": 90.0, "az_deg": 0.0})()
        self.beam = type("B", (), {"polarizations": ["V", "H"]})()


def _config() -> dict:
    tr = TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-27T00:00:00", resolution=600)
    fr = FrequencyRange(min_mhz=900.0, max_mhz=1700.0, resolution_mhz=0.2)
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
        dimensions={"time": 145, "frequency": 4096, "source": 20},
        delayed=True,
    )


def test_row_from_summary() -> None:
    row = _row_from_summary(_config())
    assert row.telescope_name == "bingo"
    assert row.observer_lat_deg == pytest.approx(-7.23)
    assert row.freq_min_mhz == 900.0
    assert row.freq_max_mhz == 1700.0
    assert row.n_channels == 4096
    assert row.fc_mhz == pytest.approx(1300.0)
    assert row.n_satellites == 2
    assert row.time_start_utc is not None
    assert row.time_end_utc is not None
    assert "GPS" in row.constellations
    assert "GALILEO" in row.constellations
    assert row.config_json


def test_row_from_summary_status_flags() -> None:
    row = _row_from_summary(
        _config(),
        filepath="/tmp/sim.nc",
        saved=True,
        registered=True,
    )
    assert row.filepath == "/tmp/sim.nc"
    assert row.saved is True
    assert row.registered is True


# ── Ciclo de vida do repositório (SQLite em memória) ─────────


@pytest.fixture
def repo() -> SimulationRepository:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    repository = SimulationRepository(session_factory=lambda: Session(engine))
    yield repository
    engine.dispose()


def _dataset() -> xr.Dataset:
    n_time, n_freq, n_src = 12, 8, 1
    time = np.arange(
        np.datetime64("2026-07-26T00:00:00", "ns"),
        np.datetime64("2026-07-26T00:00:00", "ns") + n_time * np.timedelta64(600, "s"),
        np.timedelta64(600, "s"),
    )
    freq = np.linspace(900e6, 1700e6, n_freq)
    T = np.zeros((n_time, n_freq, n_src, 1, 1))
    T[:, :, 0, 0, 0] = 5.0
    return xr.Dataset(
        {"T_antenna": (("time", "frequency", "source", "polarization", "pointing"), T)},
        coords={
            "time": time,
            "frequency": freq,
            "source": ["SAT-1"],
            "source_type": ("source", ["satellite"]),
            "polarization": ["V"],
            "pointing": [0],
        },
    )


def _composite(tmp_path) -> tuple[SimulationOrchestratorResult, str]:
    """Cria um resultado composto e o persiste em arquivo (HDF5)."""
    tr = TimeRange.from_iso("2026-07-26T00:00:00", "2026-07-26T02:00:00", resolution=600)
    result = SimulationResult(
        dataset=_dataset(),
        sources=[],
        satellites=[],
        time_range=tr,
        provenance={"pipeline": "tod"},
    )
    filepath = str(tmp_path / "sim_result.nc")
    result.to_hdf5(filepath)
    orch = SimulationOrchestratorResult(simulation_result=result, config=_config())
    return orch, filepath


def test_repository_save_flush_delete_by_id(repo: SimulationRepository) -> None:
    orch, _ = _composite(Path("/tmp"))
    sim_id = repo.save(orch, registered=True)
    assert repo.current_id == sim_id
    assert repo.current is orch

    # flush descarta o item corrente (mas não remove do banco)
    repo.flush()
    assert repo.current is None
    assert repo.current_id is None
    assert repo.get(sim_id) is not None

    # delete por by_id
    assert repo.delete(by_id=sim_id) is True
    assert repo.get(sim_id) is None

    # delete de registro inexistente → False
    assert repo.delete(by_id=99999) is False

    # delete sem argumentos → erro
    with pytest.raises(ValueError):
        repo.delete()


def test_repository_load_roundtrip(repo: SimulationRepository, tmp_path) -> None:
    orch, filepath = _composite(tmp_path)
    sim_id = repo.save(orch, filepath=filepath, saved=True, registered=True)

    loaded = repo.load(sim_id)
    assert loaded is not None
    assert repo.current is loaded
    assert loaded.simulation_result is not None
    assert "T_antenna" in loaded.simulation_result.dataset
    # config recomposto a partir do resumo do banco
    assert loaded.config["observer"]["name"] == "bingo"
    assert loaded.config["computation"]["filepath"] == filepath

    # flush libera a referência
    repo.flush()
    assert repo.current is None


def test_repository_load_missing_file(repo: SimulationRepository, tmp_path) -> None:
    orch, filepath = _composite(tmp_path)
    sim_id = repo.save(orch, filepath=filepath, saved=True, registered=True)
    # Remove o arquivo: load deve recompôr o resumo mesmo assim
    Path(filepath).unlink()

    loaded = repo.load(sim_id)
    assert loaded is not None
    assert loaded.simulation_result is None
    assert loaded.config["observer"]["name"] == "bingo"


def test_repository_delete_with_file(repo: SimulationRepository, tmp_path) -> None:
    orch, filepath = _composite(tmp_path)
    sim_id = repo.save(orch, filepath=filepath, saved=True, registered=True)
    assert Path(filepath).exists()

    assert repo.delete(by_id=sim_id, delete_file=True) is True
    assert not Path(filepath).exists()
    assert repo.get(sim_id) is None


def _summary_for_search(
    *,
    telescope_name: str = "bingo",
    time_start: str = "2026-07-26T00:00:00",
    time_end: str = "2026-07-26T02:00:00",
    freq_min_mhz: float = 900.0,
    freq_max_mhz: float = 1700.0,
) -> dict:
    cfg = _config()
    cfg["observer"]["name"] = telescope_name
    cfg["simulation"]["time_range"]["start"] = time_start
    cfg["simulation"]["time_range"]["end"] = time_end
    cfg["observer"]["frequency_range"]["min_mhz"] = freq_min_mhz
    cfg["observer"]["frequency_range"]["max_mhz"] = freq_max_mhz
    return cfg


def _set_created_at(repo: SimulationRepository, sim_id: int, dt: datetime) -> None:
    from radioskysims_simulation.infrastructure.db.simulation_db_model import SimulationRecord

    with repo._session() as session:
        row = session.get(SimulationRecord, sim_id)
        assert row is not None
        row.created_at = dt
        session.add(row)
        session.commit()


def test_search_without_filters_returns_recent_first(repo: SimulationRepository) -> None:
    s1 = repo.save(_summary_for_search(telescope_name="scope-a"), saved=True, registered=True)
    s2 = repo.save(_summary_for_search(telescope_name="scope-b"), saved=False, registered=False)
    s3 = repo.save(_summary_for_search(telescope_name="scope-c"), saved=True, registered=False)

    base = datetime(2026, 8, 1, 12, 0, 0)
    _set_created_at(repo, s1, base)
    _set_created_at(repo, s2, base + timedelta(minutes=1))
    _set_created_at(repo, s3, base + timedelta(minutes=2))

    rows = repo.search()
    ids = [r["_id"] for r in rows]
    assert ids == [s3, s2, s1]


def test_search_created_at_interval(repo: SimulationRepository) -> None:
    s1 = repo.save(_summary_for_search(telescope_name="scope-a"))
    s2 = repo.save(_summary_for_search(telescope_name="scope-b"))
    s3 = repo.save(_summary_for_search(telescope_name="scope-c"))

    base = datetime(2026, 8, 2, 10, 0, 0)
    _set_created_at(repo, s1, base)
    _set_created_at(repo, s2, base + timedelta(minutes=10))
    _set_created_at(repo, s3, base + timedelta(minutes=20))

    rows = repo.search(
        created_from=base + timedelta(minutes=5),
        created_to=base + timedelta(minutes=15),
    )
    assert [r["_id"] for r in rows] == [s2]


def test_search_observation_time_overlap_interval(repo: SimulationRepository) -> None:
    s1 = repo.save(
        _summary_for_search(
            telescope_name="scope-a",
            time_start="2026-07-26T00:00:00",
            time_end="2026-07-26T01:00:00",
        )
    )
    s2 = repo.save(
        _summary_for_search(
            telescope_name="scope-b",
            time_start="2026-07-26T02:00:00",
            time_end="2026-07-26T03:00:00",
        )
    )
    _ = s1

    rows = repo.search(
        time_start="2026-07-26T00:30:00",
        time_end="2026-07-26T00:45:00",
    )
    assert [r["_id"] for r in rows] == [s1]

    rows2 = repo.search(
        time_start="2026-07-26T01:30:00",
        time_end="2026-07-26T01:45:00",
    )
    assert [r["_id"] for r in rows2] == []
    assert repo.get(s2) is not None


def test_search_by_telescope_name(repo: SimulationRepository) -> None:
    repo.save(_summary_for_search(telescope_name="bingo"))
    target_id = repo.save(_summary_for_search(telescope_name="other_scope"))

    rows = repo.search(telescope_name="other_scope")
    assert [r["_id"] for r in rows] == [target_id]


def test_search_frequency_overlap(repo: SimulationRepository) -> None:
    s1 = repo.save(_summary_for_search(freq_min_mhz=900.0, freq_max_mhz=1100.0))
    s2 = repo.save(_summary_for_search(freq_min_mhz=1200.0, freq_max_mhz=1400.0))
    s3 = repo.save(_summary_for_search(freq_min_mhz=1500.0, freq_max_mhz=1700.0))

    rows = repo.search(frequency_min_mhz=1050.0, frequency_max_mhz=1250.0)
    ids = {r["_id"] for r in rows}
    assert ids == {s1, s2}
    assert s3 not in ids


def test_search_saved_registered_filters(repo: SimulationRepository) -> None:
    a = repo.save(_summary_for_search(telescope_name="scope-a"), saved=True, registered=True)
    b = repo.save(_summary_for_search(telescope_name="scope-b"), saved=True, registered=False)
    c = repo.save(_summary_for_search(telescope_name="scope-c"), saved=False, registered=False)

    rows = repo.search(saved=True)
    assert {r["_id"] for r in rows} == {a, b}

    rows2 = repo.search(registered=False)
    assert {r["_id"] for r in rows2} == {b, c}


def test_search_combined_filters(repo: SimulationRepository) -> None:
    repo.save(
        _summary_for_search(
            telescope_name="bingo",
            time_start="2026-07-26T00:00:00",
            time_end="2026-07-26T01:00:00",
            freq_min_mhz=900.0,
            freq_max_mhz=1100.0,
        ),
        saved=True,
        registered=True,
    )
    target = repo.save(
        _summary_for_search(
            telescope_name="bingo",
            time_start="2026-07-26T00:30:00",
            time_end="2026-07-26T01:30:00",
            freq_min_mhz=1000.0,
            freq_max_mhz=1300.0,
        ),
        saved=True,
        registered=True,
    )
    repo.save(
        _summary_for_search(
            telescope_name="other",
            time_start="2026-07-26T00:30:00",
            time_end="2026-07-26T01:30:00",
            freq_min_mhz=1000.0,
            freq_max_mhz=1300.0,
        ),
        saved=True,
        registered=True,
    )

    rows = repo.search(
        telescope_name="bingo",
        time_start="2026-07-26T00:45:00",
        time_end="2026-07-26T00:50:00",
        frequency_min_mhz=1150.0,
        frequency_max_mhz=1250.0,
        saved=True,
        registered=True,
    )
    assert [r["_id"] for r in rows] == [target]


def test_search_limit_offset_and_validation(repo: SimulationRepository) -> None:
    ids = [repo.save(_summary_for_search(telescope_name=f"scope-{i}")) for i in range(5)]
    # IDs crescem com inserção; ordenação padrão é created_at desc/id desc.
    expected_desc = list(reversed(ids))

    page = repo.search(limit=2, offset=1)
    assert [r["_id"] for r in page] == expected_desc[1:3]

    with pytest.raises(ValueError, match="limit deve ser > 0"):
        repo.search(limit=0)

    with pytest.raises(ValueError, match="offset deve ser >= 0"):
        repo.search(offset=-1)


def test_search_empty_result(repo: SimulationRepository) -> None:
    repo.save(_summary_for_search(telescope_name="bingo"))
    rows = repo.search(telescope_name="nao-existe")
    assert rows == []


def test_search_does_not_load_external_dataset(repo: SimulationRepository, monkeypatch) -> None:
    repo.save(_summary_for_search(), filepath="/tmp/never-load-me.nc", saved=True, registered=True)

    from radioskysims_simulation.domain.simulation.simulation import SimulationResult as _SR

    def _fail_load(*_args, **_kwargs):
        msg = "search nao deve carregar datasets externos"
        raise AssertionError(msg)

    monkeypatch.setattr(_SR, "load", _fail_load)

    rows = repo.search(saved=True)
    assert len(rows) == 1
