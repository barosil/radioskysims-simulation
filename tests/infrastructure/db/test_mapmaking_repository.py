from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine
from sqlmodel import Session

from radioskysims_simulation.domain.mapmaking import (
    MapMakingResult,
)
from radioskysims_skymap.domain.skymap import SkyMap
from radioskysims_simulation.infrastructure.db.mapmaking_db_model import (
    MapMakingRecord,
)
from radioskysims_simulation.infrastructure.db.mapmaking_repository import (
    MapMakingRepository,
)


def _result() -> MapMakingResult:
    nside = 2
    npix = 12 * nside**2

    pixels = np.zeros(
        npix,
        dtype=float,
    )

    hits = np.zeros(
        npix,
        dtype=np.int64,
    )

    pixels[0] = 1.5
    pixels[1] = 3.5

    hits[0] = 2
    hits[1] = 2

    skymap = SkyMap(
        dataset=pixels,
        nside=nside,
        unit="K",
        frame="equatorial",
        freq_ghz=1.0,
        field="I",
    )

    return MapMakingResult(
        skymap=skymap,
        hit_map=hits,
        observed_mask=hits > 0,
        n_samples=4,
    )


def _repository(
    tmp_path,
) -> MapMakingRepository:
    engine = create_engine(
        "sqlite://",
    )

    MapMakingRecord.metadata.create_all(engine)

    return MapMakingRepository(
        root=tmp_path,
        session_factory=lambda: Session(engine),
    )


def test_repository_round_trip(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)

    original = _result()

    record_id = repository.save(
        result=original,
        key="test_map",
        simulation_id=10,
    )

    loaded = repository.load(record_id)

    assert loaded is not None

    np.testing.assert_allclose(
        loaded.skymap.data,
        original.skymap.data,
    )

    np.testing.assert_array_equal(
        loaded.hit_map,
        original.hit_map,
    )

    assert loaded.n_samples == 4

    assert loaded.n_observed_pixels == original.n_observed_pixels


def test_repository_get_metadata_does_not_require_map_load(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)

    record_id = repository.save(
        result=_result(),
        key="metadata_map",
        simulation_id=99,
    )

    metadata = repository.get_metadata(record_id)

    assert metadata is not None

    assert metadata["simulation_id"] == 99

    assert metadata["frequency_hz"] == 1.0e9

    assert metadata["n_samples"] == 4


def test_repository_delete_removes_record_and_file(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)

    record_id = repository.save(
        result=_result(),
        key="delete_map",
    )

    metadata = repository.get_metadata(record_id)

    assert metadata is not None

    filepath = metadata["filepath"]

    assert repository.delete(record_id)

    assert repository.get_metadata(record_id) is None

    from pathlib import Path

    assert not Path(str(filepath)).exists()
