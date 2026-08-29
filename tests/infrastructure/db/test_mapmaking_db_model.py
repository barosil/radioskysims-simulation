from __future__ import annotations

from radioskysims_simulation.infrastructure.db.mapmaking_db_model import (
    MapMakingRecord,
)


def test_mapmaking_record_stores_metadata() -> None:
    record = MapMakingRecord(
        simulation_id=10,
        frequency_hz=1.0e9,
        nside=8,
        npix=768,
        unit="K",
        frame="equatorial",
        field="I",
        n_samples=1440,
        n_observed_pixels=120,
        coverage_fraction=120 / 768,
        filepath="/tmp/mapmaking_1000mhz.h5",
    )

    assert record.simulation_id == 10
    assert record.frequency_hz == 1.0e9
    assert record.nside == 8
    assert record.npix == 768

    assert record.n_samples == 1440
    assert record.n_observed_pixels == 120

    assert record.filepath.endswith(".h5")

    assert record.mapmaker == "healpy_binned"
