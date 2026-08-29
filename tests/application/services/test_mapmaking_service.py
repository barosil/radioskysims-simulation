from __future__ import annotations

import numpy as np
import xarray as xr

from radioskysims_simulation.application.mapmaking import (
    GenerateSkyMapsFromTODUseCase,
)
from radioskysims_simulation.application.services.mapmaking_service import (
    MapMakingService,
)
from radioskysims_simulation.infrastructure.mapmaking import (
    HealpyBinnedMapMakerAdapter,
)


class FakeRepository:
    def __init__(self) -> None:
        self.saved: list[dict[str, object]] = []

    def save(
        self,
        *,
        result,
        key: str,
        simulation_id: int | None = None,
        overwrite: bool = False,
    ) -> int:
        self.saved.append(
            {
                "result": result,
                "key": key,
                "simulation_id": simulation_id,
                "overwrite": overwrite,
            }
        )

        return len(self.saved)

    def load(self, record_id: int):
        return None

    def get_metadata(self, record_id: int):
        return None

    def delete(self, record_id: int) -> bool:
        return False


def _tod() -> xr.Dataset:
    time = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    return xr.Dataset(
        {
            "T_antenna": (
                (
                    "time",
                    "frequency",
                ),
                np.array(
                    [
                        [1.0, 10.0],
                        [3.0, 30.0],
                    ]
                ),
            )
        },
        coords={
            "time": time,
            "frequency": [
                1.0e9,
                1.1e9,
            ],
        },
    )


def _pointing() -> xr.Dataset:
    time = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    return xr.Dataset(
        {
            "ra": (
                "time",
                np.deg2rad(
                    [
                        0.0,
                        90.0,
                    ]
                ),
            ),
            "dec": (
                "time",
                np.deg2rad(
                    [
                        0.0,
                        0.0,
                    ]
                ),
            ),
        },
        coords={
            "time": time,
        },
    )


def test_service_generates_maps_without_persistence() -> None:
    service = MapMakingService(
        generator=GenerateSkyMapsFromTODUseCase(
            mapmaker=HealpyBinnedMapMakerAdapter(),
        )
    )

    result = service.run(
        tod=_tod(),
        pointing_track=_pointing(),
        nside=8,
    )

    assert len(result.mapmaking.results) == 2

    assert result.record_ids == ()


def test_service_persists_each_frequency() -> None:
    repository = FakeRepository()

    service = MapMakingService(
        generator=GenerateSkyMapsFromTODUseCase(
            mapmaker=HealpyBinnedMapMakerAdapter(),
        ),
        repository=repository,
    )

    result = service.run(
        tod=_tod(),
        pointing_track=_pointing(),
        nside=8,
        persist=True,
        simulation_id=42,
        key_prefix="bingo",
    )

    assert result.record_ids == (
        1,
        2,
    )

    assert len(repository.saved) == 2

    assert repository.saved[0]["simulation_id"] == 42

    assert repository.saved[0]["key"] == "bingo_1000mhz_nside8"

    assert repository.saved[1]["key"] == "bingo_1100mhz_nside8"
