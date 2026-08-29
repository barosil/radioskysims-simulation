from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from radioskysims_simulation.application.mapmaking import (
    GenerateSkyMapsFromTODUseCase,
    MapMakingResultSet,
)
from radioskysims_simulation.application.ports.mapmaking_repository import (
    MapMakingRepositoryPort,
)


@dataclass(frozen=True, slots=True)
class MapMakingServiceResult:
    """
    Resultado de alto nível do serviço de mapmaking.
    """

    mapmaking: MapMakingResultSet
    record_ids: tuple[int, ...]


class MapMakingService:
    """
    Application service para reconstrução e persistência de mapas.

    Fluxo:

        TOD
        -> GenerateSkyMapsFromTODUseCase
        -> MapMakingResultSet
        -> repository opcional
    """

    def __init__(
        self,
        *,
        generator: GenerateSkyMapsFromTODUseCase,
        repository: MapMakingRepositoryPort | None = None,
    ) -> None:
        self._generator = generator
        self._repository = repository

    def run(
        self,
        *,
        tod: xr.Dataset,
        pointing_track: xr.Dataset,
        nside: int,
        frame: str = "equatorial",
        persist: bool = False,
        simulation_id: int | None = None,
        key_prefix: str = "mapmaking",
        overwrite: bool = False,
    ) -> MapMakingServiceResult:
        result_set = self._generator.execute(
            tod=tod,
            pointing_track=pointing_track,
            nside=nside,
            frame=frame,
        )

        record_ids: list[int] = []

        if persist:
            if self._repository is None:
                raise RuntimeError("MapMakingService requires a repository when persist=True.")

            for index, result in enumerate(result_set.results):
                frequency_ghz = result.skymap.freq_ghz

                if frequency_ghz is None:
                    frequency_label = f"channel_{index}"
                else:
                    frequency_mhz = int(round(frequency_ghz * 1000.0))

                    frequency_label = f"{frequency_mhz}mhz"

                key = f"{key_prefix}_{frequency_label}_nside{result.skymap.nside_value}"

                record_id = self._repository.save(
                    result=result,
                    key=key,
                    simulation_id=simulation_id,
                    overwrite=overwrite,
                )

                record_ids.append(record_id)

        return MapMakingServiceResult(
            mapmaking=result_set,
            record_ids=tuple(record_ids),
        )
