from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from radioskysims_simulation.application.ports.mapmaking import (
    MapMakerPort,
)
from radioskysims_simulation.domain.mapmaking import (
    MapMakingResult,
)


@dataclass(frozen=True, slots=True)
class MapMakingResultSet:
    """
    Conjunto de mapas reconstruídos, um por canal de frequência.
    """

    results: tuple[MapMakingResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("MapMakingResultSet requires at least one result.")

    @property
    def maps(self):
        return tuple(result.skymap for result in self.results)

    @property
    def frequencies_ghz(self) -> tuple[float, ...]:
        return tuple(
            float(result.skymap.freq_ghz)
            for result in self.results
            if result.skymap.freq_ghz is not None
        )


class GenerateSkyMapsFromTODUseCase:
    """
    Reconstrói um mapa HEALPix para cada canal de frequência de um TOD.
    """

    def __init__(
        self,
        *,
        mapmaker: MapMakerPort,
    ) -> None:
        self._mapmaker = mapmaker

    def execute(
        self,
        *,
        tod: xr.Dataset,
        pointing_track: xr.Dataset,
        nside: int,
        frame: str = "equatorial",
    ) -> MapMakingResultSet:
        if "frequency" not in tod.coords:
            raise ValueError("TOD dataset must contain frequency coordinates.")

        frequencies = np.asarray(
            tod.coords["frequency"].values,
            dtype=np.float64,
        )

        if frequencies.ndim != 1:
            raise ValueError("TOD frequency coordinate must be one-dimensional.")

        if frequencies.size == 0:
            raise ValueError("TOD must contain at least one frequency channel.")

        results: list[MapMakingResult] = []

        for frequency_hz in frequencies:
            result = self._mapmaker.make_map(
                tod=tod,
                pointing_track=pointing_track,
                frequency_hz=float(frequency_hz),
                nside=nside,
                frame=frame,
            )

            results.append(result)

        return MapMakingResultSet(
            results=tuple(results),
        )
