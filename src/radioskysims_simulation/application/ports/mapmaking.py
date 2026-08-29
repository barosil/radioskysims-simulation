from __future__ import annotations

from typing import (
    Protocol,
    runtime_checkable,
)

import xarray as xr

from radioskysims_simulation.domain.mapmaking import (
    MapMakingResult,
)


@runtime_checkable
class MapMakerPort(Protocol):
    """
    Port para reconstrução de mapas HEALPix a partir de TOD.

    Implementações concretas pertencem à camada Infrastructure.
    """

    def make_map(
        self,
        *,
        tod: xr.Dataset,
        pointing_track: xr.Dataset,
        frequency_hz: float,
        nside: int,
        frame: str = "equatorial",
    ) -> MapMakingResult:
        """
        Reconstrói um mapa para um único canal de frequência.
        """
        ...
