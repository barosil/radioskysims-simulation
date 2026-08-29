from __future__ import annotations

from typing import Protocol, runtime_checkable

import xarray as xr

from radioskysims_skymap.domain.skymap import SkyMap


@runtime_checkable
class SkyMapTODProjector(Protocol):
    """
    Contrato para projetar um SkyMap ao longo do pointing temporal.

    A implementação concreta é responsável por:

    - transformar o pointing para o frame do mapa;
    - aplicar, quando solicitado, a convolução instrumental;
    - amostrar/interpolar o mapa;
    - retornar a temperatura do céu ao longo do tempo.

    O domínio não depende diretamente de Healpy.
    """

    def project(
        self,
        *,
        skymap: SkyMap,
        pointing_track: xr.Dataset,
        fwhm_rad: float | None = None,
    ) -> xr.DataArray:
        """
        Retorna a contribuição do mapa ao longo do pointing.

        Returns
        -------
        xr.DataArray
            Array com dimensão ``time``.
        """
        ...
