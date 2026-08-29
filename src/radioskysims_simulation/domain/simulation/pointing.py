"""
Módulo: src/radioskysims/domain/simulation/pointing.py

Descrição:
    Produção do vetor de apontamento (pointing track) do
    radiotelescópio, que varia no tempo.

    Dado um ``Antenna`` com pointing fixo em alt-az e um
    ``Observer``, calcula as coordenadas equatoriais (RA/Dec)
    apontadas em cada instante do ``TimeRange``, resultando
    num dataset análogo aos tracks das fontes.

    Isto modela corretamente que, mesmo com montagem alt-az
    fixa, o RA/Dec apontado varia com a rotação da Terra.

Copyright BINGO Collaboration
Last modified: 2026-07-28
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from radioskysims_core.domain.common import Observer, TimeRange
    from radioskysims_radiotelescope.domain.radiotelescope.antenna.models import Antenna


def build_pointing_track(
    antenna: Antenna,
    time_range: TimeRange,
    observer: Observer,
    *,
    n_samples: int | None = None,
) -> xr.Dataset:
    """Produz o dataset de pointing variável no tempo.

    Para uma antena com pointing definido em alt-az fixo (ex.:
    montagem tipo BINGO), calcula as coordenadas equatoriais
    (RA/Dec) apontadas em cada instante usando a transformação
    do Skyfield::

        observer.at(t).from_altaz(alt_degrees=alt, az_degrees=az)

    O dataset resultante tem as mesmas dimensões (time,) que um
    track de fonte, permitindo cálculo de ângulo de boresight
    ponto-a-ponto.

    Args:
        antenna: Antena com pointing (alt_deg, az_deg).
        time_range: Intervalo temporal da simulação.
        observer: Observador terrestre.
        n_samples: Número de amostras (opcional; usa resolução
            do TimeRange por padrão).

    Returns:
        xr.Dataset com variáveis:
            - ra (rad): ascensão reta apontada.
            - dec (rad): declinação apontada.
            - alt (rad): altitude (constante, rad).
            - az (rad): azimute (constante, rad).
        Dimensão: (time,).
    """
    # Import skyfield sob demanda (dependência opcional)
    import skyfield.api as sf

    alt_deg = getattr(antenna.pointing, "alt_deg", 90.0)
    az_deg = getattr(antenna.pointing, "az_deg", 0.0)

    # Converte TimeRange para skyfield times
    times_sf = time_range.to_skyfield()

    # Converte observador para skyfield
    obs_lat = float(observer.lat.value) if hasattr(observer.lat, "value") else float(observer.lat)
    obs_lon = float(observer.lon.value) if hasattr(observer.lon, "value") else float(observer.lon)
    wgs84 = sf.wgs84
    obs_geo = wgs84.latlon(obs_lat, obs_lon)

    # Cria um earth-satellite dummy no centro da Terra para
    # obter o referencial geocêntrico
    ts = sf.load.timescale()
    # Melhor: criar um observador e usar from_altaz diretamente
    observer_pos = obs_geo.at(times_sf)

    # Para cada instante, computa RA/Dec da direção (alt, az)
    # no referencial do observador
    ra_list = []
    dec_list = []

    for i in range(len(times_sf)):
        # Cria vetor direção em alt-az
        direction = observer_pos[i].from_altaz(alt_degrees=alt_deg, az_degrees=az_deg)
        # Converte para RA/Dec (aparente, época de data)
        ra, dec, _ = direction.radec()
        ra_list.append(ra.radians)
        dec_list.append(dec.radians)

    ra_arr = np.array(ra_list, dtype=float)
    dec_arr = np.array(dec_list, dtype=float)

    # Cria array de tempo no formato usado por tracks
    try:
        time_coord = np.asarray(times_sf.tt, dtype=float)
    except Exception:
        time_coord = np.arange(len(times_sf), dtype=float)

    alt_arr = np.full_like(ra_arr, np.deg2rad(alt_deg))
    az_arr = np.full_like(ra_arr, np.deg2rad(az_deg))

    return xr.Dataset(
        data_vars={
            "ra": ("time", ra_arr),
            "dec": ("time", dec_arr),
            "alt": ("time", alt_arr),
            "az": ("time", az_arr),
        },
        coords={"time": time_coord},
        attrs={
            "pointing_alt_deg": alt_deg,
            "pointing_az_deg": az_deg,
            "antenna_name": getattr(antenna, "name", "unknown"),
            "description": "Pointing track (altaz -> RA/Dec via Skyfield)",
        },
    )


def build_pointing_track_vectorized(
    antenna: Antenna,
    time_range: TimeRange,
    observer: Observer,
) -> xr.Dataset:
    """Versão vetorizada (mais rápida) de ``build_pointing_track``.

    Em vez de iterar timestep a timestep, usa a API de array
    do Skyfield para computar RA/Dec de forma vetorizada.
    """
    import skyfield.api as sf

    alt_deg = getattr(antenna.pointing, "alt_deg", 90.0)
    az_deg = getattr(antenna.pointing, "az_deg", 0.0)

    times_sf = time_range.to_skyfield()

    obs_lat = float(observer.lat.value) if hasattr(observer.lat, "value") else float(observer.lat)
    obs_lon = float(observer.lon.value) if hasattr(observer.lon, "value") else float(observer.lon)
    obs_geo = sf.wgs84.latlon(obs_lat, obs_lon)

    # Posição do observador em cada instante
    observer_pos = obs_geo.at(times_sf)

    # Vetor direção a partir de alt-az
    direction = observer_pos.from_altaz(alt_degrees=alt_deg, az_degrees=az_deg)

    # RA/Dec em radianos
    ra, dec, _ = direction.radec()

    ra_arr = np.asarray(ra.radians, dtype=float)
    dec_arr = np.asarray(dec.radians, dtype=float)

    try:
        time_coord = np.asarray(times_sf.tt, dtype=float)
    except Exception:
        time_coord = np.arange(len(times_sf), dtype=float)

    alt_arr = np.full_like(ra_arr, np.deg2rad(alt_deg))
    az_arr = np.full_like(ra_arr, np.deg2rad(az_deg))

    return xr.Dataset(
        data_vars={
            "ra": ("time", ra_arr),
            "dec": ("time", dec_arr),
            "alt": ("time", alt_arr),
            "az": ("time", az_arr),
        },
        coords={"time": time_coord},
        attrs={
            "pointing_alt_deg": alt_deg,
            "pointing_az_deg": az_deg,
            "antenna_name": getattr(antenna, "name", "unknown"),
            "description": "Pointing track (altaz -> RA/Dec via Skyfield, vectorized)",
        },
    )
