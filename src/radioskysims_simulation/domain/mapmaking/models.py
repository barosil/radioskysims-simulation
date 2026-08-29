from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from radioskysims_skymap.domain.skymap import SkyMap


@dataclass(frozen=True, slots=True)
class MapMakingResult:
    """
    Resultado da reconstrução de um mapa HEALPix a partir de TOD.

    Parameters
    ----------
    skymap
        Mapa científico reconstruído.
    hit_map
        Número de amostras TOD acumuladas em cada pixel HEALPix.
    observed_mask
        Máscara booleana indicando pixels observados.
    n_samples
        Número total de amostras temporais processadas.
    """

    skymap: SkyMap
    hit_map: NDArray[np.int64]
    observed_mask: NDArray[np.bool_]
    n_samples: int

    def __post_init__(self) -> None:
        hit_map = np.asarray(
            self.hit_map,
            dtype=np.int64,
        )

        observed_mask = np.asarray(
            self.observed_mask,
            dtype=np.bool_,
        )

        if hit_map.ndim != 1:
            raise ValueError("hit_map must be a 1D HEALPix array.")

        if observed_mask.ndim != 1:
            raise ValueError("observed_mask must be a 1D array.")

        if hit_map.shape != (self.skymap.npix,):
            raise ValueError("hit_map size must match SkyMap npix.")

        if observed_mask.shape != (self.skymap.npix,):
            raise ValueError("observed_mask size must match SkyMap npix.")

        if np.any(hit_map < 0):
            raise ValueError("hit_map cannot contain negative values.")

        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        expected_mask = hit_map > 0

        if not np.array_equal(
            observed_mask,
            expected_mask,
        ):
            raise ValueError("observed_mask must be equivalent to hit_map > 0.")

        object.__setattr__(
            self,
            "hit_map",
            hit_map.copy(),
        )

        object.__setattr__(
            self,
            "observed_mask",
            observed_mask.copy(),
        )

    @property
    def n_observed_pixels(self) -> int:
        return int(np.count_nonzero(self.observed_mask))

    @property
    def coverage_fraction(self) -> float:
        return self.n_observed_pixels / self.skymap.npix
