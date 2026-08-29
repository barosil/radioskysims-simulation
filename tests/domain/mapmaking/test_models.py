from __future__ import annotations

import numpy as np
import pytest

from radioskysims_simulation.domain.mapmaking import (
    MapMakingResult,
)
from radioskysims_skymap.domain.skymap import SkyMap


def _skymap() -> SkyMap:
    nside = 2
    npix = 12 * nside**2

    return SkyMap(
        dataset=np.zeros(
            npix,
            dtype=float,
        ),
        nside=nside,
        unit="K",
        frame="equatorial",
        freq_ghz=1.0,
        field="I",
    )


def test_mapmaking_result_is_valid() -> None:
    skymap = _skymap()

    hit_map = np.zeros(
        skymap.npix,
        dtype=np.int64,
    )

    hit_map[:3] = [2, 1, 4]

    result = MapMakingResult(
        skymap=skymap,
        hit_map=hit_map,
        observed_mask=hit_map > 0,
        n_samples=7,
    )

    assert result.n_observed_pixels == 3

    assert result.coverage_fraction == pytest.approx(3 / skymap.npix)


def test_mapmaking_result_rejects_wrong_hit_map_size() -> None:
    skymap = _skymap()

    with pytest.raises(
        ValueError,
        match="hit_map size",
    ):
        MapMakingResult(
            skymap=skymap,
            hit_map=np.zeros(
                5,
                dtype=np.int64,
            ),
            observed_mask=np.zeros(
                skymap.npix,
                dtype=bool,
            ),
            n_samples=1,
        )


def test_mapmaking_result_rejects_inconsistent_mask() -> None:
    skymap = _skymap()

    hits = np.zeros(
        skymap.npix,
        dtype=np.int64,
    )

    hits[0] = 1

    with pytest.raises(
        ValueError,
        match="observed_mask",
    ):
        MapMakingResult(
            skymap=skymap,
            hit_map=hits,
            observed_mask=np.zeros(
                skymap.npix,
                dtype=bool,
            ),
            n_samples=1,
        )
