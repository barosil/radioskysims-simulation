from __future__ import annotations

import numpy as np
import xarray as xr

from radioskysims_simulation.application.mapmaking import (
    GenerateSkyMapsFromTODUseCase,
)
from radioskysims_simulation.infrastructure.mapmaking import (
    HealpyBinnedMapMakerAdapter,
)


def _tod() -> xr.Dataset:
    time = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
            "2026-07-26T00:02:00",
            "2026-07-26T00:03:00",
        ],
        dtype="datetime64[ns]",
    )

    frequency = np.array(
        [
            1.0e9,
            1.1e9,
        ],
        dtype=float,
    )

    temperature = xr.DataArray(
        np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ],
            dtype=float,
        ),
        dims=(
            "time",
            "frequency",
        ),
        coords={
            "time": time,
            "frequency": frequency,
        },
        attrs={
            "units": "K",
        },
    )

    return xr.Dataset(
        data_vars={
            "T_antenna": temperature,
        }
    )


def _pointing() -> xr.Dataset:
    time = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
            "2026-07-26T00:02:00",
            "2026-07-26T00:03:00",
        ],
        dtype="datetime64[ns]",
    )

    return xr.Dataset(
        data_vars={
            "ra": xr.DataArray(
                np.deg2rad(
                    [
                        0.0,
                        0.0,
                        90.0,
                        90.0,
                    ]
                ),
                dims=("time",),
                coords={"time": time},
                attrs={"units": "rad"},
            ),
            "dec": xr.DataArray(
                np.deg2rad(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                dims=("time",),
                coords={"time": time},
                attrs={"units": "rad"},
            ),
        }
    )


def test_generate_maps_for_each_frequency() -> None:
    use_case = GenerateSkyMapsFromTODUseCase(
        mapmaker=HealpyBinnedMapMakerAdapter(),
    )

    result = use_case.execute(
        tod=_tod(),
        pointing_track=_pointing(),
        nside=8,
    )

    assert len(result.results) == 2
    assert len(result.maps) == 2

    assert result.maps[0].freq_ghz == 1.0
    assert result.maps[1].freq_ghz == 1.1


def test_each_channel_preserves_its_temperature_scale() -> None:
    use_case = GenerateSkyMapsFromTODUseCase(
        mapmaker=HealpyBinnedMapMakerAdapter(),
    )

    result = use_case.execute(
        tod=_tod(),
        pointing_track=_pointing(),
        nside=8,
    )

    first_values = np.sort(result.maps[0].data[result.results[0].observed_mask])

    second_values = np.sort(result.maps[1].data[result.results[1].observed_mask])

    np.testing.assert_allclose(
        first_values,
        [1.5, 3.5],
    )

    np.testing.assert_allclose(
        second_values,
        [15.0, 35.0],
    )


def test_hit_map_is_preserved_for_each_channel() -> None:
    use_case = GenerateSkyMapsFromTODUseCase(
        mapmaker=HealpyBinnedMapMakerAdapter(),
    )

    result = use_case.execute(
        tod=_tod(),
        pointing_track=_pointing(),
        nside=8,
    )

    for channel in result.results:
        assert channel.n_samples == 4
        assert int(channel.hit_map.sum()) == 4
        assert channel.n_observed_pixels == 2
