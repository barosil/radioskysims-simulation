from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_simulation.application.ports.mapmaking import (
    MapMakerPort,
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
        [1.0e9],
        dtype=float,
    )

    temperature = xr.DataArray(
        np.array(
            [
                [1.0],
                [2.0],
                [3.0],
                [4.0],
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


def test_adapter_implements_mapmaker_port() -> None:
    adapter = HealpyBinnedMapMakerAdapter()

    assert isinstance(
        adapter,
        MapMakerPort,
    )


def test_make_map_returns_valid_result() -> None:
    adapter = HealpyBinnedMapMakerAdapter()

    result = adapter.make_map(
        tod=_tod(),
        pointing_track=_pointing(),
        frequency_hz=1.0e9,
        nside=8,
    )

    assert result.skymap.nside_value == 8

    assert result.skymap.unit == "K"

    assert result.skymap.frame == "equatorial"

    assert result.skymap.freq_ghz == pytest.approx(1.0)

    assert result.n_samples == 4

    assert int(result.hit_map.sum()) == 4


def test_samples_in_same_pixel_are_averaged() -> None:
    adapter = HealpyBinnedMapMakerAdapter()

    result = adapter.make_map(
        tod=_tod(),
        pointing_track=_pointing(),
        frequency_hz=1.0e9,
        nside=8,
    )

    observed_values = result.skymap.data[result.observed_mask]

    assert result.n_observed_pixels == 2

    np.testing.assert_allclose(
        np.sort(observed_values),
        np.array(
            [
                1.5,
                3.5,
            ]
        ),
    )


def test_make_map_rejects_missing_temperature() -> None:
    adapter = HealpyBinnedMapMakerAdapter()

    with pytest.raises(
        ValueError,
        match="T_antenna",
    ):
        adapter.make_map(
            tod=xr.Dataset(),
            pointing_track=_pointing(),
            frequency_hz=1.0e9,
            nside=8,
        )


def test_make_map_accepts_pointing_coordinates_in_degrees() -> None:
    time = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
            "2026-07-26T00:02:00",
            "2026-07-26T00:03:00",
        ],
        dtype="datetime64[ns]",
    )

    frequency_hz = 1.0e9

    tod = xr.Dataset(
        data_vars={
            "T_antenna": (
                ("time", "frequency"),
                np.array(
                    [
                        [1.0],
                        [2.0],
                        [3.0],
                        [4.0],
                    ],
                    dtype=float,
                ),
            ),
        },
        coords={
            "time": time,
            "frequency": np.array([frequency_hz]),
        },
    )

    pointing_deg = xr.Dataset(
        data_vars={
            "ra": xr.DataArray(
                [0.0, 0.0, 90.0, 90.0],
                dims=("time",),
                coords={"time": time},
                attrs={"units": "deg"},
            ),
            "dec": xr.DataArray(
                [0.0, 0.0, 0.0, 0.0],
                dims=("time",),
                coords={"time": time},
                attrs={"units": "deg"},
            ),
        }
    )

    pointing_rad = xr.Dataset(
        data_vars={
            "ra": xr.DataArray(
                np.deg2rad([0.0, 0.0, 90.0, 90.0]),
                dims=("time",),
                coords={"time": time},
                attrs={"units": "rad"},
            ),
            "dec": xr.DataArray(
                np.deg2rad([0.0, 0.0, 0.0, 0.0]),
                dims=("time",),
                coords={"time": time},
                attrs={"units": "rad"},
            ),
        }
    )

    mapmaker = HealpyBinnedMapMakerAdapter()

    result_deg = mapmaker.make_map(
        tod=tod,
        pointing_track=pointing_deg,
        frequency_hz=frequency_hz,
        nside=8,
    )

    result_rad = mapmaker.make_map(
        tod=tod,
        pointing_track=pointing_rad,
        frequency_hz=frequency_hz,
        nside=8,
    )

    np.testing.assert_allclose(
        result_deg.skymap.data,
        result_rad.skymap.data,
    )

    np.testing.assert_array_equal(
        result_deg.hit_map,
        result_rad.hit_map,
    )
