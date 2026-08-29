from __future__ import annotations

import healpy as hp
import numpy as np
import pytest
import xarray as xr

from radioskysims_skymap.domain.skymap import NSide, SkyMap
from radioskysims_simulation.infrastructure.skymap.tod_projector import (
    HealpySkyMapTODProjector,
)


def _constant_map(
    value: float = 10.0,
    *,
    nside: int = 16,
    frame: str = "galactic",
) -> SkyMap:
    return SkyMap(
        dataset=np.full(
            hp.nside2npix(nside),
            value,
            dtype=np.float64,
        ),
        nside=NSide(nside),
        unit="uK_CMB",
        frame=frame,
        freq_ghz=1.0,
    )


def _pointing_track() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "ra": (
                "time",
                np.deg2rad(
                    np.array(
                        [
                            0.0,
                            30.0,
                            90.0,
                        ]
                    )
                ),
            ),
            "dec": (
                "time",
                np.deg2rad(
                    np.array(
                        [
                            0.0,
                            10.0,
                            -20.0,
                        ]
                    )
                ),
            ),
        },
        coords={
            "time": np.array(
                [
                    0.0,
                    1.0,
                    2.0,
                ]
            )
        },
    )


def test_project_constant_galactic_map():
    projector = HealpySkyMapTODProjector()

    result = projector.project(
        skymap=_constant_map(),
        pointing_track=_pointing_track(),
    )

    assert result.dims == ("time",)
    assert result.shape == (3,)

    np.testing.assert_allclose(
        result.values,
        10.0,
        rtol=0.0,
        atol=1e-10,
    )


def test_project_preserves_time_coordinate():
    projector = HealpySkyMapTODProjector()

    pointing = _pointing_track()

    result = projector.project(
        skymap=_constant_map(),
        pointing_track=pointing,
    )

    np.testing.assert_array_equal(
        result.coords["time"].values,
        pointing.coords["time"].values,
    )


def test_project_equatorial_map():
    projector = HealpySkyMapTODProjector()

    result = projector.project(
        skymap=_constant_map(
            frame="equatorial",
        ),
        pointing_track=_pointing_track(),
    )

    np.testing.assert_allclose(
        result.values,
        10.0,
        atol=1e-10,
    )


def test_project_with_gaussian_beam():
    projector = HealpySkyMapTODProjector()

    result = projector.project(
        skymap=_constant_map(),
        pointing_track=_pointing_track(),
        fwhm_rad=np.deg2rad(1.0),
    )

    # Convolução de um campo constante por beam normalizado
    # deve preservar o valor do campo.
    np.testing.assert_allclose(
        result.values,
        10.0,
        atol=1e-5,
    )

    assert result.attrs["beam_convolved"] is True


def test_project_requires_ra_and_dec():
    projector = HealpySkyMapTODProjector()

    invalid = xr.Dataset(
        {
            "ra": (
                "time",
                np.array([0.0]),
            ),
        },
        coords={
            "time": np.array([0.0]),
        },
    )

    with pytest.raises(ValueError):
        projector.project(
            skymap=_constant_map(),
            pointing_track=invalid,
        )


def test_project_rejects_invalid_fwhm():
    projector = HealpySkyMapTODProjector()

    with pytest.raises(ValueError):
        projector.project(
            skymap=_constant_map(),
            pointing_track=_pointing_track(),
            fwhm_rad=0.0,
        )


def test_project_accepts_pointing_in_degrees():
    nside = NSide(8)

    skymap = SkyMap(
        dataset=np.ones(
            hp.nside2npix(nside.value),
            dtype=np.float64,
        ),
        nside=nside,
        unit="K",
        frame="galactic",
        freq_ghz=1.0,
        field="I",
    )

    pointing = xr.Dataset(
        data_vars={
            "ra": (
                "time",
                np.array(
                    [
                        0.0,
                        45.0,
                        90.0,
                    ]
                ),
                {
                    "units": "deg",
                },
            ),
            "dec": (
                "time",
                np.array(
                    [
                        0.0,
                        10.0,
                        -10.0,
                    ]
                ),
                {
                    "units": "deg",
                },
            ),
        },
        coords={
            "time": np.arange(3),
        },
    )

    projector = HealpySkyMapTODProjector()

    result = projector.project(
        skymap=skymap,
        pointing_track=pointing,
    )

    assert result.shape == (3,)

    np.testing.assert_allclose(
        result.values,
        1.0,
    )
