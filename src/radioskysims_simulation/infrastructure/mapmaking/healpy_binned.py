from __future__ import annotations

import healpy as hp
import numpy as np
import xarray as xr

from radioskysims_simulation.application.ports.mapmaking import (
    MapMakerPort,
)
from radioskysims_simulation.domain.mapmaking import (
    MapMakingResult,
)
from radioskysims_skymap.domain.skymap import SkyMap


class HealpyBinnedMapMakerAdapter(MapMakerPort):
    """
    Naive HEALPix binned mapmaker.

    Para cada amostra temporal:

        pointing -> pixel HEALPix

    e então calcula:

        map[p] = sum(T_antenna[p]) / hits[p]

    Pixels não observados recebem zero.
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
        if not hp.isnsideok(
            nside,
            nest=True,
        ):
            msg = f"Invalid HEALPix nside: {nside}"
            raise ValueError(msg)

        if "T_antenna" not in tod:
            msg = "TOD dataset must contain 'T_antenna'."
            raise ValueError(msg)

        if "frequency" not in tod.coords:
            msg = "TOD dataset must contain frequency coordinates."
            raise ValueError(msg)

        if "ra" not in pointing_track:
            msg = "pointing_track must contain 'ra'."
            raise ValueError(msg)

        if "dec" not in pointing_track:
            msg = "pointing_track must contain 'dec'."
            raise ValueError(msg)

        temperature = tod["T_antenna"].sel(
            frequency=frequency_hz,
        )

        # Remove dimensões extras unitárias, caso existam.
        temperature = temperature.squeeze()

        if temperature.dims != ("time",):
            msg = "Selected TOD channel must have only the 'time' dimension."
            raise ValueError(msg)

        ra_values = np.asarray(
            pointing_track["ra"].values,
            dtype=np.float64,
        )

        dec_values = np.asarray(
            pointing_track["dec"].values,
            dtype=np.float64,
        )

        ra_unit = str(
            pointing_track["ra"].attrs.get(
                "units",
                "rad",
            )
        ).lower()

        dec_unit = str(
            pointing_track["dec"].attrs.get(
                "units",
                "rad",
            )
        ).lower()

        if ra_unit in {"deg", "degree", "degrees"}:
            ra = np.deg2rad(ra_values)
        elif ra_unit in {"rad", "radian", "radians"}:
            ra = ra_values
        else:
            msg = f"Unsupported RA unit in pointing_track: {ra_unit!r}."
            raise ValueError(msg)

        if dec_unit in {"deg", "degree", "degrees"}:
            dec = np.deg2rad(dec_values)
        elif dec_unit in {"rad", "radian", "radians"}:
            dec = dec_values
        else:
            msg = f"Unsupported Dec unit in pointing_track: {dec_unit!r}."
            raise ValueError(msg)

        values = np.asarray(
            temperature.values,
            dtype=np.float64,
        )

        if ra.shape != dec.shape or ra.shape != values.shape:
            msg = "TOD and pointing time dimensions must match."
            raise ValueError(msg)

        if not np.all(np.isfinite(values)):
            msg = "TOD contains non-finite temperatures."
            raise ValueError(msg)

        # HEALPix recebe coordenadas angulares em radianos.
        theta = np.pi / 2.0 - dec

        phi = np.mod(
            ra,
            2.0 * np.pi,
        )

        pixels = hp.ang2pix(
            nside,
            theta,
            phi,
            nest=False,
        )

        npix = hp.nside2npix(nside)

        hit_map = np.bincount(
            pixels,
            minlength=npix,
        ).astype(
            np.int64,
        )

        temperature_sum = np.bincount(
            pixels,
            weights=values,
            minlength=npix,
        ).astype(
            np.float64,
        )

        map_data = np.zeros(
            npix,
            dtype=np.float64,
        )

        observed = hit_map > 0

        map_data[observed] = temperature_sum[observed] / hit_map[observed]

        skymap = SkyMap(
            dataset=map_data,
            nside=nside,
            unit="K",
            frame=frame,
            freq_ghz=(float(frequency_hz) / 1.0e9),
            field="I",
        )

        return MapMakingResult(
            skymap=skymap,
            hit_map=hit_map,
            observed_mask=observed,
            n_samples=int(values.size),
        )
