from __future__ import annotations

import healpy as hp
import numpy as np
import xarray as xr

from radioskysims_skymap.domain.skymap import SkyMap
from radioskysims_simulation.domain.tod.map_projector import SkyMapTODProjector


class HealpySkyMapTODProjector(SkyMapTODProjector):
    """
    Projeta mapas HEALPix ao longo de uma track de apontamento.

    O pointing recebido pelo pipeline possui coordenadas
    equatoriais RA/Dec. As unidades são lidas dos atributos
    ``units`` das DataArrays e podem ser graus ou radianos.

    Na ausência do atributo ``units``, assume-se radianos
    por compatibilidade com a API anterior.

    Os mapas PySM3 do RadioSkySim são, por padrão, galácticos.
    Portanto, antes da amostragem, as coordenadas são convertidas
    de equatorial para galáctico.

    A convolução instrumental inicial utiliza uma aproximação
    gaussiana isotrópica definida pelo FWHM equivalente do beam.
    """

    def project(
        self,
        *,
        skymap: SkyMap,
        pointing_track: xr.Dataset,
        fwhm_rad: float | None = None,
    ) -> xr.DataArray:
        if not isinstance(skymap, SkyMap):
            msg = "skymap must be a SkyMap."
            raise TypeError(msg)

        if not isinstance(pointing_track, xr.Dataset):
            msg = "pointing_track must be an xarray.Dataset."
            raise TypeError(msg)

        required = {"ra", "dec"}
        missing = required - set(pointing_track.data_vars)

        if missing:
            msg = f"pointing_track must contain RA/Dec variables; missing: {sorted(missing)}."
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
            ra_rad = np.deg2rad(ra_values)
        elif ra_unit in {"rad", "radian", "radians"}:
            ra_rad = ra_values
        else:
            msg = f"Unsupported RA unit in pointing_track: {ra_unit!r}."
            raise ValueError(msg)

        if dec_unit in {"deg", "degree", "degrees"}:
            dec_rad = np.deg2rad(dec_values)
        elif dec_unit in {"rad", "radian", "radians"}:
            dec_rad = dec_values
        else:
            msg = f"Unsupported Dec unit in pointing_track: {dec_unit!r}."
            raise ValueError(msg)

        if ra_rad.shape != dec_rad.shape:
            msg = "pointing_track RA and Dec must have the same shape."
            raise ValueError(msg)

        map_data = np.asarray(
            skymap.data,
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # Beam convolution
        # --------------------------------------------------------------
        if fwhm_rad is not None:
            if fwhm_rad <= 0:
                msg = "fwhm_rad must be positive."
                raise ValueError(msg)

            # A convolução de um campo estritamente constante com um beam
            # normalizado preserva exatamente o valor do campo.
            #
            # Além de ser fisicamente correto, este atalho evita pequenos
            # resíduos numéricos introduzidos pelo round-trip harmônico
            # map -> alm -> smoothing -> map do Healpy.
            is_constant = np.all(map_data == map_data[0])

            if not is_constant:
                map_data = np.asarray(
                    hp.smoothing(
                        map_data,
                        fwhm=float(fwhm_rad),
                    ),
                    dtype=np.float64,
                )

        # --------------------------------------------------------------
        # Coordenadas
        # --------------------------------------------------------------
        # HEALPix usa:
        #
        #   theta = colatitude
        #   phi   = longitude
        #
        # Para equatorial:
        #   theta = π/2 - Dec
        #   phi   = RA
        theta_equatorial = (np.pi / 2.0) - dec_rad
        phi_equatorial = ra_rad

        frame = skymap.frame.lower()

        if frame == "galactic":
            # C = Celestial/Equatorial
            # G = Galactic
            rotator = hp.Rotator(
                coord=["C", "G"],
            )

            theta_map, phi_map = rotator(
                theta_equatorial,
                phi_equatorial,
            )

        elif frame == "equatorial":
            theta_map = theta_equatorial
            phi_map = phi_equatorial

        else:
            msg = (
                "SkyMap TOD projection currently supports "
                "'galactic' and 'equatorial' frames; "
                f"received {skymap.frame!r}."
            )
            raise ValueError(msg)

        # --------------------------------------------------------------
        # HEALPix interpolation
        # --------------------------------------------------------------
        samples = hp.get_interp_val(
            map_data,
            theta_map,
            phi_map,
        )

        result = xr.DataArray(
            np.asarray(samples, dtype=np.float64),
            dims=("time",),
            coords={
                "time": pointing_track.coords["time"],
            },
            name="sky_temperature",
            attrs={
                "units": skymap.unit,
                "frame": skymap.frame,
                "frequency_ghz": skymap.freq_ghz,
                "beam_convolved": fwhm_rad is not None,
                "beam_fwhm_rad": fwhm_rad,
            },
        )

        return result
