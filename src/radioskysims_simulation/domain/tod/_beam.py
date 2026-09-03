"""Mixin de manejo de beam do TODSimulator (E5).

Agrupa a normalização do feixe e o cálculo do FWHM gaussiano-equivalente por
frequência. Extraído de ``domain/tod/simulator.py`` (decomposição do hotspot).

Copyright BINGO Collaboration
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from radioskysims_beam.domain.beam.operations import BeamOps
from radioskysims_beam.domain.beam.properties import BeamProperties


class BeamHandlingMixin:
    """Métodos de normalização e FWHM do feixe usados pelo TODSimulator."""

    @staticmethod
    def _normalize_beam(
        beam: Any | None,
    ) -> Any | None:
        """
        Normalize a concrete Beam or a Beam-compatible object.

        Concrete domain Beam objects expose ``dataset`` and continue using
        ``BeamOps.normalize``. Adapters and test doubles may instead expose
        the public ``normalized`` and ``evaluate`` methods.
        """

        if beam is None:
            return None

        dataset = getattr(
            beam,
            "dataset",
            None,
        )

        # Preserve the existing behavior for the concrete domain Beam.
        if dataset is not None:
            return BeamOps.normalize(beam)

        normalized = getattr(
            beam,
            "normalized",
            None,
        )

        evaluate = getattr(
            beam,
            "evaluate",
            None,
        )

        if not callable(normalized) or not callable(evaluate):
            msg = (
                "beam deve fornecer um atributo 'dataset' ou implementar "
                "os métodos normalized() e evaluate()."
            )
            raise TypeError(msg)

        normalized_beam = normalized(
            kind="integral",
        )

        if normalized_beam is None:
            msg = "beam.normalized() deve retornar um objeto de feixe."
            raise TypeError(msg)

        if not callable(
            getattr(
                normalized_beam,
                "evaluate",
                None,
            )
        ):
            msg = "O feixe normalizado deve implementar evaluate()."
            raise TypeError(msg)

        return normalized_beam

    def _beam_fwhm_by_frequency(
        self,
    ) -> xr.DataArray | None:
        """
        Return the Gaussian-equivalent beam FWHM by frequency.

        The FWHM is computed from the configured beam profile using
        ``BeamProperties.compute_properties``.

        When the beam has multiple polarizations, the initial diffuse-sky
        implementation uses the mean FWHM across polarization for each
        frequency.

        Returns
        -------
        xr.DataArray | None
            FWHM indexed by ``frequency``, in degrees.

            Returns ``None`` when no concrete beam dataset is configured.
        """

        if self.beam is None:
            return None

        beam_dataset = getattr(
            self.beam,
            "dataset",
            None,
        )

        if beam_dataset is None:
            return None

        if "amplitude" not in beam_dataset:
            msg = "Beam dataset must contain 'amplitude' to compute FWHM."
            raise ValueError(msg)

        if "frequency" not in beam_dataset.coords:
            msg = "Beam dataset must contain a frequency coordinate."
            raise ValueError(msg)

        properties = BeamProperties.compute_properties(beam_dataset)

        if "fwhm" not in properties:
            msg = "BeamProperties did not produce FWHM."
            raise ValueError(msg)

        fwhm = properties["fwhm"]

        # BeamProperties returns polarization x frequency
        # when polarization exists.
        if "polarization" in fwhm.dims:
            fwhm = fwhm.mean(
                dim="polarization",
                skipna=True,
            )

        # Ensure canonical frequency-only ordering.
        if "frequency" not in fwhm.dims:
            msg = "Beam FWHM must depend on frequency."
            raise ValueError(msg)

        fwhm = fwhm.transpose(
            "frequency",
        )

        fwhm.attrs.update(
            {
                "units": "deg",
                "description": ("Gaussian-equivalent beam FWHM used for diffuse sky convolution."),
            }
        )

        # Preserve frequency units from the original beam table.
        frequency_units = str(
            beam_dataset.coords["frequency"].attrs.get(
                "units",
                "",
            )
        )

        if frequency_units:
            fwhm.coords["frequency"].attrs["units"] = frequency_units

        return fwhm
