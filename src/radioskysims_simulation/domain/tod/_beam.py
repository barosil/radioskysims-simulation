"""Mixin de manejo de beam do TODSimulator (E5).

Agrupa a normalização do feixe e o cálculo do FWHM gaussiano-equivalente por
frequência. Extraído de ``domain/tod/simulator.py`` (decomposição do hotspot).

Copyright BINGO Collaboration
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from radioskysims_beam.domain.beam.operations import BeamOps
from radioskysims_beam.domain.beam.properties import BeamProperties
from scipy.constants import c as _C_LIGHT


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

    def _beam_frequency_hz(self) -> np.ndarray:
        """Frequências nativas do feixe em Hz (converte pelas unidades/valores)."""
        ds = self.beam.dataset
        freqs = np.asarray(ds["frequency"].values, dtype=float)
        units = str(ds["frequency"].attrs.get("units", "")).lower()
        if "ghz" in units:
            return freqs * 1e9
        if "mhz" in units:
            return freqs * 1e6
        if "hz" in units:
            return freqs
        return freqs * 1e6 if freqs.size and np.nanmax(freqs) < 1e6 else freqs

    def _resolve_effective_area(self, freq: xr.DataArray) -> float | xr.DataArray:
        """Área efetiva A_e por frequência para a radiometria de fontes discretas.

        Ordem de precedência:

        1. ``effective_area_m2`` explícito (override de config);
        2. ``A_e(ν) = λ²/Ω_A(ν)`` (Kraus), com Ω_A = 2π ∫ B_n(θ) sinθ dθ
           integrado do feixe (``BeamProperties`` → ``omega``);

        Raises
        ------
        ValueError
            Quando não há área efetiva configurada nem feixe com eixo de
            frequência para derivar Ω_A.
        """
        configured = getattr(self, "effective_area_m2", None)
        if configured is not None:
            return float(configured)

        beam_dataset = getattr(getattr(self, "beam", None), "dataset", None)
        if beam_dataset is None or "frequency" not in getattr(beam_dataset, "coords", {}):
            msg = (
                "effective_area_m2 must be provided for absolute radiometric "
                "conversion of discrete-source flux density to antenna "
                "temperature, or the beam must expose a frequency axis so that "
                "A_e = λ²/Ω_A can be derived."
            )
            raise ValueError(msg)

        properties = BeamProperties.compute_properties(beam_dataset)
        if "omega" not in properties:
            msg = "BeamProperties did not produce omega (beam solid angle)."
            raise ValueError(msg)

        omega = properties["omega"]
        has_pol = "polarization" in omega.dims
        pols = omega["polarization"].values if has_pol else None

        # Ω integrado na grade NATIVA do feixe → interpola para as frequências
        # pedidas (em Hz).
        freq_vals = np.asarray(freq.values, dtype=float)
        freq_hz = freq_vals * 1e6 if (freq_vals.size and np.nanmax(freq_vals) < 1e6) else freq_vals
        native_hz = self._beam_frequency_hz()

        omega_native = np.asarray(omega.values, dtype=float)
        if native_hz.size == 1:
            # Eixo de frequência com um único ponto: Ω é constante (broadcast).
            omega_at = np.repeat(omega_native, freq_hz.size, axis=-1)
        else:
            omega_hz = omega.assign_coords(frequency_hz=("frequency", native_hz)).swap_dims(
                {"frequency": "frequency_hz"}
            )
            omega_at = np.asarray(
                omega_hz.interp(
                    frequency_hz=xr.DataArray(freq_hz, dims="frequency_hz"),
                    kwargs={"fill_value": None},
                ).values,
                dtype=float,
            )

        lam2 = (_C_LIGHT / freq_hz) ** 2
        area_vals = np.where(omega_at > 0.0, lam2 / omega_at, np.nan)
        dims = ("polarization", "frequency") if has_pol else ("frequency",)
        coords = {"frequency": freq.coords["frequency"]}
        if has_pol:
            coords["polarization"] = pols
        return xr.DataArray(area_vals, dims=dims, coords=coords, attrs={"units": "m^2"})
