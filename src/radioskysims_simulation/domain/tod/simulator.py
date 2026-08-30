from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from scipy.constants import k as k_B

from radioskysims_beam.domain.beam.operations import BeamOps
from radioskysims_beam.domain.beam.properties import BeamProperties

# Unidades físicas aceitas para o sinal de entrada do TODSimulator.
# - "W/Hz"     → potência total transmitida (aplica diluição isotrópica 1/(4π r²)).
# - "W/m²/Hz"  → densidade de fluxo no observador (distância já embutida, sem 1/r²).
_TOTAL_POWER_UNITS = {"W/Hz", "W Hz^-1"}
_FLUX_DENSITY_UNITS = {"W/m²/Hz", "W m^-2 Hz^-1", "W/(m² Hz)"}


class TODSimulator:
    """
    Time-Ordered Data (TOD) simulator in time x frequency domain.

    O simulador possui dois caminhos físicos distintos:

    1. Fontes discretas
       ----------------
       Satélites, fontes locais e rádio-fontes, descritos por PSD/fluxo
       e tracks temporais.

       T_antenna(t, v) = G · B(theta(t), v) · A_e · S_obs(t, v) / k_B

       com A_e = λ² (on-axis, beam normalizado ∫ B dΩ = 1).

    2. Céu difuso baseado em mapas
       ---------------------------
       Mapas HEALPix representam diretamente temperatura de brilho do céu.
       Neste caso, o mapa é convolvido com o beam equivalente e amostrado
       ao longo do pointing temporal.

       T_antenna(t, v) = G · [B_v * T_sky(v)](pointing(t))

    Para mapas não é aplicada novamente a conversão fluxo -> temperatura,
    pois o mapa já contém temperatura de brilho.

    power(t, v) = k_B · T_antenna(t, v)  [W / Hz]
    """

    def __init__(
        self,
        *,
        beam: Any | None,
        gain: float,
        effective_area_m2: float | None = None,
    ) -> None:
        self.beam = self._normalize_beam(beam)
        self.gain = gain
        self.effective_area_m2 = effective_area_m2

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
            raise TypeError(
                msg
            )

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

    def simulate_from_source(
        self,
        *,
        sat: Any,
        angle: xr.DataArray | None = None,
        _sky: xr.Dataset | None = None,
        _noise: xr.Dataset | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
    ) -> xr.Dataset:
        """
        Simulate time-ordered data from a discrete signal source.

        Parameters
        ----------
        sat
            Satellite/source adapter providing:

            - ``psd() -> xr.DataArray(frequency)``;
            - ``angle() -> xr.DataArray(time)``;
            - ``range() -> xr.DataArray(time)``.

        angle
            Optional angle to use when evaluating the beam. When omitted,
            the source angle is used.

        _sky
            Reserved for backwards compatibility.

        _noise
            Reserved for a future noise contribution.

        mueller
            Optional Mueller matrix with dimensions ``polarization`` and
            ``stokes``, used to map Stokes I into instrument polarizations.

        combine_polarization
            Accepted values are ``sum``, ``mean`` and ``separate``.
            Polarizations remain separate in this implementation; the
            downstream Simulation layer decides how to combine them.

        Returns
        -------
        xr.Dataset
            TOD dataset with time and frequency dimensions.
        """

        if combine_polarization not in {
            "sum",
            "mean",
            "separate",
        }:
            msg = "combine_polarization must be one of: sum, mean, separate."
            raise ValueError(msg)

        # --------------------------------------------------
        # Satellite/source inputs
        # --------------------------------------------------
        psd = sat.psd()
        freq = psd.coords["frequency"]
        theta = sat.angle()
        distance = sat.range()

        theta_beam = angle if angle is not None else theta

        # --------------------------------------------------
        # Contract checks
        # --------------------------------------------------
        signal_units = str(
            psd.attrs.get(
                "units",
                "",
            )
        )

        if signal_units not in (_TOTAL_POWER_UNITS | _FLUX_DENSITY_UNITS):
            msg = (
                "Signal units must be 'W/Hz' (total power) or "
                "'W/m²/Hz' (flux density); got "
                f"{signal_units!r}"
            )
            raise ValueError(
                msg
            )

        if self.beam is None:
            msg = (
                "TODSimulator requer beam configurado antes de simulate(). "
                "Forneça um beam no construtor ou injete em "
                "`simulator.beam`."
            )
            raise ValueError(
                msg
            )

        evaluate = getattr(
            self.beam,
            "evaluate",
            None,
        )

        if not callable(evaluate):
            msg = "O beam configurado deve implementar evaluate()."
            raise TypeError(msg)

        # --------------------------------------------------
        # Beam metadata and polarization
        # --------------------------------------------------
        beam_dataset = getattr(
            self.beam,
            "dataset",
            None,
        )

        beam_dimensions = getattr(
            beam_dataset,
            "dims",
            (),
        )

        has_polarization = beam_dataset is not None and "polarization" in beam_dimensions

        polarizations: np.ndarray | None = None

        # --------------------------------------------------
        # Beam convolution / point-source response
        # --------------------------------------------------
        if has_polarization:
            beam_coordinates = getattr(
                beam_dataset,
                "coords",
                {},
            )

            if "polarization" not in beam_coordinates:
                msg = "Beam dataset declares polarization dimension without polarization coordinates."
                raise ValueError(
                    msg
                )

            polarizations = np.asarray(beam_coordinates["polarization"].values)

            beam_responses = [
                evaluate(
                    frequency=freq,
                    angle=theta_beam,
                    polarization=polarization,
                )
                for polarization in polarizations
            ]

            beam_response = xr.concat(
                beam_responses,
                dim=xr.DataArray(
                    polarizations,
                    dims="polarization",
                    name="polarization",
                ),
            )

        else:
            beam_response = evaluate(
                frequency=freq,
                angle=theta_beam,
            )
        # --------------------------------------------------
        # Polarization and Mueller response
        # --------------------------------------------------
        if has_polarization and polarizations is not None and mueller is not None:
            required_dimensions = {
                "polarization",
                "stokes",
            }

            if not required_dimensions.issubset(mueller.dims):
                msg = "Mueller matrix must have dims ('polarization', 'stokes')."
                raise ValueError(msg)

            stokes_coordinate = mueller.coords.get("stokes")

            if stokes_coordinate is None or "I" not in stokes_coordinate.values:
                msg = "Mueller matrix must include stokes='I'."
                raise ValueError(msg)

            mueller_i = mueller.sel(
                stokes="I",
                drop=True,
            )

            mueller_i = mueller_i.sel(
                polarization=polarizations,
            )

            psd_polarized = psd * mueller_i

        elif has_polarization and polarizations is not None:
            psd_polarized = xr.concat(
                [psd] * len(polarizations),
                dim=xr.DataArray(
                    polarizations,
                    dims="polarization",
                    name="polarization",
                ),
            )

        else:
            psd_polarized = psd

        # --------------------------------------------------
        # Conversão física: sinal -> temperatura de antena
        # --------------------------------------------------
        # O fator de diluição isotrópica 1/(4π r²) só se aplica quando o
        # sinal é potência total transmitida (W/Hz). Se o sinal já é
        # densidade de fluxo no observador (W/m²/Hz), a distância já está
        # embutida e NÃO se aplica o 1/r².
        if signal_units in _TOTAL_POWER_UNITS:
            flux_at_observer = psd_polarized / (4 * np.pi * distance**2)

        elif signal_units in _FLUX_DENSITY_UNITS:
            flux_at_observer = psd_polarized

        else:  # pragma: no cover - já validado no contrato
            msg = (
                "Unrecognized signal units for T_antenna: "
                f"{signal_units!r}. Use 'W/Hz' "
                "(total power) or 'W/m²/Hz' "
                "(flux density)."
            )
            raise ValueError(
                msg
            )

        if self.effective_area_m2 is None:
            msg = (
                "effective_area_m2 must be provided for absolute radiometric "
                "conversion of discrete-source flux density to antenna temperature."
            )
            raise ValueError(
                msg
            )

        # Fonte não polarizada observada em uma polarização instrumental:
        #
        #   P_nu = A_e S_nu / 2
        #   T_A  = P_nu / k_B
        #
        # Portanto:
        #
        #   T_A(theta, nu)
        #       = G B(theta, nu) A_e S_nu,obs / (2 k_B)
        antenna_temperature = (
            self.gain * beam_response * flux_at_observer * self.effective_area_m2 / (2.0 * k_B)
        )

        power = k_B * antenna_temperature

        # --------------------------------------------------
        # Dataset assembly
        # --------------------------------------------------
        dataset = xr.Dataset(
            data_vars={
                "T_antenna": antenna_temperature,
                "power": power,
            },
            coords={
                "time": theta.coords["time"],
                "frequency": freq,
            },
            attrs={
                "description": "Time-Ordered Data",
                "contributions": ["satellite"],
                "backend": "ideal",
                "polarization_handling": (
                    "mueller" if (has_polarization and mueller is not None) else "unpolarized"
                ),
            },
        )

        dataset["T_antenna"].attrs.update(
            {
                "units": "K",
            }
        )

        dataset["power"].attrs.update(
            {
                "units": "W/Hz",
            }
        )

        return dataset

    def simulate(
        self,
        *,
        sat: Any,
        angle: xr.DataArray | None = None,
        _sky: xr.Dataset | None = None,
        _noise: xr.Dataset | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
    ) -> xr.Dataset:
        """
        Backward-compatible entry point for discrete-source TOD simulation.

        Notes
        -----
        This method preserves the historical ``TODSimulator.simulate()``
        API used throughout RadioSkySim.

        New code should prefer ``simulate_from_source()`` when the input
        is explicitly a discrete SignalSource.

        Diffuse HEALPix maps use ``simulate_from_maps()``.
        """

        return self.simulate_from_source(
            sat=sat,
            angle=angle,
            _sky=_sky,
            _noise=_noise,
            mueller=mueller,
            combine_polarization=combine_polarization,
        )

    def simulate_from_maps(
        self,
        *,
        maps: tuple[Any, ...] | list[Any],
        pointing_track: xr.Dataset,
        projector: Any,
        fwhm_deg: float | xr.DataArray | None = None,
        use_beam: bool = True,
    ) -> xr.Dataset:
        """
        Generate TOD from diffuse HEALPix sky maps.

        Este caminho é fisicamente distinto de ``simulate()``.

        Os mapas já representam temperatura de brilho do céu e,
        portanto, não passam pela conversão de PSD/fluxo utilizada
        para fontes discretas.

        O fluxo é:

            SkyMap(freq)
                ->
            beam convolution
                ->
            sampling at telescope pointing(time)
                ->
            antenna temperature(time, frequency)

        Parameters
        ----------
        maps
            Sequência de objetos SkyMap.

            Cada mapa deve possuir ``freq_ghz`` indicando a
            frequência física à qual pertence.

        pointing_track
            Dataset temporal de apontamento.

            Deve possuir as variáveis:

            - ``ra``;
            - ``dec``.

            Ambas são esperadas em radianos conforme o contrato
            atual do pointing temporal do RadioSkySim.

        projector
            Implementação compatível com ``SkyMapTODProjector``.

            Deve fornecer:

            ``project(skymap, pointing_track, fwhm_rad)``

        fwhm_deg
            FWHM gaussiano equivalente do beam em graus.

            Pode ser:

            - ``float``:
              mesmo FWHM para todos os mapas;

            - ``xr.DataArray``:
              FWHM dependente da frequência;

            - ``None``:
              mapa é projetado sem smoothing adicional.

        Returns
        -------
        xr.Dataset
            Dataset TOD com dimensões:

            ``time x frequency``

            contendo:

            - ``T_antenna`` em K;
            - ``power`` em W/Hz.
        """

        # --------------------------------------------------
        # Contract checks
        # --------------------------------------------------
        if not maps:
            msg = "simulate_from_maps requires at least one SkyMap."
            raise ValueError(msg)

        if not isinstance(
            pointing_track,
            xr.Dataset,
        ):
            msg = "pointing_track must be an xarray.Dataset."
            raise TypeError(msg)

        required_pointing = {
            "ra",
            "dec",
        }

        missing_pointing = required_pointing - set(pointing_track.data_vars)

        if missing_pointing:
            msg = f"pointing_track must contain 'ra' and 'dec'; missing: {sorted(missing_pointing)}."
            raise ValueError(
                msg
            )

        if "time" not in pointing_track.coords:
            msg = "pointing_track must contain a time coordinate."
            raise ValueError(msg)

        project = getattr(
            projector,
            "project",
            None,
        )

        if not callable(project):
            msg = "projector must implement project()."
            raise TypeError(msg)

        resolved_fwhm_input: float | xr.DataArray | None = fwhm_deg

        if resolved_fwhm_input is None and use_beam:
            resolved_fwhm_input = self._beam_fwhm_by_frequency()

        projected_maps: list[xr.DataArray] = []
        frequencies_hz: list[float] = []

        # --------------------------------------------------
        # Process each frequency map independently
        # --------------------------------------------------
        for sky_map in maps:
            freq_ghz = getattr(
                sky_map,
                "freq_ghz",
                None,
            )

            if freq_ghz is None:
                msg = "Each SkyMap must define freq_ghz."
                raise ValueError(msg)

            freq_ghz = float(freq_ghz)

            if not np.isfinite(freq_ghz) or freq_ghz <= 0:
                msg = "SkyMap freq_ghz must be a positive finite value."
                raise ValueError(msg)

            freq_hz = freq_ghz * 1.0e9

            # --------------------------------------------------
            # Resolve beam FWHM for this map frequency
            # --------------------------------------------------
            resolved_fwhm_deg: float | None = None

            if resolved_fwhm_input is not None:
                if isinstance(
                    resolved_fwhm_input,
                    xr.DataArray,
                ):
                    if "frequency" not in resolved_fwhm_input.coords:
                        msg = "fwhm_deg DataArray must have a frequency coordinate."
                        raise ValueError(msg)

                    beam_frequency = resolved_fwhm_input.coords["frequency"]
                    beam_units = str(
                        beam_frequency.attrs.get(
                            "units",
                            "",
                        )
                    ).lower()

                    # Frequency coordinate used to interpolate
                    # within the beam FWHM table.
                    target_frequency = freq_hz

                    if "ghz" in beam_units:
                        target_frequency = freq_ghz

                    elif "mhz" in beam_units:
                        target_frequency = freq_hz / 1.0e6

                    elif "hz" in beam_units:
                        target_frequency = freq_hz

                    else:
                        # Fallback compatible with the existing
                        # Beam implementation:
                        # beam tables commonly use MHz.
                        try:
                            beam_fmax = float(beam_frequency.max())
                        except Exception:
                            beam_fmax = None

                        if beam_fmax is not None and beam_fmax < 1.0e6:
                            target_frequency = freq_hz / 1.0e6

                    resolved_fwhm_deg = float(
                        resolved_fwhm_input.interp(
                            frequency=target_frequency,
                        ).values
                    )

                else:
                    resolved_fwhm_deg = float(resolved_fwhm_input)

                if not np.isfinite(resolved_fwhm_deg) or resolved_fwhm_deg <= 0:
                    msg = "Resolved beam FWHM must be a positive finite value."
                    raise ValueError(msg)

            resolved_fwhm_rad = (
                np.deg2rad(resolved_fwhm_deg) if resolved_fwhm_deg is not None else None
            )

            # --------------------------------------------------
            # HEALPix map -> temporal sky temperature
            # --------------------------------------------------
            sky_temperature = project(
                skymap=sky_map,
                pointing_track=pointing_track,
                fwhm_rad=resolved_fwhm_rad,
            )

            if not isinstance(
                sky_temperature,
                xr.DataArray,
            ):
                msg = "projector.project() must return an xarray.DataArray."
                raise TypeError(msg)

            if "time" not in sky_temperature.dims:
                msg = "Projected SkyMap must contain the time dimension."
                raise ValueError(msg)

            # --------------------------------------------------
            # Sky temperature unit -> Kelvin
            # --------------------------------------------------
            map_unit = str(
                getattr(
                    sky_map,
                    "unit",
                    "",
                )
            )

            if map_unit in {"uK", "uK_CMB"}:
                temperature_k = sky_temperature * 1.0e-6

            elif map_unit == "K":
                temperature_k = sky_temperature

            else:
                msg = (
                    f"Unsupported SkyMap temperature unit {map_unit!r}. "
                    "Expected 'uK', 'uK_CMB' or 'K'."
                )
                raise ValueError(
                    msg
                )

            # --------------------------------------------------
            # Instrument gain
            # --------------------------------------------------
            temperature_k = self.gain * temperature_k

            projected_maps.append(temperature_k)

            frequencies_hz.append(freq_hz)

        # --------------------------------------------------
        # Stack channels -> time x frequency
        # --------------------------------------------------
        frequency_coordinate = xr.DataArray(
            np.asarray(
                frequencies_hz,
                dtype=np.float64,
            ),
            dims="frequency",
            name="frequency",
            attrs={
                "units": "Hz",
            },
        )

        antenna_temperature = xr.concat(
            projected_maps,
            dim=frequency_coordinate,
        )

        antenna_temperature = antenna_temperature.transpose(
            "time",
            "frequency",
        )

        antenna_temperature.name = "T_antenna"

        # --------------------------------------------------
        # Radiometric power
        # --------------------------------------------------
        power = k_B * antenna_temperature

        power.name = "power"

        # --------------------------------------------------
        # Dataset assembly
        # --------------------------------------------------
        dataset = xr.Dataset(
            data_vars={
                "T_antenna": antenna_temperature,
                "power": power,
            },
            coords={
                "time": pointing_track.coords["time"],
                "frequency": (antenna_temperature.coords["frequency"]),
            },
            attrs={
                "description": "Time-Ordered Data",
                "contributions": [
                    "skymap",
                ],
                "backend": "ideal",
                "sky_model": "diffuse",
                "beam_model": (
                    "gaussian_equivalent" if resolved_fwhm_input is not None else "none"
                ),
                "beam_fwhm_source": (
                    "explicit"
                    if fwhm_deg is not None
                    else ("beam_properties" if resolved_fwhm_input is not None else "none")
                ),
            },
        )

        dataset["T_antenna"].attrs.update(
            {
                "units": "K",
            }
        )

        dataset["power"].attrs.update(
            {
                "units": "W/Hz",
            }
        )

        dataset.coords["frequency"].attrs.update(
            {
                "units": "Hz",
            }
        )

        return dataset
