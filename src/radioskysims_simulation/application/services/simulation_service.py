from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import xarray as xr

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from radioskysims_beam.domain.beam import Beam, BeamOps
from radioskysims_core.domain.common import FrequencyRange, Observer, TimeRange
from radioskysims_core.domain.geometry import GeometryProvider
from radioskysims_core.domain.tracks.factory import TrackFactory
from radioskysims_radiotelescope.domain.radiotelescope.antenna.models import Antenna
from radioskysims_radiotelescope.domain.radiotelescope.specs import RadiotelescopeSpec
from radioskysims_sources.application.services.satellite_service import SatelliteService
from radioskysims_sources.application.services.source_assembler import SourceAssembler
from radioskysims_sources.domain.satellite import Satellite, SatelliteAssembled
from radioskysims_sources.domain.signal import SignalFactory
from radioskysims_sources.domain.signal_source import (
    SatelliteSignalSource,
    SignalSource,
)

from radioskysims_simulation.domain.simulation import Simulation, SimulationResult
from radioskysims_simulation.domain.simulation.grids import (
    _beam_fwhm_deg,
    _compute_freq_mhz,
    _compute_time_grid,
)
from radioskysims_simulation.domain.tod import TODSimulator

T = TypeVar("T")


class _FallbackStructuredLogger:
    """Fallback no-op para manter API quando logger nao for injetado."""

    def info(self, event: str, **fields: Any) -> None:
        _ = (event, fields)

    def exception(self, event: str, **fields: Any) -> None:
        _ = (event, fields)


class SimulationService:
    """
    Serviço de aplicação responsável por orquestrar uma simulação completa.

    Responsabilidades:
    - Selecionar satélites (IDs, constelações, filtros)
    - Instanciar dependências físicas
    - Executar a Simulation
    """

    def __init__(
        self,
        satellite_service: SatelliteService,
        geometry: GeometryProvider,
        signal_factory: SignalFactory,
        tod_simulator: TODSimulator,
        *,
        telescope_loader: Callable[[str], RadiotelescopeSpec] | None = None,
        beam_loader_factory: Callable[[str], Beam] | None = None,
        source_assembler: SourceAssembler | None = None,
        logger: Any | None = None,
    ) -> None:
        self._satellite_service = satellite_service
        self._geometry = geometry
        self._signal_factory = signal_factory
        self._tod_simulator = tod_simulator
        self._telescope_loader = telescope_loader
        self._beam_loader_factory = beam_loader_factory
        self._source_assembler = source_assembler
        self._logger = logger or _FallbackStructuredLogger()

    def _resolve_beam(self, antenna: Antenna) -> Any | None:
        """Resolve o beam da antena: arquivo CSV ou analítico (gaussiano)."""
        beam_spec = antenna.beam
        if beam_spec is None:
            return None

        # Beam via arquivo CSV
        if beam_spec.beam_pattern_file is not None:
            return self._load_beam_from_file(beam_spec.beam_pattern_file)

        # Beam analítico (gaussiano, etc.)
        if beam_spec.profile is not None:
            from radioskysims_beam.infrastructure.beams.loaders.analytical import (
                AnalyticalBeamLoader,
                BeamProfile,
            )

            width_deg = beam_spec.beam_width_deg or 10.0
            loader = AnalyticalBeamLoader(
                beam_profile=BeamProfile(beam_spec.profile),
                sigma=width_deg,
                frequency=1000.0,  # MHz placeholder — interpolado depois
                angular_range=(-90.0, 90.0),
                angular_resolution=1.0,
            )
            return loader.load()

        return None

    def _log_and_execute(
        self,
        operation: str,
        start_fields: dict[str, Any],
        success_fields: dict[str, Any],
        fn: Callable[[], T],
    ) -> T:
        """
        Helper para executar operações com logging estruturado.

        Args:
            operation: Nome base da operação (ex: "simulation_run_by_ids")
            start_fields: Campos para log de início
            success_fields: Campos para log de sucesso
            fn: Função a executar

        Returns:
            Resultado da função

        Raises:
            Exception: Propaga exceções da função
        """
        try:
            log_msg = f"{operation}_start"
            self._logger.info(log_msg, **start_fields)
            result = fn()
            log_msg = f"{operation}_success"
            self._logger.info(log_msg, **success_fields)
            return result
        except Exception:
            log_msg = f"{operation}_failed"
            self._logger.exception(log_msg)
            raise

    # ------------------------------------------------------------------
    # Casos de uso
    # ------------------------------------------------------------------

    def run_by_constellations_from_settings(
        self,
        constellations: Iterable[str],
        *,
        telescope: str = "default",
        time_range: TimeRange | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
    ) -> SimulationResult:
        rt = self._load_telescope_from_settings(telescope)

        observer = rt.observer
        antenna = rt.antennas[0]
        backend = antenna.backend
        pointing = antenna.pointing

        tr = time_range or observer.timeframe
        frequency_range = None
        num_channels = None
        if backend is not None:
            frequency_range = backend.frequency_range
            num_channels = backend.num_channels
        if frequency_range is None:
            frequency_range = antenna.frequency_range

        beam = self._resolve_beam(antenna)

        return self.run_by_constellations(
            constellations,
            tr,
            observer=observer,
            frequency_range=frequency_range,
            num_channels=num_channels,
            beam=beam,
            pointing_alt_deg=getattr(pointing, "alt_deg", None),
            pointing_az_deg=getattr(pointing, "az_deg", None),
            mueller=mueller,
            combine_polarization=combine_polarization,
        )

    def run_by_ids_from_settings(
        self,
        norad_ids: Iterable[int],
        *,
        telescope: str = "default",
        time_range: TimeRange | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
    ) -> SimulationResult:
        rt = self._load_telescope_from_settings(telescope)

        observer = rt.observer
        antenna = rt.antennas[0]
        backend = antenna.backend
        pointing = antenna.pointing

        tr = time_range or observer.timeframe
        frequency_range = None
        num_channels = None
        if backend is not None:
            frequency_range = backend.frequency_range
            num_channels = backend.num_channels
        if frequency_range is None:
            frequency_range = antenna.frequency_range

        beam = self._resolve_beam(antenna)

        return self.run_by_ids(
            norad_ids,
            tr,
            observer=observer,
            frequency_range=frequency_range,
            num_channels=num_channels,
            beam=beam,
            pointing_alt_deg=getattr(pointing, "alt_deg", None),
            pointing_az_deg=getattr(pointing, "az_deg", None),
            mueller=mueller,
            combine_polarization=combine_polarization,
        )

    def run_by_ids(
        self,
        norad_ids: Iterable[int],
        time_range: TimeRange,
        *,
        observer: Observer | None = None,
        frequency_range: FrequencyRange | None = None,
        num_channels: int | None = None,
        beam: Beam | None = None,
        pointing_alt_deg: float | None = None,
        pointing_az_deg: float | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
    ) -> SimulationResult:
        """
        Executa simulação para um conjunto explícito de satélites (NORAD IDs).
        """
        norad_ids_list = list(norad_ids)

        def execute() -> SimulationResult:
            satellites = self._satellite_service.get_by_ids(norad_ids_list)
            return self._run(
                satellites,
                time_range,
                observer=observer,
                frequency_range=frequency_range,
                num_channels=num_channels,
                beam=beam,
                pointing_alt_deg=pointing_alt_deg,
                pointing_az_deg=pointing_az_deg,
                mueller=mueller,
                combine_polarization=combine_polarization,
            )

        return self._log_and_execute(
            operation="simulation_run_by_ids",
            start_fields={
                "num_satellites": len(norad_ids_list),
                "start_time": str(time_range.to_tuple()[0]),
                "end_time": str(time_range.to_tuple()[1]),
            },
            success_fields={"num_satellites": len(norad_ids_list)},
            fn=execute,
        )

    def run_by_constellations(
        self,
        constellations: Iterable[str],
        time_range: TimeRange,
        *,
        observer: Observer | None = None,
        frequency_range: FrequencyRange | None = None,
        num_channels: int | None = None,
        beam: Beam | None = None,
        pointing_alt_deg: float | None = None,
        pointing_az_deg: float | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
    ) -> SimulationResult:
        """
        Executa simulação para uma ou mais constelações.
        """
        constellations_list = list(constellations)

        def execute() -> SimulationResult:
            satellites = self._satellite_service.get_by_constellations(constellations_list)
            return self._run(
                satellites,
                time_range,
                observer=observer,
                frequency_range=frequency_range,
                num_channels=num_channels,
                beam=beam,
                pointing_alt_deg=pointing_alt_deg,
                pointing_az_deg=pointing_az_deg,
                mueller=mueller,
                combine_polarization=combine_polarization,
            )

        return self._log_and_execute(
            operation="simulation_run_by_constellations",
            start_fields={
                "constellations": constellations_list,
                "start_time": str(time_range.to_tuple()[0]),
                "end_time": str(time_range.to_tuple()[1]),
            },
            success_fields={"constellations": constellations_list},
            fn=execute,
        )

    def run(
        self,
        satellites: Iterable[Satellite],
        time_range: TimeRange,
        *,
        observer: Observer | None = None,
        frequency_range: FrequencyRange | None = None,
        num_channels: int | None = None,
        beam: Beam | None = None,
        pointing_alt_deg: float | None = None,
        pointing_az_deg: float | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
    ) -> SimulationResult:
        """
        Executa simulação para um conjunto já resolvido de satélites.

        Útil para:
        - testes
        - simulações sintéticas
        - pipelines externos
        """
        satellites_list = list(satellites)

        def execute() -> SimulationResult:
            return self._run(
                satellites_list,
                time_range,
                observer=observer,
                frequency_range=frequency_range,
                num_channels=num_channels,
                beam=beam,
                pointing_alt_deg=pointing_alt_deg,
                pointing_az_deg=pointing_az_deg,
                mueller=mueller,
                combine_polarization=combine_polarization,
            )

        return self._log_and_execute(
            operation="simulation_run",
            start_fields={
                "num_satellites": len(satellites_list),
                "start_time": str(time_range.to_tuple()[0]),
                "end_time": str(time_range.to_tuple()[1]),
            },
            success_fields={"num_satellites": len(satellites_list)},
            fn=execute,
        )

    # ------------------------------------------------------------------
    # Implementação interna
    # ------------------------------------------------------------------

    def _run(
        self,
        satellites: list[Satellite],
        time_range: TimeRange,
        *,
        observer: Observer | None,
        frequency_range: FrequencyRange | None,
        num_channels: int | None,
        beam: Beam | None,
        pointing_alt_deg: float | None,
        pointing_az_deg: float | None,
        mueller: xr.DataArray | None,
        combine_polarization: str,
    ) -> SimulationResult:
        has_requested_source = bool(satellites)

        if not has_requested_source:
            msg = "No sources selected for simulation."
            raise ValueError(msg)

        # Ensure signal definitions exist in the DB when the repository supports it.
        try:
            sig_repo = getattr(self._signal_factory, "signal_repo", None)
            init = getattr(sig_repo, "initialize_signals", None)
            if callable(init):
                init()
        except Exception:
            # Falha em inicializar sinais não deve quebrar a simulação.
            pass

        if frequency_range is None and beam is not None:
            frequency_range = self._frequency_range_from_beam(beam)

        frequency_range = self._normalize_frequency_range(
            frequency_range,
            num_channels,
        )

        if beam is not None:
            freq_grid = self._beam_frequency_grid(frequency_range, num_channels)
            if freq_grid is not None:
                beam = self._interpolate_beam_frequency(beam, freq_grid)

        if frequency_range is not None:
            self._signal_factory.fr = frequency_range

        if beam is not None:
            self._tod_simulator.beam = BeamOps.normalize(beam)

        if not self._satellites_are_assembled(satellites):
            if observer is None:
                msg = "Observer is required to assemble satellites for TOD."
                raise ValueError(
                    msg,
                )
            satellites = self._assemble_satellites(
                satellites=satellites,
                time_range=time_range,
                observer=observer,
            )

        # Converte SatelliteAssembled -> SignalSource
        sources: list[SignalSource] = [SatelliteSignalSource(sat) for sat in satellites]

        simulation = Simulation(
            sources=sources,
            time_range=time_range,
            tod_simulator=self._tod_simulator,
            pointing_alt_deg=pointing_alt_deg,
            pointing_az_deg=pointing_az_deg,
            mueller=mueller,
            combine_polarization=combine_polarization,
        )

        return simulation.run()

    # ------------------------------------------------------------------
    # Simulação multi-fonte (satélites + locais + rádio)
    # ------------------------------------------------------------------

    def run_multi_source(
        self,
        satellites: Iterable[Satellite],
        time_range: TimeRange,
        *,
        observer: Observer | None = None,
        frequency_range: FrequencyRange | None = None,
        num_channels: int | None = None,
        beam: Beam | None = None,
        pointing_ra_deg: float | None = None,
        pointing_dec_deg: float | None = None,
        pointing_alt_deg: float | None = None,
        pointing_az_deg: float | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
        include_local_sources: bool = False,
        include_radio_sources: bool = False,
        radio_search_radius_deg: float | None = None,
        sun_flux_mean_window_s: float | None = None,
        radio_min_flux_jy: float = 0.1,
    ) -> SimulationResult:
        """Executa simulação incluindo satélites + fontes locais + fontes de rádio.

        É o método principal que unifica todos os tipos de fonte no pipeline.

        Parameters
        ----------
        pointing_ra_deg, pointing_dec_deg:
            Pointing em coordenadas equatoriais (preferencial).
        pointing_alt_deg, pointing_az_deg:
            Pointing em coordenadas horizontais (fallback).
        sun_flux_mean_window_s:
            Largura da janela temporal (s) para a média do fluxo solar. Se
            ``None``, usa o tempo aproximado de um trânsito do Sol pelo feixe.
        radio_min_flux_jy:
            Corte de fluxo mínimo (Jy) da consulta ao catálogo de rádio.
            Vem do ``settings.toml`` ([radiosources] query_str, S > X Jy).
        """
        satellites_list = list(satellites)

        def execute() -> SimulationResult:
            return self._run_multi_source(
                satellites_list,
                time_range,
                observer=observer,
                frequency_range=frequency_range,
                num_channels=num_channels,
                beam=beam,
                pointing_ra_deg=pointing_ra_deg,
                pointing_dec_deg=pointing_dec_deg,
                pointing_alt_deg=pointing_alt_deg,
                pointing_az_deg=pointing_az_deg,
                mueller=mueller,
                combine_polarization=combine_polarization,
                include_local_sources=include_local_sources,
                include_radio_sources=include_radio_sources,
                radio_search_radius_deg=radio_search_radius_deg,
                radio_min_flux_jy=radio_min_flux_jy,
                sun_flux_mean_window_s=sun_flux_mean_window_s,
            )

        return self._log_and_execute(
            operation="simulation_run_multi_source",
            start_fields={
                "num_satellites": len(satellites_list),
                "include_local_sources": include_local_sources,
                "include_radio_sources": include_radio_sources,
                "start_time": str(time_range.to_tuple()[0]),
                "end_time": str(time_range.to_tuple()[1]),
            },
            success_fields={
                "num_satellites": len(satellites_list),
                "include_local_sources": include_local_sources,
                "include_radio_sources": include_radio_sources,
            },
            fn=execute,
        )

    def _run_multi_source(
        self,
        satellites: list[Satellite],
        time_range: TimeRange,
        *,
        observer: Observer | None,
        frequency_range: FrequencyRange | None,
        num_channels: int | None,
        beam: Beam | None,
        pointing_ra_deg: float | None,
        pointing_dec_deg: float | None,
        pointing_alt_deg: float | None,
        pointing_az_deg: float | None,
        mueller: xr.DataArray | None,
        combine_polarization: str,
        include_local_sources: bool,
        include_radio_sources: bool,
        radio_search_radius_deg: float | None = None,
        sun_flux_mean_window_s: float | None = None,
        radio_min_flux_jy: float = 0.1,
    ) -> SimulationResult:
        """Implementação interna da simulação multi-fonte."""
        has_requested_source = bool(satellites or include_local_sources or include_radio_sources)

        if not has_requested_source:
            msg = "No sources selected for simulation."
            raise ValueError(msg)
        try:
            sig_repo = getattr(self._signal_factory, "signal_repo", None)
            init = getattr(sig_repo, "initialize_signals", None)
            if callable(init):
                init()
        except Exception:
            pass

        if frequency_range is None and beam is not None:
            frequency_range = self._frequency_range_from_beam(beam)

        frequency_range = self._normalize_frequency_range(
            frequency_range,
            num_channels,
        )

        if beam is not None:
            freq_grid = self._beam_frequency_grid(frequency_range, num_channels)
            if freq_grid is not None:
                beam = self._interpolate_beam_frequency(beam, freq_grid)

        if frequency_range is not None:
            self._signal_factory.fr = frequency_range

        if beam is not None:
            self._tod_simulator.beam = BeamOps.normalize(beam)

        # 1. Monta satélites
        if not self._satellites_are_assembled(satellites):
            if observer is None:
                msg = "Observer is required to assemble satellites for TOD."
                raise ValueError(
                    msg,
                )
            satellites = self._assemble_satellites(
                satellites=satellites,
                time_range=time_range,
                observer=observer,
            )

        sources: list[SignalSource] = [SatelliteSignalSource(sat) for sat in satellites]

        # 2. Monta fontes locais
        source_assembler = self._source_assembler
        if include_local_sources and source_assembler is not None and observer is not None:
            try:
                time_grid = _compute_time_grid(time_range)
                freq_mhz = _compute_freq_mhz(frequency_range)
                local_sources = source_assembler.assemble_local_sources(
                    observer=observer,
                    time_range=time_range,
                    time_grid=time_grid,
                    freq_mhz=freq_mhz,
                    sun_flux_mean_window_s=sun_flux_mean_window_s,
                    beam_fwhm_deg=_beam_fwhm_deg(beam),
                )
                sources.extend(local_sources)
                self._logger.info(
                    "local_sources_assembled",
                    n_sources=len(local_sources),
                )
            except Exception as exc:
                self._logger.warning(
                    "local_sources_assembly_failed",
                    error=str(exc),
                )

        from radioskysims_core.domain.geometry import compute_pointing_track

        _effective_radio_search_radius_deg = (
            float(radio_search_radius_deg) if radio_search_radius_deg is not None else 20.0
        )
        _effective_radio_min_flux_jy = (
            float(radio_min_flux_jy) if radio_min_flux_jy is not None else 0.1
        )
        pointing_track: xr.Dataset | None = None
        if observer is not None:
            try:
                time_grid = _compute_time_grid(time_range)
                if pointing_alt_deg is not None and pointing_az_deg is not None:
                    pointing_track = compute_pointing_track(
                        observer=observer,
                        time_grid=time_grid,
                        alt_deg=pointing_alt_deg,
                        az_deg=pointing_az_deg,
                    )
            except Exception as exc:
                self._logger.warning("pointing_track_failed", error=str(exc))

        # 3. Monta fontes de rádio
        if include_radio_sources and source_assembler is not None and observer is not None:
            try:
                time_grid = _compute_time_grid(time_range)
                freq_mhz = _compute_freq_mhz(frequency_range)

                # Usa pointing real em vez de RA/Dec hardcoded
                _ra_center = pointing_ra_deg
                _dec_center = pointing_dec_deg
                if _ra_center is None and pointing_track is not None:
                    _ra_center = float(pointing_track["ra"].mean().values)
                    _dec_center = float(pointing_track["dec"].mean().values)
                if _ra_center is None:
                    _ra_center = 190.0
                    _dec_center = 0.0
                    self._logger.info(
                        "radio_sources_fallback_center",
                        ra=_ra_center,
                        dec=_dec_center,
                    )

                radio_sources = source_assembler.assemble_radio_sources(
                    observer=observer,
                    time_range=time_range,
                    time_grid=time_grid,
                    freq_mhz=freq_mhz,
                    ra_center_deg=_ra_center,
                    dec_center_deg=_dec_center,
                    search_radius_deg=_effective_radio_search_radius_deg,
                    min_flux_jy=_effective_radio_min_flux_jy,
                    pointing_track=pointing_track,
                    pointing_ra_deg=pointing_ra_deg,
                    pointing_dec_deg=pointing_dec_deg,
                    beam_fwhm_deg=_beam_fwhm_deg(beam),
                )
                sources.extend(radio_sources)
                self._logger.info(
                    "radio_sources_assembled",
                    n_sources=len(radio_sources),
                )
            except Exception as exc:
                self._logger.warning(
                    "radio_sources_assembly_failed",
                    error=str(exc),
                )

        # 5. Executa simulação com todas as fontes
        simulation = Simulation(
            sources=sources,
            time_range=time_range,
            tod_simulator=self._tod_simulator,
            pointing_ra_deg=pointing_ra_deg,
            pointing_dec_deg=pointing_dec_deg,
            pointing_alt_deg=pointing_alt_deg,
            pointing_az_deg=pointing_az_deg,
            pointing_track=pointing_track,
            mueller=mueller,
            combine_polarization=combine_polarization,
        )

        return simulation.run()

    def _load_telescope_from_settings(self, telescope: str) -> RadiotelescopeSpec:
        if self._telescope_loader is None:
            msg = "SimulationService telescope_loader is not configured."
            raise RuntimeError(
                msg,
            )
        return self._telescope_loader(telescope)

    def _load_beam_from_file(self, file_path: str) -> Beam:
        if self._beam_loader_factory is None:
            msg = "SimulationService beam_loader_factory is not configured."
            raise RuntimeError(
                msg,
            )
        return self._beam_loader_factory(file_path)

    @staticmethod
    def _satellites_are_assembled(satellites: list[Satellite]) -> bool:
        return all(
            hasattr(sat, "psd") and hasattr(sat, "angle") and hasattr(sat, "range")
            for sat in satellites
        )

    @staticmethod
    def _frequency_range_from_beam(beam: Beam) -> FrequencyRange | None:
        ds = beam.dataset
        if "frequency" not in ds.coords:
            return None
        freqs = ds.coords["frequency"].values
        if freqs.size == 0:
            return None
        try:
            return FrequencyRange.from_array(freqs.tolist())
        except Exception:
            return None

    def _assemble_satellites(
        self,
        *,
        satellites: list[Satellite],
        time_range: TimeRange,
        observer: Observer,
    ) -> list[SatelliteAssembled]:
        satellites = self._normalize_satellites(satellites, time_range=time_range)

        coarse_tr = self._coarse_time_range(time_range)

        sat_blocks = {
            float(coarse_tr.start_jd): {
                "timerange": coarse_tr,
                "satellites": satellites,
            }
        }

        track_factory = TrackFactory()
        tracks = track_factory.build(sat_blocks)

        geometry_ds = self._compute_geometry(tracks.dataset, observer, sat_blocks)
        geometry_ds = self._interpolate_geometry(geometry_ds, time_range)

        sat_signals = self._signal_factory.build(sat_blocks)

        # ── Resolve EIRP por constelação ──
        eirp_map: dict[str, float] = {}
        try:
            eirp_map = self._signal_factory.signal_repo.get_eirp_map()
        except Exception:
            self._logger.warning(
                "eirp_map_unavailable", msg="Não foi possível carregar EIRP. Usando 0 dBW."
            )

        # ── Resolve EIRP individual por satélite (opcional) ──
        # signals.json pode declarar EIRPs reais por NORAD ID; elas têm
        # precedência sobre o valor nominal da constelação.
        eirp_by_sat: dict[int, float] = {}
        try:
            get_eirp_by_sat = getattr(
                self._signal_factory.signal_repo, "get_eirp_by_satellite", None
            )
            if get_eirp_by_sat is not None:
                eirp_by_sat = get_eirp_by_sat()
        except Exception:
            self._logger.warning(
                "eirp_by_satellite_unavailable",
                msg="Não foi possível carregar EIRP individual; usando valor da constelação.",
            )

        assembled: list[SatelliteAssembled] = []
        freq_hz = self._signal_factory.frequency_grid_hz()
        freq_da = xr.DataArray(freq_hz, dims="frequency", name="frequency")

        for sat in satellites:
            norad_id = sat.norad_id
            signals = sat_signals.get(norad_id, [])
            if not signals:
                continue

            # ── Determina EIRP para este satélite ──
            # Precedência: EIRP individual (NORAD) > EIRP nominal da constelação.
            const = (sat.constellation or "").upper()
            eirp_individual = eirp_by_sat.get(norad_id)
            if eirp_individual is None:
                try:
                    eirp_individual = eirp_by_sat.get(int(norad_id))
                except (TypeError, ValueError):
                    eirp_individual = None
            eirp_dbw = eirp_individual if eirp_individual is not None else eirp_map.get(const, 0.0)
            eirp_linear = 10.0 ** (eirp_dbw / 10.0)  # dBW → Watts

            psd_arrays = []
            for sig in signals:
                psd_db = sig.psd
                psd_lin = delayed(lambda x: np.power(10.0, x / 10.0))(psd_db)
                psd_arrays.append(
                    da.from_delayed(
                        psd_lin,
                        shape=(freq_hz.size,),
                        dtype=np.float64,
                    )
                )

            # Soma contribuições espectrais e escala pela potência transmitida
            psd_sum = da.stack(psd_arrays, axis=0).sum(axis=0)
            psd_scaled = psd_sum * eirp_linear  # W/Hz absoluto

            psd_da = xr.DataArray(
                psd_scaled,
                dims="frequency",
                coords={"frequency": freq_da},
                attrs={"units": "W/Hz"},
            )

            geom_sat = geometry_ds.sel(satellite=norad_id)

            signals_ds = xr.Dataset(data_vars={"PSD": psd_da})
            assembled.append(
                SatelliteAssembled(
                    name=sat.name or str(norad_id),
                    tracks=geom_sat,
                    signals=signals_ds,
                )
            )

        if not assembled:
            msg = "No satellites could be assembled for TOD."
            raise ValueError(msg)

        return assembled

    def _normalize_satellites(
        self, satellites: list[Satellite], *, time_range: TimeRange
    ) -> list[Satellite]:
        """
        Accepts Domain Satellite or SatID (infra) and returns Domain Satellite list.
        """
        normalized: list[Satellite] = []
        tr_start, tr_end, _ = time_range.to_tuple()
        target = tr_start + (tr_end - tr_start) / 2

        for sat in satellites:
            if hasattr(sat, "element"):
                normalized.append(sat)
                continue

            elements = getattr(sat, "elements", None)
            if not elements:
                continue

            closest = min(
                (e for e in elements if getattr(e, "epoch", None) is not None),
                key=lambda e: abs((e.epoch - target).total_seconds()),
            )
            normalized.append(
                Satellite(
                    norad_id=sat.norad_id,
                    name=sat.name,
                    constellation=sat.constellation,
                    epoch=closest.epoch,
                    element=closest,
                )
            )

        if not normalized:
            msg = "No satellites with orbital elements for assembly."
            raise ValueError(msg)

        return normalized

    def _compute_geometry(self, tracks: xr.Dataset, observer: Observer, sat_blocks):
        try:
            return self._geometry.compute(tracks, observer, sat_blocks)
        except TypeError:
            return self._geometry.compute(tracks, observer)

    @staticmethod
    def _coarse_time_range(time_range: TimeRange) -> TimeRange:
        min_step = 1800.0
        start, end, _ = time_range.to_tuple()
        # Avoid coarse grids with a single time sample; they break xarray.interp()
        # and produce NaNs when the simulation interval is shorter than 30 minutes.
        try:
            duration_sec = float((end - start).sec)
        except Exception:
            duration_sec = float(time_range.max - time_range.min) * 86400.0
        if duration_sec <= min_step:
            return time_range

        step = max(min_step, time_range.resolution)
        if duration_sec / step < 1.0:
            return time_range
        return TimeRange.from_astropy(start, end, step)

    @staticmethod
    def _interpolate_geometry(geometry_ds: xr.Dataset, time_range: TimeRange) -> xr.Dataset:
        import warnings as _warnings

        target_time = time_range.to_skyfield().tt
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", message="All-NaN slice")
            _warnings.filterwarnings("ignore", message="divide by zero")
            _warnings.filterwarnings("ignore", message="invalid value")
            result = geometry_ds.interp(time=target_time, kwargs={"fill_value": "extrapolate"})
        # Reatribui coordenada time para UTC JDs (em vez de TT) para evitar
        # ruído de ponto flutuante na conversão TT→UTC posterior. Isso garante
        # que todos os tracks usem a mesma escala UTC, evitando duplicação do
        # eixo temporal no xr.concat.
        result = result.assign_coords(time=time_range.to_datetime64())
        return result

    @staticmethod
    def _normalize_frequency_range(
        frequency_range: FrequencyRange | None,
        num_channels: int | None,
    ) -> FrequencyRange | None:
        if frequency_range is None:
            return None
        if num_channels is None or num_channels <= 1:
            return frequency_range
        # Keep min/max and derive resolution from channelization.
        res = (frequency_range.max - frequency_range.min) / (num_channels - 1)
        return FrequencyRange.from_floats(
            frequency_range.min,
            frequency_range.max,
            res,
        )

    @staticmethod
    def _beam_frequency_grid(
        frequency_range: FrequencyRange | None,
        num_channels: int | None,
    ) -> np.ndarray | None:
        if frequency_range is None:
            return None
        if num_channels is not None and num_channels > 1:
            return np.linspace(frequency_range.min, frequency_range.max, num_channels)
        return frequency_range.value

    @staticmethod
    def _interpolate_beam_frequency(beam: Beam, freq_grid_mhz: np.ndarray) -> Beam:
        ds = beam.dataset
        if "frequency" not in ds.coords:
            return beam

        freq_units = str(ds["frequency"].attrs.get("units", "")).lower()
        freqs = np.asarray(ds["frequency"].values)
        use_hz = False
        if ("hz" in freq_units and "mhz" not in freq_units) or (
            freqs.size > 0 and np.nanmax(freqs) > 1e5
        ):
            use_hz = True

        target = freq_grid_mhz * 1e6 if use_hz else freq_grid_mhz

        # Se o beam tem apenas 1 frequência, usa broadcast simples em vez de interp
        # (evita divisão por zero e warnings do scipy)
        if len(freqs) == 1 and len(target) > 1:
            ds_out = ds.copy(deep=False)
            # Expande a dimensão frequency para o grid alvo
            expanded = ds_out["amplitude"].values  # shape (angle, 1, pol)
            new_amplitude = np.repeat(expanded, len(target), axis=1)
            ds_out = xr.Dataset(
                {"amplitude": (("angle", "frequency", "polarization"), new_amplitude)},
                coords={
                    "angle": ds.coords["angle"].values,
                    "frequency": target,
                    "polarization": ds.coords["polarization"].values,
                },
                attrs=ds.attrs,
            )
            ds_out["frequency"].attrs.update(ds["frequency"].attrs)
            return Beam(ds_out)

        ds_interp = ds.interp(frequency=target, kwargs={"fill_value": 0.0})
        ds_interp = ds_interp.assign_coords(frequency=target)
        ds_interp["frequency"].attrs.update(ds["frequency"].attrs)
        return Beam(ds_interp)


# ═══════════════════════════════════════════════════════════════════
# Helpers de grid para simulação multi-fonte
# ═══════════════════════════════════════════════════════════════════
# Nota (E1/E5): os helpers de grid (_compute_time_grid, _compute_freq_mhz,
# _beam_fwhm_deg) vivem em domain/simulation/grids.py (funções puras de
# domínio), reexportados acima por compatibilidade.
