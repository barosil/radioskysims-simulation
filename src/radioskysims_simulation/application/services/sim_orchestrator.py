from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from radioskysims_core.config.config_loader import radio_min_flux_jy
from radioskysims_core.domain.common import (
    FrequencyRange,
    Observer,
    TimeRange,
)
from radioskysims_core.domain.geometry import compute_pointing_track

if TYPE_CHECKING:
    from radioskysims_core.application.adapters.external_geometry import (
        SkyfieldGeometryProvider,
    )
    from radioskysims_core.infrastructure.tracks import TrackFactory

from radioskysims_sources.application.ports.satellite import SatelliteRepositoryPort
from radioskysims_sources.application.services.catalog_services import (
    RadioSourceCatalogService,
)
from radioskysims_sources.application.services.local_sources_service import (
    LocalSourcesService,
)
from radioskysims_sources.application.services.satellite_service import (
    SatelliteService,
)
from radioskysims_sources.domain.radiosources.continuum import (
    Galaxy,
    NeutralHydrogen,
)
from radioskysims_sources.domain.signal import (
    SignalFactory,
    SignalRepositoryPort,
)

from radioskysims_simulation.application.ports.processing import (
    ProcessorPort,
)
from radioskysims_simulation.application.services.simulation_service import (
    SimulationService,
)
from radioskysims_simulation.application.services.tod_orchestrator import (
    TODOrchestrator,
)
from radioskysims_simulation.domain.simulation import (
    SimulationOrchestratorResult,
    SimulationResult,
)
from radioskysims_simulation.domain.simulation.grids import (
    _compute_freq_mhz,
    _compute_time_grid,
)
from radioskysims_simulation.domain.tod import TODSimulator


class _NoopLogger:
    """Fallback de logger para manter API quando logger nao for injetado."""

    def info(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def warning(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def error(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)


def _radio_query_for_summary(
    *,
    include_radio_sources: bool,
    pointings: list[tuple[float | None, float | None]],
    search_radius_deg: float = 20.0,
    min_flux_jy: float = 0.1,
) -> dict[str, Any] | None:
    """Query de rádio fontes do resumo, espelhando a simulação real.

    ``simulation_service.run_multi_source`` consulta o catálogo centrado no
    pointing da antena (``ra_center``/``dec_center``), com fallback
    ``(190.0, 0.0)`` quando o pointing é equatorial (modo trânsito), usando
    os valores efetivos de raio/fluxo aplicados na simulação (o corte de
    fluxo ``min_flux_jy`` vem do ``settings.toml``, [radiosources] query_str).
    Usa o pointing da primeira antena, pois o resumo tem um único bloco
    ``radiosources``.

    Returns
    -------
    dict | None
        ``None`` quando ``include_radio_sources=False``.
    """
    if not include_radio_sources:
        return None
    ra_center, dec_center = pointings[0] if pointings else (None, None)
    if ra_center is None:
        ra_center, dec_center = 190.0, 0.0
    return {
        "catalog": "NVSS",
        "ra_center_deg": ra_center,
        "dec_center_deg": dec_center,
        "search_radius_deg": search_radius_deg,
        "min_flux_jy": min_flux_jy,
    }


@dataclass(slots=True)
class SimulationContext:
    telescope_name: str
    observer: Observer
    time_range: TimeRange
    frequency_range: FrequencyRange
    db: Any

    satellite_repository: SatelliteRepositoryPort
    signal_repository: SignalRepositoryPort

    track_factory: TrackFactory
    geometry: SkyfieldGeometryProvider
    signal_factory: SignalFactory
    tod_simulator: TODSimulator

    satellite_service: SatelliteService
    simulation_service: SimulationService

    tod_orchestrator: TODOrchestrator | None = None

    galaxy_map_generator: Any | None = None
    neutral_hydrogen_map_generator: Any | None = None
    cosmology_parameters: Any | None = None
    skymap_repository: Any | None = None
    skymap_generator: Any | None = None
    skymap_tod_projector: Any | None = None

    local_sources_service: LocalSourcesService | None = None
    radio_source_catalog_service: RadioSourceCatalogService | None = None
    config_loader: Callable[[str], dict[str, Any] | None] | None = None
    logger: Any | None = None
    simulation_repository: Any | None = None


class SimOrchestrator:
    """Orquestrador de simulações de TOD.

    Responsabilidades principais:
    - carregar configurações do ``settings.toml``;
    - inicializar infraestrutura (DB, repositórios);
    - preparar fábricas e serviços de domínio;
    - expor um método ``run`` que dispara a simulação e retorna o
      :class:`~radioskysims_simulation.domain.simulation.SimulationResult`.
    """

    def __init__(
        self,
        context: SimulationContext,
        *,
        processor: ProcessorPort | None = None,
        processor_factory: Callable[..., ProcessorPort] | None = None,
    ) -> None:
        self._ctx = context
        # Corte de fluxo (Jy) para o catálogo de rádio — vem do settings.toml
        # ([radiosources] query_str, ex.: "flux > .1" → 0.1 Jy).
        self._radio_min_flux_jy = radio_min_flux_jy()

        self._local_sources_service = context.local_sources_service

        self._radio_source_catalog_service = context.radio_source_catalog_service

        self._tod_orchestrator = context.tod_orchestrator

        self._galaxy_map_generator = context.galaxy_map_generator

        self._skymap_repository = context.skymap_repository

        self._skymap_tod_projector = context.skymap_tod_projector
        # Cache temporário dos mapas galácticos durante uma execução.
        #
        # O céu difuso não depende da antena. Portanto, os mapas PySM3
        # devem ser gerados apenas uma vez por SimOrchestrator.run() e
        # reutilizados pelos diferentes pointings/beams.
        self._galaxy_maps_cache: tuple[Any, ...] | None = None

        self._config_loader = context.config_loader

        self._logger = context.logger or _NoopLogger()

        # Processor opcional injetado pelo usuário; factory do composition root
        # (bootstrap/dependencies.py) cria sob demanda quando não injetado.
        self._processor: ProcessorPort | None = processor
        self._processor_factory = processor_factory
        self._neutral_hydrogen_map_generator = getattr(
            context,
            "neutral_hydrogen_map_generator",
            None,
        )

        self._cosmology_parameters = getattr(
            context,
            "cosmology_parameters",
            None,
        )

        self._neutral_hydrogen_maps_cache: tuple[Any, ...] | None = None

    def _generate_galaxy_maps(
        self,
        *,
        nside: int,
        persist_skymaps: bool,
        skymap_overwrite: bool,
    ) -> tuple[Any, ...]:
        """
        Gera os mapas físicos da Via Láctea.

        Esta operação é independente da antena e deve ocorrer apenas
        uma vez por execução do SimOrchestrator.

        Parameters
        ----------
        nside:
            Resolução HEALPix.
        persist_skymaps:
            Persiste os mapas gerados no repository HDF5.
        skymap_overwrite:
            Permite sobrescrever mapas já persistidos.

        Returns
        -------
        tuple
            Mapas HEALPix da Galaxy, um para cada bin de frequência.
        """

        if self._galaxy_map_generator is None:
            msg = "Galaxy map generator is not configured."
            raise RuntimeError(msg)

        galaxy = Galaxy()

        galaxy_map_set = self._galaxy_map_generator.execute(
            source=galaxy,
            frequency_range=self._ctx.frequency_range,
            nside=nside,
            persist=persist_skymaps,
            overwrite=skymap_overwrite,
        )

        maps = tuple(galaxy_map_set.maps)

        self._logger.info(
            "galaxy_maps_generated",
            n_maps=len(maps),
            nside=nside,
            persisted=persist_skymaps,
        )

        return maps

    def _run_galaxy_maps_for_antenna(
        self,
        *,
        maps: tuple[Any, ...],
        time_range: TimeRange,
        alt_deg: float | None,
        az_deg: float | None,
        use_beam: bool = True,
    ) -> xr.Dataset:
        """
        Projeta mapas galácticos previamente gerados no pointing
        de uma antena específica.

        Ao contrário de ``_generate_galaxy_maps``, esta operação
        depende da antena e portanto é executada para cada pointing.
        """

        if self._tod_orchestrator is None:
            msg = "TODOrchestrator is not configured."
            raise RuntimeError(msg)

        if self._skymap_tod_projector is None:
            msg = "SkyMap TOD projector is not configured."
            raise RuntimeError(msg)

        if alt_deg is None or az_deg is None:
            msg = "Galaxy TOD requires antenna Alt/Az pointing."
            raise ValueError(msg)

        if not maps:
            msg = "Galaxy TOD requires at least one SkyMap."
            raise ValueError(msg)

        time_grid = _compute_time_grid(time_range)

        pointing_track = compute_pointing_track(
            observer=self._ctx.observer,
            time_grid=time_grid,
            alt_deg=alt_deg,
            az_deg=az_deg,
        )

        galaxy_tod = self._tod_orchestrator.run_maps(
            maps=maps,
            pointing_track=pointing_track,
            projector=self._skymap_tod_projector,
            use_beam=use_beam,
        )

        self._logger.info(
            "galaxy_tod_generated",
            n_maps=len(maps),
            pointing_alt=alt_deg,
            pointing_az=az_deg,
        )

        return galaxy_tod

    def _run_galaxy_maps_for_antenna(
        self,
        *,
        maps: tuple[Any, ...],
        time_range: TimeRange,
        alt_deg: float | None,
        az_deg: float | None,
        use_beam: bool = True,
    ) -> xr.Dataset:
        """
        Project previously generated Galaxy maps for one antenna.

        Compatibility wrapper around the generic diffuse-map path.
        """

        return self._run_diffuse_maps_for_antenna(
            maps=maps,
            time_range=time_range,
            alt_deg=alt_deg,
            az_deg=az_deg,
            use_beam=use_beam,
            source_name="Galaxy",
        )

    def _run_galaxy_for_antenna(
        self,
        *,
        time_range: TimeRange,
        alt_deg: float | None,
        az_deg: float | None,
        nside: int,
        persist_skymaps: bool,
        skymap_overwrite: bool,
        use_beam: bool = True,
    ) -> xr.Dataset:
        """
        Compatibility wrapper for the Galaxy -> TOD workflow.

        Galaxy maps are generated once per SimOrchestrator.run()
        and reused for all antenna pointings.
        """

        if self._galaxy_maps_cache is None:
            self._galaxy_maps_cache = self._generate_galaxy_maps(
                nside=nside,
                persist_skymaps=persist_skymaps,
                skymap_overwrite=skymap_overwrite,
            )

        return self._run_galaxy_maps_for_antenna(
            maps=self._galaxy_maps_cache,
            time_range=time_range,
            alt_deg=alt_deg,
            az_deg=az_deg,
            use_beam=use_beam,
        )

    # ----------------------------------------------------------
    # Os métodos já existentes continuam abaixo:
    #
    # - from_settings(...)
    # - default_constellations(...)
    # - default_time_range(...)
    # - _resolve_antenna_config(...)
    # - run(...)
    # - _run_with_processor(...)
    # - etc.
    # ----------------------------------------------------------
    @property
    def context(self) -> SimulationContext:
        """Contexto interno com todos os componentes da simulação."""

        return self._ctx

    @property
    def processor(self) -> ProcessorPort | None:
        """Processor computacional associado ao orquestrador."""
        return self._processor

    @property
    def dashboard_url(self) -> str | None:
        """URL do dashboard do Processor, quando disponível."""
        if self._processor is None:
            return None
        return self._processor.dashboard_url

    def _resolve_processor(
        self,
        *,
        max_batch_mb: float | None,
        cpu_count: int | None,
        memory_mb: float | None,
        dask_scheduler: str | None,
        dask_processes: bool | None,
        processor_diagnostics: bool,
    ) -> ProcessorPort:
        """Retorna o Processor injetado ou cria via factory do composition root."""
        if self._processor is not None:
            return self._processor

        # A application não importa o adapter concreto (TD-001): a factory é
        # injetada pelo composition root (bootstrap/dependencies.py).
        if self._processor_factory is None:
            msg = (
                "Processor não injetado e nenhuma factory configurada. Use "
                "SimOrchestrator(..., processor=...) ou "
                "processor_factory=get_container().get_processor."
            )
            raise RuntimeError(msg)

        processor = self._processor_factory(
            max_batch_mb=max_batch_mb,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            dask_scheduler=dask_scheduler,
            dask_processes=dask_processes,
            diagnostics_enabled=processor_diagnostics,
            logger=self._logger,
        )

        self._processor = processor

        return processor

    def run_maps(
        self,
        *,
        maps: Iterable[Any],
        pointing_track: xr.Dataset,
        use_beam: bool = True,
    ) -> xr.Dataset:
        """Gera TOD a partir de mapas HEALPix difusos.

        Esta fachada pública mantém o projector concreto encapsulado no
        composition root da simulação e delega a geração física do TOD ao
        TODOrchestrator configurado.

        Parameters
        ----------
        maps:
            Mapas HEALPix a serem projetados no pointing do radiotelescópio.
        pointing_track:
            Trajetória de apontamento do radiotelescópio.
        use_beam:
            Se True, aplica a resposta do beam durante a projeção.

        Returns
        -------
        xr.Dataset
            Time-Ordered Data gerado a partir dos mapas.

        Raises
        ------
        RuntimeError
            Se o TODOrchestrator ou o projector mapa→TOD não estiverem
            configurados.
        """
        if self._tod_orchestrator is None:
            msg = "TODOrchestrator is not configured."
            raise RuntimeError(msg)

        if self._skymap_tod_projector is None:
            msg = "SkyMap TOD projector is not configured."
            raise RuntimeError(msg)

        return self._tod_orchestrator.run_maps(
            maps=maps,
            pointing_track=pointing_track,
            projector=self._skymap_tod_projector,
            use_beam=use_beam,
        )

    @property
    def simulation_repository(self) -> Any | None:
        """Repositório de simulações (persistência dos resumos)."""
        return getattr(self._ctx, "simulation_repository", None)

    def save_result(
        self,
        result: Any,
        *,
        filepath: str | None = None,
        saved: bool = False,
        registered: bool = True,
    ) -> int:
        """Persiste o resumo de uma simulação no banco.

        Parameters
        ----------
        result:
            ``SimulationOrchestratorResult`` (ou objeto com ``summary()``).
        filepath:
            Caminho do dataset persistido, se aplicável.
        saved:
            Marca o dataset como salvo em arquivo.
        registered:
            Marca o registro como completo.

        Returns
        -------
        int
            Id do registro criado no repositório.

        Raises
        ------
        RuntimeError
            Se o repositório não estiver configurado no contexto.
        """
        repo = self.simulation_repository
        if repo is None:
            msg = "SimulationRepository não configurado no SimulationContext."
            raise RuntimeError(msg)
        return repo.save(
            result,
            filepath=filepath,
            saved=saved,
            registered=registered,
        )

    def shutdown(self) -> None:
        """Encerra recursos do scheduler distribuído da última simulação.

        Quando ``use_processor=True`` com ``dask_scheduler="distributed"``,
        o ``ResourceAwareProcessor`` mantém um ``LocalCluster``/``Client``
        ativo enquanto o dataset estiver *lazy* (o ``compute()``/``save()``
        precisa do cluster). Após materializar/persistir o resultado, chame
        este método para liberar workers, memória e o dashboard:

        .. code-block:: python

            result = orchestrator.run_simulation(...)
            result.compute()          # ou result.save_result(...)
            orchestrator.shutdown()

        Para schedulers ``"threads"``/``"synchronous"`` o método é no-op
        (o processor já restaura o scheduler ao fim de ``run_batches``).
        """
        if self._processor is not None:
            self._processor.shutdown()
            self._processor = None

    # ------------------------------------------------------------------
    # Fábrica principal a partir do settings.toml
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(
        cls,
        telescope: str = "default",
        *,
        settings_file: str | Path | None = None,
        processor: ProcessorPort | None = None,
    ) -> SimOrchestrator:
        """Cria um :class:`SimOrchestrator` usando um arquivo de settings.

        Parameters
        ----------
        telescope:
            Nome do radiotelescópio definido no arquivo de settings
            (ex.: ``"default"``).
        settings_file:
            Arquivo de settings nomeado (ex.: ``"settings_planning.toml"``),
            resolvido em relação à raiz do projeto. Quando informado, passa a
            ser o arquivo ativo para todo o contexto da simulação.
        """
        from radioskysims_simulation.bootstrap.composition import create_simulation_context

        ctx = create_simulation_context(
            telescope=telescope,
            settings_file=settings_file,
        )
        return cls(
            ctx,
            processor=processor,
        )

    # ------------------------------------------------------------------
    # Helpers de configuração
    # ------------------------------------------------------------------
    def default_constellations(self) -> list[str]:
        """Constelações padrão lidas de ``settings.toml``.

        Procura a chave ``satellites.constellations``; se não existir,
        retorna ``["GPS"]``.
        """

        if self._config_loader is None:
            return ["GPS"]

        cfg = self._config_loader("satellites") or {}
        const = cfg.get("constellations")
        if not const:
            return ["GPS"]
        # Garante lista de strings
        if isinstance(const, str):
            return [const]
        return list(const)

    def default_time_range(self) -> TimeRange:
        """Retorna o :class:`TimeRange` padrão do observer."""

        return self._ctx.time_range

    @staticmethod
    def _resolve_antenna_config(
        antenna: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Resolve beam, backend, num_channels e pointing de uma antena.

        Args:
            antenna: Objeto Antenna do RadiotelescopeSpec.

        Returns:
            tuple: (beam, backend, num_channels, pointing_ra_deg,
                    pointing_dec_deg, pointing_alt_deg, pointing_az_deg)
        """
        from radioskysims_beam.infrastructure.beams.loaders.analytical import (
            AnalyticalBeamLoader,
            BeamProfile,
        )
        from radioskysims_beam.infrastructure.beams.loaders.csv import CSVBeamLoader

        backend = getattr(antenna, "backend", None)
        pointing = getattr(antenna, "pointing", None)
        num_channels = getattr(backend, "num_channels", None) if backend else None

        beam = None
        beam_spec = getattr(antenna, "beam", None)
        if beam_spec is not None:
            if beam_spec.beam_pattern_file is not None:
                beam = CSVBeamLoader(file=beam_spec.beam_pattern_file).load()
            elif beam_spec.profile is not None:
                width_deg = beam_spec.beam_width_deg or 10.0
                loader = AnalyticalBeamLoader(
                    beam_profile=BeamProfile(beam_spec.profile),
                    sigma=width_deg,
                    frequency=1000.0,
                    angular_range=(-90.0, 90.0),
                    angular_resolution=1.0,
                )
                beam = loader.load()

        _ra_deg = getattr(pointing, "ra_deg", None)
        _dec_deg = getattr(pointing, "dec_deg", None)
        _alt_deg = getattr(pointing, "alt_deg", None)
        _az_deg = getattr(pointing, "az_deg", None)

        return beam, backend, num_channels, _ra_deg, _dec_deg, _alt_deg, _az_deg

    # ------------------------------------------------------------------
    # Execução de simulação
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        constellations: Iterable[str] | None = None,
        time_range: TimeRange | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
        include_satellites: bool = True,
        include_local_sources: bool = False,
        include_radio_sources: bool = False,
        radio_search_radius_deg: float | None = None,
        radio_min_flux_jy: float | None = None,
        include_galaxy: bool = False,
        galaxy_nside: int = 64,
        persist_skymaps: bool = False,
        skymap_overwrite: bool = False,
        sun_flux_mean_window_s: float | None = None,
        use_processor: bool = False,
        max_batch_mb: float | None = None,
        cpu_count: int | None = None,
        memory_mb: float | None = None,
        dask_scheduler: str | None = None,
        dask_processes: bool | None = None,
        batch_output_dir: str | None = None,
        processor_diagnostics: bool = False,
        include_neutral_hydrogen: bool = False,
        neutral_hydrogen_nside: int = 64,
        neutral_hydrogen_seed: int | None = None,
    ) -> SimulationResult:
        """Roda uma simulação completa de TOD.

        Suporta múltiplas antenas/beams: itera sobre ``rt.antennas``,
        executa uma simulação para cada uma e consolida os resultados
        na dimensão ``pointing``.

        Os três tipos de fonte são independentes entre si — você pode rodar
        **apenas satélites**, **apenas fontes locais** (Sol) ou **apenas
        fontes de rádio** (NVSS), em qualquer combinação:

        .. code-block:: python

            # Apenas satélites (padrão)
            orch.run(constellations=["GPS"], use_processor=True)

            # Apenas fontes locais (Sol)
            orch.run(include_satellites=False, include_local_sources=True,
                     use_processor=True)

            # Apenas fontes de rádio (NVSS)
            orch.run(include_satellites=False, include_radio_sources=True,
                     use_processor=True)

        Quando ``use_processor=True``, utiliza o
        :class:`~radioskysims.infrastructure.processing.ResourceAwareProcessor`
        para dividir as fontes em batches seguros para memória. Se a **saída
        total** estimada exceder a memória disponível, os batches são
        gravados em disco (streaming) e reabertos como dataset *lazy* (dask)
        — o dataset completo nunca é materializado em RAM, evitando o
        travamento da máquina.

        Parameters
        ----------
        constellations:
            Lista de constelações GNSS (ex.: ["GPS", "GALILEO"]).
            Se ``None``, usa o valor padrão de ``settings.toml``. Ignorado
            quando ``include_satellites=False``.
        time_range:
            Intervalo temporal da simulação. Se ``None``, utiliza o
            ``timeframe`` do observer do radiotelescópio.

        mueller:
            Matriz de Mueller opcional com dims (polarization, stokes).

        combine_polarization:
            Estratégia para combinar polarizações no ``TODSimulator``.
        include_satellites:
            Se ``True`` (padrão), inclui satélites GNSS das constelações
            selecionadas. Se ``False``, não consulta o banco de satélites —
            útil para simulações apenas com fontes locais/rádio.
        include_local_sources:
            Se ``True``, inclui fontes locais (Sun) na simulação.

        include_radio_sources:
            Se ``True``, inclui fontes de rádio (NVSS) na simulação.

        include_galaxy:
            Se ``True``, inclui a contribuição difusa da Via Láctea
            através de mapas HEALPix gerados com PySM3.

        galaxy_nside:
            Resolução HEALPix utilizada para os mapas galácticos.

        persist_skymaps:
            Se ``True``, persiste em HDF5 os mapas galácticos gerados.

        skymap_overwrite:
            Se ``True``, permite sobrescrever mapas galácticos já
            persistidos no repository.

        sun_flux_mean_window_s:
            Largura da janela temporal (s) para a média do fluxo solar.
            Se ``None``, usa o tempo aproximado de um trânsito do Sol
            pelo feixe.

        use_processor:
            Se ``True``, usa o ``ResourceAwareProcessor`` para executar
            a simulação em batches seguros para memória.

        max_batch_mb:
            Limite de memória por batch em MB (válido apenas quando
            ``use_processor=True``). Se ``None``, detecta automaticamente.

        cpu_count:
            Número de CPUs a usar (válido apenas quando
            ``use_processor=True``). Se ``None``, detecta automaticamente.

        memory_mb:
            Memória total disponível em MB (válido apenas quando
            ``use_processor=True``). Se ``None``, detecta automaticamente.

        dask_scheduler:
            Scheduler dask do ``ResourceAwareProcessor``:
            ``"threads"``, ``"synchronous"`` ou ``"distributed"``.
            Se ``None``, usa ``settings.toml [processing]``.

        dask_processes:
            Estratégia do scheduler distribuído (``True``/``False``/``None``).
            Se ``None``, usa ``settings.toml [processing]`` (ou o padrão do
            processor: processos separados).
        batch_output_dir:
            Diretório para streaming dos batches a disco (``use_processor=True``).
            Se ``None`` e a saída total estimada exceder a memória disponível,
            um diretório temporário é criado automaticamente em
            ``data/TOD/batch_parts_<timestamp>``.

        Returns
        -------
        SimulationResult
            Resultado contendo o TOD consolidado para todas as antenas.

            Quando ``include_galaxy=True``, a contribuição difusa
            galáctica é somada ao TOD das fontes discretas.
        """

        if time_range is None:
            time_range = self.default_time_range()

        # Valida que ao menos um tipo de fonte foi selecionado.
        if not (
            include_satellites
            or include_local_sources
            or include_radio_sources
            or include_galaxy
            or include_neutral_hydrogen
        ):
            msg = (
                "Nenhum tipo de fonte selecionado. Defina pelo menos um de "
                "include_satellites, include_local_sources, "
                "include_radio_sources, include_galaxy ou "
                "include_neutral_hydrogen."
            )
            raise ValueError(msg)

        if include_satellites:
            if constellations is None:
                constellations = self.default_constellations()
        else:
            constellations = []

        const_list = list(constellations)

        # ----------------------------------------------------------
        # Configuração de processamento
        # ----------------------------------------------------------
        if self._config_loader is not None:
            _proc_cfg = self._config_loader("processing") or {}

            if max_batch_mb is None:
                max_batch_mb = _proc_cfg.get("max_batch_mb")

            if cpu_count is None:
                cpu_count = _proc_cfg.get("cpu_count")

            if dask_scheduler is None:
                dask_scheduler = _proc_cfg.get(
                    "dask_scheduler",
                    "threads",
                )

            if dask_processes is None and _proc_cfg.get("dask_processes") is not None:
                dask_processes = bool(_proc_cfg["dask_processes"])

        if dask_scheduler is None:
            dask_scheduler = "threads"

        # Configuração de busca de radio sources.
        # Precedência aplicada: argumento explícito > default histórico.
        _effective_radio_search_radius_deg = (
            float(radio_search_radius_deg) if radio_search_radius_deg is not None else 20.0
        )
        _effective_radio_min_flux_jy = (
            float(radio_min_flux_jy) if radio_min_flux_jy is not None else self._radio_min_flux_jy
        )

        # Cada execução possui seu próprio conjunto de mapas galácticos.
        # O cache existe apenas para reutilização entre antenas do mesmo run().
        self._galaxy_maps_cache = None
        self._neutral_hydrogen_maps_cache = None

        # ----------------------------------------------------------
        # Cronômetro da simulação (bloco Computation do resumo)
        # ----------------------------------------------------------
        import time as _time
        from datetime import UTC, datetime

        _t0 = _time.monotonic()

        _start_utc = datetime.now(UTC).isoformat()

        # Propaga o time_range da simulação para o repositório de satélites,
        # para que a busca de TLEs (SpaceTrack) use a época da simulação em vez
        # de "agora" — evita "0 TLEs" para satélites cujos TLEs estão próximos
        # da janela de simulação mas fora da janela relativa "now".
        sat_repo = getattr(self._ctx, "satellite_repository", None)
        if include_satellites and sat_repo is not None and time_range is not None:
            try:
                sat_repo.tr = time_range

            except AttributeError:
                self._logger.info(
                    "sim_orchestrator_satellite_repo_tr_unsupported",
                    detail=(
                        "repositório de satélites não aceita o atributo 'tr'; "
                        "epoch da simulação não será propagada ao repositório"
                    ),
                )

        # ----------------------------------------------------------
        # Logging inicial
        # ----------------------------------------------------------
        self._logger.info(
            "sim_orchestrator_run",
            constellations=const_list,
            time_range=str(time_range),
            telescope=self._ctx.telescope_name,
            include_satellites=include_satellites,
            include_local_sources=include_local_sources,
            include_radio_sources=include_radio_sources,
            include_galaxy=include_galaxy,
            include_neutral_hydrogen=include_neutral_hydrogen,
            galaxy_nside=(galaxy_nside if include_galaxy else None),
            persist_skymaps=(
                persist_skymaps if (include_galaxy or include_neutral_hydrogen) else False
            ),
            use_processor=use_processor,
        )

        # Obtém satélites (somente se incluídos)
        sats: list = []
        if const_list:
            sats = self._ctx.satellite_service.get_by_constellations(const_list)

        # Ausência de satélites: permite simulação sem satélites quando há
        # outra contribuição (fontes locais/rádio ou céu difuso).
        has_non_satellite_contribution = bool(
            include_local_sources
            or include_radio_sources
            or include_galaxy
            or include_neutral_hydrogen
        )

        if not sats and not has_non_satellite_contribution:
            msg = (
                "Nenhum satélite disponível após tentativa de "
                "atualização para constelações %s. "
                "Verifique configuração do banco/credenciais."
            )
            self._logger.error(
                "satellites_unavailable",
                constellations=const_list,
            )
            raise RuntimeError(msg % const_list)

        if not sats and has_non_satellite_contribution:
            self._logger.info(
                "non_satellite_simulation",
                include_local_sources=include_local_sources,
                include_radio_sources=include_radio_sources,
                include_galaxy=include_galaxy,
                include_neutral_hydrogen=include_neutral_hydrogen,
                reason="no_satellites_available",
                constellations=const_list,
            )

        # ----------------------------------------------------------
        # Radiotelescópio
        # ----------------------------------------------------------
        from radioskysims_core.config.config_loader import (
            build_radiotelescope_from_settings,
        )

        rt = build_radiotelescope_from_settings(self._ctx.telescope_name)

        antennas = list(rt.antennas)
        # ----------------------------------------------------------
        # Executa uma simulação para cada antena
        # ----------------------------------------------------------
        results: list[SimulationResult] = []

        beams: list = []

        pointings: list[
            tuple[
                float | None,
                float | None,
            ]
        ] = []

        for idx, antenna in enumerate(antennas):
            (
                beam,
                backend,
                num_channels,
                ra_deg,
                dec_deg,
                alt_deg,
                az_deg,
            ) = self._resolve_antenna_config(antenna)

            beams.append(beam)

            pointings.append(
                (
                    ra_deg,
                    dec_deg,
                )
            )

            self._logger.info(
                "antenna_simulation",
                index=idx,
                n_antennas=len(antennas),
                beam_profile=getattr(
                    getattr(
                        antenna,
                        "beam",
                        None,
                    ),
                    "profile",
                    None,
                ),
                pointing_alt=(alt_deg),
                pointing_az=(az_deg),
                include_galaxy=(include_galaxy),
            )

            # ------------------------------------------------------
            # TOD de fontes discretas
            # ------------------------------------------------------
            source_result: SimulationResult | None = None

            # Executa o pipeline discreto quando existem satélites
            # ou quando outras fontes discretas foram solicitadas.
            has_discrete_contribution = bool(sats or include_local_sources or include_radio_sources)

            if has_discrete_contribution:
                if use_processor:
                    source_result = self._run_with_processor(
                        satellites=sats,
                        time_range=time_range,
                        observer=self._ctx.observer,
                        frequency_range=self._ctx.frequency_range,
                        num_channels=num_channels,
                        beam=beam,
                        pointing_ra_deg=ra_deg,
                        pointing_dec_deg=dec_deg,
                        pointing_alt_deg=alt_deg,
                        pointing_az_deg=az_deg,
                        mueller=mueller,
                        combine_polarization=combine_polarization,
                        include_local_sources=include_local_sources,
                        include_radio_sources=include_radio_sources,
                        radio_search_radius_deg=_effective_radio_search_radius_deg,
                        radio_min_flux_jy=_effective_radio_min_flux_jy,
                        sun_flux_mean_window_s=sun_flux_mean_window_s,
                        max_batch_mb=max_batch_mb,
                        cpu_count=cpu_count,
                        memory_mb=memory_mb,
                        dask_scheduler=dask_scheduler,
                        dask_processes=dask_processes,
                        batch_output_dir=batch_output_dir,
                        processor_diagnostics=processor_diagnostics,
                    )

                else:
                    source_result = self._ctx.simulation_service.run_multi_source(
                        satellites=sats,
                        time_range=time_range,
                        observer=self._ctx.observer,
                        frequency_range=self._ctx.frequency_range,
                        num_channels=num_channels,
                        beam=beam,
                        pointing_ra_deg=ra_deg,
                        pointing_dec_deg=dec_deg,
                        pointing_alt_deg=alt_deg,
                        pointing_az_deg=az_deg,
                        mueller=mueller,
                        combine_polarization=combine_polarization,
                        include_local_sources=include_local_sources,
                        include_radio_sources=include_radio_sources,
                        radio_search_radius_deg=_effective_radio_search_radius_deg,
                        radio_min_flux_jy=_effective_radio_min_flux_jy,
                        sun_flux_mean_window_s=sun_flux_mean_window_s,
                    )

            # ------------------------------------------------------
            # TOD difuso da Galaxy
            # ------------------------------------------------------
            # ------------------------------------------------------
            # TOD de componentes difusos
            # ------------------------------------------------------
            diffuse_tods: list[xr.Dataset] = []

            if include_galaxy:
                galaxy_tod = self._run_galaxy_for_antenna(
                    time_range=time_range,
                    alt_deg=alt_deg,
                    az_deg=az_deg,
                    nside=galaxy_nside,
                    persist_skymaps=persist_skymaps,
                    skymap_overwrite=skymap_overwrite,
                    use_beam=True,
                )

                diffuse_tods.append(galaxy_tod)

            if include_neutral_hydrogen:
                neutral_hydrogen_tod = self._run_neutral_hydrogen_for_antenna(
                    time_range=time_range,
                    alt_deg=alt_deg,
                    az_deg=az_deg,
                    nside=neutral_hydrogen_nside,
                    seed=neutral_hydrogen_seed,
                    persist_skymaps=persist_skymaps,
                    skymap_overwrite=skymap_overwrite,
                    use_beam=True,
                )

                diffuse_tods.append(neutral_hydrogen_tod)

            diffuse_tod: xr.Dataset | None = None

            if len(diffuse_tods) == 1:
                diffuse_tod = diffuse_tods[0]

            elif len(diffuse_tods) > 1:
                if self._tod_orchestrator is None:
                    msg = "TODOrchestrator is not configured."
                    raise RuntimeError(msg)

                diffuse_tod = self._tod_orchestrator.combine_maps(
                    map_tods=diffuse_tods,
                )

            # ------------------------------------------------------
            # Combina as contribuições solicitadas
            # ------------------------------------------------------
            # ------------------------------------------------------
            # Combina as contribuições solicitadas
            # ------------------------------------------------------
            if source_result is not None and diffuse_tod is not None:
                if self._tod_orchestrator is None:
                    msg = "TODOrchestrator is not configured."
                    raise RuntimeError(msg)

                combined_dataset = self._tod_orchestrator.combine(
                    source_tod=source_result.dataset,
                    map_tod=diffuse_tod,
                )

                result = SimulationResult(
                    dataset=combined_dataset,
                    sources=source_result.sources,
                    satellites=source_result.satellites,
                    time_range=time_range,
                )

                self._logger.info(
                    "diffuse_tod_combined",
                    antenna_index=idx,
                    include_galaxy=include_galaxy,
                    include_neutral_hydrogen=(include_neutral_hydrogen),
                )

            elif source_result is not None:
                # Fluxo histórico:
                # satellites / Sun / radio sources only.
                result = source_result

            elif diffuse_tod is not None:
                diffuse_tod.attrs["contributions"] = [
                    "skymap",
                ]

                result = SimulationResult(
                    dataset=diffuse_tod,
                    sources=[],
                    satellites=[],
                    time_range=time_range,
                )

                self._logger.info(
                    "diffuse_only_tod_generated",
                    antenna_index=idx,
                    include_galaxy=include_galaxy,
                    include_neutral_hydrogen=(include_neutral_hydrogen),
                )

            else:
                msg = "No TOD contribution was generated."
                raise RuntimeError(msg)

            results.append(result)
            # ----------------------------------------------------------
        # Consolidação entre antenas
        # ----------------------------------------------------------
        if len(results) == 1:
            sim_result = results[0]

        else:
            datasets = [result.dataset for result in results]

            renamed = []

            for i, ds in enumerate(datasets):
                _ds = ds.assign_coords(pointing=[i])

                renamed.append(_ds)

            merged = xr.concat(
                renamed,
                dim="pointing",
                compat="override",
                coords="minimal",
                combine_attrs=("drop_conflicts"),
            )

            sim_result = SimulationResult(
                dataset=merged,
                sources=(results[0].sources),
                satellites=(results[0].satellites),
                time_range=(time_range),
            )

        # ----------------------------------------------------------
        # Resumo/configuração
        # ----------------------------------------------------------
        from radioskysims_simulation.domain.simulation.summary import (
            assemble_config,
        )

        _computing_time_s = _time.monotonic() - _t0

        config = assemble_config(
            telescope_name=(self._ctx.telescope_name),
            observer=(self._ctx.observer),
            frequency_range=(self._ctx.frequency_range),
            antennas=antennas,
            beams=beams,
            num_channels=(num_channels),
            time_range=(time_range),
            satellites=sats,
            include_satellites=include_satellites,
            include_local_sources=include_local_sources,
            include_radio_sources=include_radio_sources,
            radio_query=_radio_query_for_summary(
                include_radio_sources=include_radio_sources,
                pointings=pointings,
                search_radius_deg=_effective_radio_search_radius_deg,
                min_flux_jy=self._radio_min_flux_jy,
            ),
            start_utc=(_start_utc),
            computing_time_s=(_computing_time_s),
            delayed=True,
        )

        # ----------------------------------------------------------
        # Backward compatibility
        # ----------------------------------------------------------
        config["telescope"] = self._ctx.telescope_name

        config["constellations"] = const_list

        config["n_antennas"] = len(antennas)

        config["n_satellites"] = len(sats)

        # Novos metadados Galaxy
        config["include_galaxy"] = include_galaxy

        config["galaxy_nside"] = galaxy_nside if include_galaxy else None

        config["persist_skymaps"] = (
            persist_skymaps if (include_galaxy or include_neutral_hydrogen) else False
        )

        config["include_neutral_hydrogen"] = include_neutral_hydrogen

        config["neutral_hydrogen_nside"] = (
            neutral_hydrogen_nside if include_neutral_hydrogen else None
        )

        config["neutral_hydrogen_seed"] = (
            neutral_hydrogen_seed if include_neutral_hydrogen else None
        )

        return SimulationOrchestratorResult(
            simulation_result=(sim_result),
            config=config,
            events=None,
        )

    def _run_with_processor(
        self,
        *,
        satellites: list,
        time_range: TimeRange,
        observer: Observer,
        frequency_range: FrequencyRange | None,
        num_channels: int | None,
        beam: Any,
        pointing_ra_deg: float | None,
        pointing_dec_deg: float | None,
        pointing_alt_deg: float | None,
        pointing_az_deg: float | None,
        mueller: xr.DataArray | None,
        combine_polarization: str,
        include_local_sources: bool,
        include_radio_sources: bool,
        radio_search_radius_deg: float,
        radio_min_flux_jy: float,
        sun_flux_mean_window_s: float | None = None,
        max_batch_mb: float | None = None,
        cpu_count: int | None = None,
        memory_mb: float | None = None,
        dask_scheduler: str | None = None,
        dask_processes: bool | None = None,
        batch_output_dir: str | None = None,
        processor_diagnostics: bool = False,
    ) -> SimulationResult:
        """Executa a simulação usando o ResourceAwareProcessor em batches.

        Em vez de processar todas as fontes de uma vez, o processador:
        1. Detecta recursos disponíveis (CPUs, memória)
        2. Estima o consumo de memória por fonte
        3. Divide as fontes em batches cabíveis na memória
        4. Executa cada batch sequencialmente
        5. Consolida resultados com merge seguro (compat='override')

        Se a **saída total** estimada (T_antenna + power, todas as fontes)
        exceder a memória utilizável, os batches são gravados em disco
        (``batch_output_dir``) e reabertos como dataset *lazy* (dask) — o
        dataset completo nunca é materializado em RAM.
        """
        # Pré-configura o SimulationService com a mesma grade espectral
        # utilizada pelo caminho sem ResourceAwareProcessor.
        svc = self._ctx.simulation_service

        from radioskysims_beam.domain.beam import BeamOps

        if frequency_range is None and beam is not None:
            frequency_range = svc._frequency_range_from_beam(beam)

        frequency_range = svc._normalize_frequency_range(
            frequency_range,
            num_channels,
        )

        if beam is not None:
            freq_grid = svc._beam_frequency_grid(
                frequency_range,
                num_channels,
            )

            if freq_grid is not None:
                beam = svc._interpolate_beam_frequency(
                    beam,
                    freq_grid,
                )

        if frequency_range is not None:
            svc._signal_factory.fr = frequency_range

        if beam is not None:
            svc._tod_simulator.beam = BeamOps.normalize(beam)

        # Monta satélites com tracks
        if not svc._satellites_are_assembled(satellites):
            satellites = svc._assemble_satellites(
                satellites=satellites,
                time_range=time_range,
                observer=observer,
            )

        from radioskysims_sources.domain.signal_source import SatelliteSignalSource

        sources: list = [SatelliteSignalSource(sat) for sat in satellites]

        # Extrai FWHM do beam para filtro angular e para a janela padrão do Sol.
        # Nota: beam.params().fwhm pode conter NaN (ex.: beam CSV do BINGO com
        # 20/68 NaNs) — usa np.nanmean e descarta valores não finitos.
        _beam_fwhm_deg: float | None = None
        if beam is not None:
            try:
                _fwhm_vals = beam.params().fwhm
                _fwhm_mean = (
                    float(np.nanmean(_fwhm_vals))
                    if hasattr(_fwhm_vals, "mean")
                    else float(_fwhm_vals)
                )
                if np.isfinite(_fwhm_mean) and _fwhm_mean > 0:
                    _beam_fwhm_deg = _fwhm_mean
            except Exception as exc:
                self._logger.warning(
                    "sim_orchestrator_beam_fwhm_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        # Monta fontes locais
        source_assembler = svc._source_assembler
        if include_local_sources and source_assembler is not None:
            try:
                time_grid = _compute_time_grid(time_range)
                freq_mhz = _compute_freq_mhz(frequency_range)
                local_srcs = source_assembler.assemble_local_sources(
                    observer=observer,
                    time_range=time_range,
                    time_grid=time_grid,
                    freq_mhz=freq_mhz,
                    sun_flux_mean_window_s=sun_flux_mean_window_s,
                    beam_fwhm_deg=_beam_fwhm_deg,
                )
                sources.extend(local_srcs)
                self._logger.info("local_sources_assembled", n_sources=len(local_srcs))
            except Exception as exc:
                self._logger.warning("local_sources_assembly_failed", error=str(exc))

        # Monta fontes de rádio
        if include_radio_sources and source_assembler is not None:
            try:
                from radioskysims_core.domain.geometry import compute_pointing_track

                time_grid = _compute_time_grid(time_range)
                freq_mhz = _compute_freq_mhz(frequency_range)

                # Pointing track (RA/Dec temporal) para filtrar fontes no feixe
                pointing_track: xr.Dataset | None = None

                if pointing_alt_deg is not None and pointing_az_deg is not None:
                    try:
                        pointing_track = compute_pointing_track(
                            observer=observer,
                            time_grid=time_grid,
                            alt_deg=pointing_alt_deg,
                            az_deg=pointing_az_deg,
                        )
                    except Exception as exc:
                        self._logger.warning(
                            "processor_pointing_track_failed",
                            error=str(exc),
                        )

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

                radio_srcs = source_assembler.assemble_radio_sources(
                    observer=observer,
                    time_range=time_range,
                    time_grid=time_grid,
                    freq_mhz=freq_mhz,
                    ra_center_deg=_ra_center,
                    dec_center_deg=_dec_center,
                    search_radius_deg=radio_search_radius_deg,
                    min_flux_jy=radio_min_flux_jy,
                    pointing_track=pointing_track,
                    pointing_ra_deg=pointing_ra_deg,
                    pointing_dec_deg=pointing_dec_deg,
                    beam_fwhm_deg=_beam_fwhm_deg,
                )

                sources.extend(radio_srcs)

                self._logger.info(
                    "radio_sources_assembled",
                    n_sources=len(radio_srcs),
                )

            except Exception as exc:
                self._logger.warning(
                    "radio_sources_assembly_failed",
                    error=str(exc),
                )

        if not sources:
            msg = "Nenhuma fonte disponível para simulação."
            raise RuntimeError(msg)

        self._logger.info(
            "processor_run_start",
            total_sources=len(sources),
            max_batch_mb=max_batch_mb,
            cpu_count=cpu_count,
        )

        processor = self._resolve_processor(
            max_batch_mb=max_batch_mb,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            dask_scheduler=dask_scheduler,
            dask_processes=dask_processes,
            processor_diagnostics=processor_diagnostics,
        )

        resources = processor.resources

        batches = processor.partition_sources(
            sources=sources,
            time_range=time_range,
            frequency_range=frequency_range,
            num_channels=num_channels,
        )

        # ── Decisão de streaming a disco ──────────────────────────────
        # A consolidação em memória segura o dataset FINAL inteiro (todas as
        # fontes). Para simulações grandes (ex.: 24 h @ 10 s × 5000 canais)
        # isso excede a RAM e trava a máquina. Se a saída total estimada não
        # cabe na memória utilizável, grava cada batch em disco e reabre como
        # dataset lazy (dask) — memória limitada a um batch por vez.
        from pathlib import Path as _Path

        from radioskysims_simulation.application.processing_resources import (
            MemoryEstimator,
        )

        total_out_mb = MemoryEstimator.estimate_total_output_mb(
            sources=sources,
            time_range=time_range,
            frequency_range=frequency_range,
            num_channels=num_channels,
        )
        _stream_to_disk = batch_output_dir is not None or (
            total_out_mb > resources.usable_memory_mb
        )
        if _stream_to_disk and batch_output_dir is None:
            from datetime import datetime as _dt

            batch_output_dir = str(
                _Path("data/TOD") / f"batch_parts_{_dt.now().strftime('%Y%m%dT%H%M%S')}"
            )
        if _stream_to_disk:
            self._logger.warning(
                "processor_stream_to_disk",
                total_output_mb=round(total_out_mb, 0),
                usable_memory_mb=round(resources.usable_memory_mb, 0),
                batch_output_dir=batch_output_dir,
                n_batches=len(batches),
            )
            if total_out_mb > resources.usable_memory_mb:
                print(
                    f"\n💾 Saída total ~{total_out_mb / 1024:.1f} GB não cabe em RAM "
                    f"(usable ~{resources.usable_memory_mb / 1024:.1f} GB). "
                    f"Gravando {len(batches)} batches em:\n   {batch_output_dir}\n"
                    "O dataset final será reaberto de forma LAZY (dask) — memória "
                    "limitada a um batch por vez.\n"
                )
            else:
                print(
                    f"\n💾 Streaming por batches solicitado — gravando "
                    f"{len(batches)} batches em:\n   {batch_output_dir}\n"
                    "O dataset final será reaberto de forma LAZY (dask).\n"
                )

        def _sim_factory(batch_sources: list) -> Any:
            from radioskysims_simulation.domain.simulation import Simulation

            return Simulation(
                sources=batch_sources,
                time_range=time_range,
                tod_simulator=svc._tod_simulator,
                pointing_ra_deg=pointing_ra_deg,
                pointing_dec_deg=pointing_dec_deg,
                pointing_alt_deg=pointing_alt_deg,
                pointing_az_deg=pointing_az_deg,
                mueller=mueller,
                combine_polarization=combine_polarization,
            )

        dataset = processor.run_batches(
            batches=batches,
            simulation_factory=_sim_factory,
            batch_output_dir=batch_output_dir,
        )

        self._logger.info(
            "processor_run_complete",
            total_sources=len(sources),
            **processor.summary,
        )

        return SimulationResult(
            dataset=dataset,
            sources=sources,
            satellites=list(satellites),
            time_range=time_range,
        )

    def _run_diffuse_maps_for_antenna(
        self,
        *,
        maps: tuple[Any, ...],
        time_range: TimeRange,
        alt_deg: float | None,
        az_deg: float | None,
        use_beam: bool = True,
        source_name: str = "diffuse",
    ) -> xr.Dataset:
        """
        Project diffuse HEALPix maps into TOD for one antenna pointing.

        This operation is source-agnostic. Galaxy, neutral hydrogen and
        future diffuse components share the same map -> pointing -> TOD path.
        """

        if self._tod_orchestrator is None:
            msg = "TODOrchestrator is not configured."
            raise RuntimeError(msg)

        if self._skymap_tod_projector is None:
            msg = "SkyMap TOD projector is not configured."
            raise RuntimeError(msg)

        if alt_deg is None or az_deg is None:
            msg = f"{source_name} TOD requires antenna Alt/Az pointing."
            raise ValueError(msg)

        if not maps:
            msg = f"{source_name} TOD requires at least one SkyMap."
            raise ValueError(msg)

        time_grid = _compute_time_grid(time_range)

        pointing_track = compute_pointing_track(
            observer=self._ctx.observer,
            time_grid=time_grid,
            alt_deg=alt_deg,
            az_deg=az_deg,
        )

        diffuse_tod = self._tod_orchestrator.run_maps(
            maps=maps,
            pointing_track=pointing_track,
            projector=self._skymap_tod_projector,
            use_beam=use_beam,
        )

        self._logger.info(
            "diffuse_tod_generated",
            source=source_name,
            n_maps=len(maps),
            pointing_alt=alt_deg,
            pointing_az=az_deg,
        )

        return diffuse_tod

    def _generate_neutral_hydrogen_maps(
        self,
        *,
        nside: int,
        seed: int | None = None,
        persist_skymaps: bool = False,
        skymap_overwrite: bool = False,
    ) -> tuple[Any, ...]:
        """
        Gera os mapas cosmológicos de hidrogênio neutro.

        Como os mapas não dependem da antena, eles podem ser
        reutilizados pelos diferentes pointings da mesma execução.
        """

        if self._neutral_hydrogen_map_generator is None:
            msg = "Neutral hydrogen map generator is not configured."
            raise RuntimeError(msg)

        if self._cosmology_parameters is None:
            msg = "Cosmology parameters are not configured."
            raise RuntimeError(msg)

        source = NeutralHydrogen()

        map_set = self._neutral_hydrogen_map_generator.execute(
            source=source,
            cosmology=self._cosmology_parameters,
            frequency_range=self._ctx.frequency_range,
            nside=nside,
            seed=seed,
            persist=persist_skymaps,
            overwrite=skymap_overwrite,
        )

        maps = tuple(map_set.maps)

        self._logger.info(
            "neutral_hydrogen_maps_generated",
            n_maps=len(maps),
            nside=nside,
            seed=seed,
            persisted=persist_skymaps,
        )
        return maps

    def _run_neutral_hydrogen_for_antenna(
        self,
        *,
        time_range: TimeRange,
        alt_deg: float | None,
        az_deg: float | None,
        nside: int,
        seed: int | None = None,
        persist_skymaps: bool = False,
        skymap_overwrite: bool = False,
        use_beam: bool = True,
    ) -> xr.Dataset:
        """
        Gera/reutiliza mapas de HI e os projeta no pointing da antena.
        """

        if self._neutral_hydrogen_maps_cache is None:
            self._neutral_hydrogen_maps_cache = self._generate_neutral_hydrogen_maps(
                nside=nside,
                seed=seed,
                persist_skymaps=persist_skymaps,
                skymap_overwrite=skymap_overwrite,
            )

        return self._run_diffuse_maps_for_antenna(
            maps=self._neutral_hydrogen_maps_cache,
            time_range=time_range,
            alt_deg=alt_deg,
            az_deg=az_deg,
            use_beam=use_beam,
            source_name="NeutralHydrogen",
        )
