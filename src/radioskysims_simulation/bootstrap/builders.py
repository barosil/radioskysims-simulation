"""Builders de componentes do pacote (composition root — strategy).
Extraído de ``radioskysims.bootstrap.builders`` na Fase 5.
Copyright BINGO Collaboration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from radioskysims_beam.bootstrap.builders import _load_beam_from_csv
from radioskysims_core.bootstrap.builders import build_database, build_logging
from radioskysims_core.config.config_loader import build_radiotelescope_from_settings, load_config
from radioskysims_core.domain.common import FrequencyRange
from radioskysims_core.domain.geometry import SkyfieldGeometryProvider
from radioskysims_core.domain.tracks.factory import TrackFactory
from radioskysims_core.skyfield_loader import get_skyfield_dir
from radioskysims_radiotelescope.bootstrap.builders import build_configuration
from radioskysims_skymap.infrastructure.skymap.providers.pysm import PySMSkyMapGeneratorAdapter
from radioskysims_skymap.infrastructure.skymap.repository.hdf5 import HDF5SkyMapRepository
from radioskysims_sources.application.services.catalog_services import RadioSourceCatalogService
from radioskysims_sources.application.services.satellite_service import SatelliteService
from radioskysims_sources.application.services.source_assembler import SourceAssembler
from radioskysims_sources.application.skymap.generate_continuum import GenerateGalaxyMapsUseCase
from radioskysims_sources.bootstrap.builders import (
    build_cosmology_components,
    build_local_sources_service,
    build_repositories,
)
from radioskysims_sources.domain.signal import SignalFactory
from radioskysims_sources.infrastructure.db.radiosources_repository import (
    PostgresRadioSourcesRepository,
)
from radioskysims_sources.infrastructure.db.satellite.satelliterepository import SatelliteRepository
from radioskysims_sources.infrastructure.db.signal.repository import SignalRepository
from radioskysims_sources.infrastructure.external.radiosource_track_provider import (
    SkyfieldRadioSourceTrackProvider,
)
from radioskysims_sources.infrastructure.external.radiosources_ingest import VizierTapIngestAdapter

from radioskysims_simulation.application.services.simulation_service import SimulationService
from radioskysims_simulation.application.services.tod_orchestrator import TODOrchestrator
from radioskysims_simulation.domain.tod import TODSimulator
from radioskysims_simulation.infrastructure.db.simulation_repository import SimulationRepository
from radioskysims_simulation.infrastructure.skymap.tod_projector import HealpySkyMapTODProjector


def build_skymap_components(
    *,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """
    Constroi componentes concretos do pipeline de SkyMap/Galaxy.

    O caminho padrao do repository e resolvido a partir da raiz
    do projeto, e nao a partir do current working directory.

    Isso garante o mesmo destino para execucoes via terminal,
    notebooks, testes ou interfaces:

        <project_root>/data/aux/skymaps
    """

    if repository_root is None:
        project_root = Path(__file__).resolve().parents[3]

        resolved_repository_root = project_root / "data" / "aux" / "skymaps"

    else:
        resolved_repository_root = Path(repository_root)

    skymap_repository = HDF5SkyMapRepository(
        root=resolved_repository_root,
    )

    skymap_generator = PySMSkyMapGeneratorAdapter(
        unit="uK_CMB",
        field="I",
    )

    galaxy_map_generator = GenerateGalaxyMapsUseCase(
        generator=skymap_generator,
        repository=skymap_repository,
    )

    skymap_tod_projector = HealpySkyMapTODProjector()

    return {
        "skymap_repository": skymap_repository,
        "skymap_generator": skymap_generator,
        "galaxy_map_generator": galaxy_map_generator,
        "skymap_tod_projector": skymap_tod_projector,
    }


def build_domain_components(
    *,
    frequency_range: FrequencyRange,
    signal_repository: SignalRepository,
    effective_area_m2: float | None = None,
) -> dict[str, object]:
    """Constroi componentes de dominio usados pela simulacao."""

    track_factory = TrackFactory(skyfield_data_path=str(get_skyfield_dir()))

    geometry = SkyfieldGeometryProvider(track_factory.skyfield.ts)

    from radioskysims_sources.infrastructure.signal.gnsstools_psd import (
        GnsstoolsAnalyticPsdProvider,
    )

    signal_factory = SignalFactory(
        frequency_range=frequency_range,
        signal_repo=signal_repository,
        psd_provider=GnsstoolsAnalyticPsdProvider(),
    )

    tod_simulator = TODSimulator(
        beam=None,
        gain=1.0,
        effective_area_m2=effective_area_m2,
    )

    return {
        "track_factory": track_factory,
        "geometry": geometry,
        "signal_factory": signal_factory,
        "tod_simulator": tod_simulator,
    }


def build_services(
    *,
    satellite_repository: SatelliteRepository,
    geometry: SkyfieldGeometryProvider,
    signal_factory: SignalFactory,
    tod_simulator: TODSimulator,
    satellite_logger: Any,
    simulation_logger: Any,
    source_assembler: Any | None = None,
) -> dict[str, object]:
    """
    Constroi services de aplicacao para o contexto de simulacao.

    O mesmo TODSimulator e compartilhado entre o SimulationService
    legado e o TODOrchestrator, garantindo configuracao consistente
    de beam, gain e demais parametros instrumentais.
    """

    satellite_service = SatelliteService(
        satellite_repository,
        logger=satellite_logger,
    )

    simulation_service = SimulationService(
        satellite_service=satellite_service,
        geometry=geometry,
        signal_factory=signal_factory,
        tod_simulator=tod_simulator,
        telescope_loader=(build_radiotelescope_from_settings),
        beam_loader_factory=(_load_beam_from_csv),
        source_assembler=source_assembler,
        logger=simulation_logger,
    )

    # Novo caminho de orquestracao de TOD.
    #
    # O TODOrchestrator nao substitui o SimulationService.
    # Ambos compartilham a mesma instancia de TODSimulator:
    #
    # SimulationService
    #     -> fluxo legado/multi-source
    #
    # TODOrchestrator
    #     -> sources
    #     -> maps
    #     -> sources + maps
    tod_orchestrator = TODOrchestrator(
        simulator=tod_simulator,
    )

    return {
        "satellite_service": satellite_service,
        "simulation_service": simulation_service,
        "tod_orchestrator": tod_orchestrator,
    }


def build_simulation_context_dependencies(
    telescope: str = "default",
    *,
    settings_file: str | Path | None = None,
) -> dict[str, object]:
    """
    Monta dependencias concretas para o contexto de simulacao.

    Mantem o wiring atual para preservar comportamento e compatibilidade.

    O TODOrchestrator e construido pelo ``build_services`` e passa
    automaticamente a integrar o dicionario de dependencias por meio
    de ``**services``.
    """

    cfg = build_configuration(telescope=telescope, settings_file=settings_file)
    loggers = build_logging()

    db = build_database()

    repos = build_repositories(db)

    domain_components = build_domain_components(
        frequency_range=cfg["frequency_range"],
        signal_repository=repos["signal_repository"],
        effective_area_m2=(cfg["antenna"].effective_area_m2 if "antenna" in cfg else None),
    )

    skymap_components = build_skymap_components()

    try:
        cosmology_components = build_cosmology_components(
            skymap_repository=skymap_components["skymap_repository"],
        )
    except ModuleNotFoundError as exc:
        if exc.name != "pyccl":
            raise
        cosmology_components = {
            "cosmology_parameters": None,
            "neutral_hydrogen_map_generator": None,
        }

    local_sources_service = build_local_sources_service(
        db,
        solar_flux_source=str((load_config("solar_flux") or {}).get("source", "srd")),
    )

    # --------------------------------------------------
    # RadioSourceCatalogService
    # --------------------------------------------------

    vizier_tap = VizierTapIngestAdapter(tap_url=("https://tapvizier.u-strasbg.fr/TAPVizieR/tap"))

    pg_radio_repo = PostgresRadioSourcesRepository(session_factory=db.session)

    radio_source_catalog_service = RadioSourceCatalogService(
        ingest_port=vizier_tap,
        repository=pg_radio_repo,
        track_provider=SkyfieldRadioSourceTrackProvider(),
    )

    # --------------------------------------------------
    # SourceAssembler
    #
    # Unifica tracks/sinais das fontes discretas:
    # satelites, fontes locais e radiofontes.
    # --------------------------------------------------
    source_assembler = SourceAssembler(
        local_sources_service=(local_sources_service),
        radio_source_catalog_service=(radio_source_catalog_service),
    )

    # --------------------------------------------------
    # Application services
    #
    # Inclui agora:
    #
    # - SatelliteService
    # - SimulationService
    # - TODOrchestrator
    # --------------------------------------------------
    services = build_services(
        satellite_repository=(repos["satellite_repository"]),
        geometry=(domain_components["geometry"]),
        signal_factory=(domain_components["signal_factory"]),
        tod_simulator=(domain_components["tod_simulator"]),
        satellite_logger=(loggers["satellite_logger"]),
        simulation_logger=(loggers["simulation_logger"]),
        source_assembler=(source_assembler),
    )

    # ``cfg`` vem de build_configuration e inclui ``antenna`` (objeto concreto
    # da antena), que o ``SimulationContext`` não aceita — o orquestrador
    # re-deriva o radiotelescópio a partir do ``telescope_name``. Filtra para
    # manter apenas as chaves de contexto.
    cfg_ctx = {k: v for k, v in cfg.items() if k != "antenna"}

    return {
        **cfg_ctx,
        "db": db,
        **repos,
        **domain_components,
        **skymap_components,
        **cosmology_components,
        **services,
        "local_sources_service": local_sources_service,
        "radio_source_catalog_service": radio_source_catalog_service,
        "config_loader": load_config,
        "logger": loggers["orchestrator_logger"],
        "simulation_repository": SimulationRepository(db),
    }
