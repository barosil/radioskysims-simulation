"""\
Módulo: radioskysims_simulation/application/services/diffuse_tod_runner.py

Descrição:
    Runner de fontes difusas (Galáxia via PySM3 e hidrogênio neutro) durante a
    simulação de TOD. Extraído de ``sim_orchestrator.py`` (SRP): concentra a
    geração de mapas HEALPix difusos (independente da antena, com cache por
    execução) e a projeção map → pointing → TOD (dependente da antena).

Copyright BINGO Collaboration
Last modified: 2026-11-06
"""

from __future__ import annotations

from typing import Any

import xarray as xr
from radioskysims_core.domain.common import TimeRange
from radioskysims_core.domain.geometry import compute_pointing_track
from radioskysims_sources.domain.radiosources.continuum import (
    Galaxy,
    NeutralHydrogen,
)

from radioskysims_simulation.domain.simulation.grids import _compute_time_grid


class DiffuseTODRunner:
    """Gera/projeta mapas difusos (Galáxia, HI) para o TOD da simulação.

    Responsabilidades (SRP):
    - gerar mapas HEALPix difusos (PySM3 para a Galáxia; campo gaussiano
      para o hidrogênio neutro);
    - cachear os mapas por execução (reutilizados entre antenas);
    - projetar mapas → pointing → TOD via ``TODOrchestrator`` e
      ``SkyMapTODProjector``.
    """

    def __init__(
        self,
        *,
        ctx: Any,
        galaxy_map_generator: Any,
        neutral_hydrogen_map_generator: Any,
        cosmology_parameters: Any,
        tod_orchestrator: Any,
        skymap_tod_projector: Any,
        logger: Any,
    ) -> None:
        self._ctx = ctx
        self._galaxy_map_generator = galaxy_map_generator
        self._neutral_hydrogen_map_generator = neutral_hydrogen_map_generator
        self._cosmology_parameters = cosmology_parameters
        self._tod_orchestrator = tod_orchestrator
        self._skymap_tod_projector = skymap_tod_projector
        self._logger = logger
        self._galaxy_maps_cache: tuple[Any, ...] | None = None
        self._neutral_hydrogen_maps_cache: tuple[Any, ...] | None = None

    def reset_caches(self) -> None:
        """Limpa os caches de mapas (início de uma nova execução do ``run``)."""
        self._galaxy_maps_cache = None
        self._neutral_hydrogen_maps_cache = None

    def _generate_galaxy_maps(
        self,
        *,
        nside: int,
        persist_skymaps: bool,
        skymap_overwrite: bool,
    ) -> tuple[Any, ...]:
        """Gera os mapas físicos da Via Láctea (independente da antena)."""
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

    def _generate_neutral_hydrogen_maps(
        self,
        *,
        nside: int,
        seed: int | None = None,
        persist_skymaps: bool = False,
        skymap_overwrite: bool = False,
    ) -> tuple[Any, ...]:
        """Gera os mapas cosmológicos de hidrogênio neutro (independente da antena)."""
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

    def run_diffuse_maps_for_antenna(
        self,
        *,
        maps: tuple[Any, ...],
        time_range: TimeRange,
        alt_deg: float | None,
        az_deg: float | None,
        use_beam: bool = True,
        source_name: str = "diffuse",
    ) -> xr.Dataset:
        """Projeta mapas HEALPix difusos no TOD de um pointing de antena.

        Operação source-agnostic: Galáxia, hidrogênio neutro e futuros
        componentes difusos compartilham o mesmo caminho
        map → pointing → TOD.
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

    def run_galaxy_for_antenna(
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
        """Gera (ou reutiliza) mapas da Galáxia e projeta no pointing da antena."""
        if self._galaxy_maps_cache is None:
            self._galaxy_maps_cache = self._generate_galaxy_maps(
                nside=nside,
                persist_skymaps=persist_skymaps,
                skymap_overwrite=skymap_overwrite,
            )

        return self.run_diffuse_maps_for_antenna(
            maps=self._galaxy_maps_cache,
            time_range=time_range,
            alt_deg=alt_deg,
            az_deg=az_deg,
            use_beam=use_beam,
            source_name="Galaxy",
        )

    def run_neutral_hydrogen_for_antenna(
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
        """Gera (ou reutiliza) mapas de HI e projeta no pointing da antena."""
        if self._neutral_hydrogen_maps_cache is None:
            self._neutral_hydrogen_maps_cache = self._generate_neutral_hydrogen_maps(
                nside=nside,
                seed=seed,
                persist_skymaps=persist_skymaps,
                skymap_overwrite=skymap_overwrite,
            )

        return self.run_diffuse_maps_for_antenna(
            maps=self._neutral_hydrogen_maps_cache,
            time_range=time_range,
            alt_deg=alt_deg,
            az_deg=az_deg,
            use_beam=use_beam,
            source_name="NeutralHydrogen",
        )


__all__ = ["DiffuseTODRunner"]
