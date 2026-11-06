"""\
Módulo: packages/radioskysims-simulation/src/radioskysims_simulation/application/orchestrators/mapmaking_orchestrator.py

Descrição:
    Fachada de alto nível (UX — Onda 3/R6) para mapmaking (reconstrução de
    mapas a partir de TOD).

    Encapsula o ``MapMakingService`` (GenerateSkyMapsFromTODUseCase +
    persistência opcional). O usuário entrega TOD + pointing e recebe o
    resultado — sem conhecer mapmaker/repositórios.

    Uso (notebook):
        >>> from radioskysims_simulation import MapMakingOrchestrator
        >>> orch = MapMakingOrchestrator(service=mm_service)
        >>> res = orch.run(tod=tod, pointing_track=track, nside=256)

Copyright BINGO Collaboration
Last modified: 2026-09-03
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import xarray as xr

from radioskysims_simulation.application.services.mapmaking_service import (
    MapMakingService,
    MapMakingServiceResult,
)

__all__ = ["MapMakingOrchestrator"]

# Factory do mapmaker injetada pelo composition root (TD-001).
_DEFAULT_MAPMAKER_FACTORY: Callable[[], Any] | None = None


def set_default_mapmaker_factory(factory: Callable[[], Any] | None) -> None:
    """Registra a factory do mapmaker binned (bootstrap)."""
    global _DEFAULT_MAPMAKER_FACTORY
    _DEFAULT_MAPMAKER_FACTORY = factory


def _mapmaker() -> Any:
    """Retorna o mapmaker injetado ou falha com erro claro."""
    if _DEFAULT_MAPMAKER_FACTORY is None:
        msg = (
            "Mapmaker não configurado: chame set_default_mapmaker_factory() "
            "no composition root."
        )
        raise RuntimeError(msg)
    return _DEFAULT_MAPMAKER_FACTORY()


class MapMakingOrchestrator:
    """Orquestrador/fachada pública de mapmaking (TOD → mapas)."""

    def __init__(self, service: MapMakingService) -> None:
        self._service = service

    # ──────────────────────────────────────────────────────────
    # Fábricas
    # ──────────────────────────────────────────────────────────
    @classmethod
    def from_service(cls, service: MapMakingService) -> "MapMakingOrchestrator":
        """Embrulha um ``MapMakingService`` já montado (composition)."""
        return cls(service)

    @classmethod
    def from_generator(
        cls,
        generator: object,
        repository: object | None = None,
    ) -> "MapMakingOrchestrator":
        """Monta o serviço a partir de um ``GenerateSkyMapsFromTODUseCase``."""
        return cls(
            MapMakingService(generator=generator, repository=repository)  # type: ignore[arg-type]
        )

    @classmethod
    def default(cls) -> "MapMakingOrchestrator":
        """Monta com o mapmaker binned padrão (``HealpyBinnedMapMakerAdapter``).

        Sem repositório de persistência (``persist=True`` exige passar um via
        ``from_generator``/``from_service``).
        """
        from radioskysims_simulation.application.mapmaking.generate_from_tod import (
            GenerateSkyMapsFromTODUseCase,
        )

        generator = GenerateSkyMapsFromTODUseCase(mapmaker=_mapmaker())
        return cls.from_generator(generator)

    # ──────────────────────────────────────────────────────────
    # Fluxo
    # ──────────────────────────────────────────────────────────
    def run(
        self,
        *,
        tod: xr.Dataset,
        pointing_track: xr.Dataset,
        nside: int,
        frame: str = "equatorial",
        persist: bool = False,
        simulation_id: int | None = None,
        key_prefix: str = "mapmaking",
        overwrite: bool = False,
    ) -> MapMakingServiceResult:
        """Reconstrói mapas a partir do TOD e do pointing.

        Args:
            tod: Dataset TOD (tempo × frequência × polarização).
            pointing_track: Dataset de pointing (RA/Dec por tempo).
            nside: Resolução HEALPix dos mapas reconstruídos.
            frame: Sistema de referência ("equatorial" por padrão).
            persist: Se ``True``, persiste via repositório.
            simulation_id: ID da simulação (para persistência).
            key_prefix: Prefixo das chaves persistidas.
            overwrite: Sobrescreve registros persistidos.
        """
        return self._service.run(
            tod=tod,
            pointing_track=pointing_track,
            nside=nside,
            frame=frame,
            persist=persist,
            simulation_id=simulation_id,
            key_prefix=key_prefix,
            overwrite=overwrite,
        )

    def __repr__(self) -> str:
        return f"<MapMakingOrchestrator service={type(self._service).__name__}>"
