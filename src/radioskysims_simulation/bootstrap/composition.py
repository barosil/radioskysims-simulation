"""\
Composition root de simulação (Fase 5–T1).

Cria `SimulationContext` e `SimOrchestrator` com dependências concretas.
Integração com DIContainer para injetar ProcessorPort.

Moveu-se para o pacote `radioskysims_simulation` para quebrar o ciclo
lazy `sim_orchestrator ↔ bootstrap.composition` do monólito.

Copyright BINGO Collaboration
Last modified: 2026-11-05
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from radioskysims_simulation.bootstrap.builders import (
    build_simulation_context_dependencies,
)
from radioskysims_simulation.bootstrap.dependencies import get_container


def create_simulation_context(
    telescope: str = "default",
    *,
    settings_file: str | Path | None = None,
):
    """Cria SimulationContext com dependências concretas atuais."""

    from radioskysims_simulation.application.services.sim_orchestrator import (
        SimulationContext,
    )

    deps = build_simulation_context_dependencies(
        telescope=telescope, settings_file=settings_file
    )
    return SimulationContext(**deps)


def create_sim_orchestrator(
    telescope: str = "default",
    *,
    settings_file: str | Path | None = None,
    processor_config: dict[str, Any] | None = None,
):
    """Cria SimOrchestrator totalmente montado a partir do settings.

    Parameters
    ----------
    telescope : str
        Nome do telescópio.
    settings_file : Path | None
        Arquivo de configuração.
    processor_config : dict[str, Any] | None
        Config para ProcessorPort (ex: max_batch_mb, cpu_count, etc).
        Se None, usa defaults do DIContainer.

    Returns
    -------
    SimOrchestrator
        Orquestrador totalmente injetado com ProcessorPort.
    """

    from radioskysims_simulation.application.services.sim_orchestrator import (
        SimOrchestrator,
    )

    context = create_simulation_context(
        telescope=telescope, settings_file=settings_file
    )

    # Criar ProcessorPort via DI container
    processor = None
    if processor_config is not None:
        container = get_container()
        processor = container.get_processor(**processor_config)

    return SimOrchestrator(context, processor=processor)


__all__ = [
    "create_sim_orchestrator",
    "create_simulation_context",
]
