"""\
Composition root de simulação (Fase 5).

Cria `SimulationContext` e `SimOrchestrator` com dependências concretas.
Moveu-se para o pacote `radioskysims_simulation` para quebrar o ciclo
lazy `sim_orchestrator ↔ bootstrap.composition` do monólito.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from pathlib import Path

from radioskysims_simulation.bootstrap.builders import (
    build_simulation_context_dependencies,
)


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
):
    """Cria SimOrchestrator totalmente montado a partir do settings."""

    from radioskysims_simulation.application.services.sim_orchestrator import (
        SimOrchestrator,
    )

    context = create_simulation_context(
        telescope=telescope, settings_file=settings_file
    )
    return SimOrchestrator(context)


__all__ = ["create_sim_orchestrator", "create_simulation_context"]
