"""
Pacote: radioskysims_simulation

Simulação do ecossistema RadioSkySim — o integrador. Contém TOD,
domínio de simulação (Simulation/SimulationResult/SimulationContext),
noise, RFI, mapmaking, orchestrators (SimOrchestrator/TODOrchestrator),
processamento ciente de recursos (ResourceAwareProcessor) e o
composition root de simulação (bootstrap).

Copyright BINGO Collaboration
"""

from __future__ import annotations

__version__ = "0.1.0"

from .application.orchestrators import MapMakingOrchestrator
from .domain.simulation import Simulation, SimulationResult
from .domain.tod import TODSimulator

__all__ = [
    "MapMakingOrchestrator",
    "ResourceAwareProcessor",
    "SimOrchestrator",
    "Simulation",
    "SimulationResult",
    "TODSimulator",
]


def __getattr__(name: str):
    if name == "SimOrchestrator":
        from .application.services.sim_orchestrator import SimOrchestrator

        return SimOrchestrator

    if name == "ResourceAwareProcessor":
        from .infrastructure.processing.resource_processor import (
            ResourceAwareProcessor,
        )

        return ResourceAwareProcessor

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
