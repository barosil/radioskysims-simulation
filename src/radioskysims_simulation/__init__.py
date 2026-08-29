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

from .domain.simulation import Simulation, SimulationResult
from .domain.tod import TODSimulator

__all__ = ["Simulation", "SimulationResult", "TODSimulator"]
