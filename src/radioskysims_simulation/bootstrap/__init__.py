"""
Módulo: bootstrap

Builders e composition root do pacote radioskysims_simulation.

Camada responsável por instanciar dependências e montar o contexto de simulação.

Copyright BINGO Collaboration
Last modified: 2026-11-05
"""

from __future__ import annotations

from .composition import create_sim_orchestrator, create_simulation_context
from .dependencies import (
    DIContainer,
    configure_default_processor,
    get_container,
    reset_container,
)

__all__ = [
    "create_sim_orchestrator",
    "create_simulation_context",
    "DIContainer",
    "get_container",
    "reset_container",
    "configure_default_processor",
]
