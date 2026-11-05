"""
Módulo: application/ports/__init__.py

Puertos (contracts) abstratos da camada application do simulador.

Copyright BINGO Collaboration
Last modified: 2026-11-05
"""

from __future__ import annotations

from .processing import ProcessorPort, ProcessingResources

__all__ = ["ProcessorPort", "ProcessingResources"]
