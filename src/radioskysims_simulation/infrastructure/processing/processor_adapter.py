"""
Módulo: infrastructure/processing/processor_adapter.py

Adapter que faz ResourceAwareProcessor implementar ProcessorPort.

Desacopla a implementação concreta de infra do port abstrato de application.
Permite múltiplas implementações do ProcessorPort sem alterar application.

Copyright BINGO Collaboration
Last modified: 2026-11-05
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Generic, TypeVar

from radioskysims_sources.domain.signal_source import SignalSource
from radioskysims_simulation.application.ports.processing import (
    ProcessingResources,
    ProcessorPort,
)
from radioskysims_simulation.domain.simulation import SimulationResult
from radioskysims_simulation.infrastructure.processing.resource_processor import (
    ResourceAwareProcessor as _ResourceAwareProcessorImpl,
)

T = TypeVar("T")


class ResourceAwareProcessorAdapter(ProcessorPort[SimulationResult], Generic[T]):
    """
    Adapter que faz ResourceAwareProcessor implementar ProcessorPort.

    Wraps a instância existente de ResourceAwareProcessor e fornece a interface
    abstrata de ProcessorPort, permitindo DI e polimorfismo.
    """

    def __init__(self, processor: _ResourceAwareProcessorImpl) -> None:
        """
        Inicializa o adapter.

        Parameters
        ----------
        processor : ResourceAwareProcessor
            Instância existente do processador de recurso.
        """
        self._processor = processor

    @property
    def resources(self) -> ProcessingResources:
        """Recursos disponíveis (CPUs, memória)."""
        return self._processor.resources

    @property
    def progress_callback(self) -> Callable[[int, int], None] | None:
        """Callback opcional para progresso."""
        return self._processor.progress_callback

    @property
    def cpu_count(self) -> int:
        """Número de CPUs lógicos."""
        return self._processor.cpu_count

    @property
    def max_batch_mb(self) -> float:
        """Tamanho máximo de batch em MB."""
        return self._processor.max_batch_mb

    @property
    def parallel_workers(self) -> int:
        """Número de workers paralelos."""
        return self._processor.parallel_workers

    @property
    def dashboard_url(self) -> str | None:
        """URL do dashboard (dask ou None)."""
        return self._processor.dashboard_url

    def batch_budget_mb(self, single_source_mb: float) -> float:
        """Calcula orçamento para um batch."""
        return self._processor.batch_budget_mb(single_source_mb)

    def partition_sources(
        self, sources: Sequence[SignalSource]
    ) -> list[list[SignalSource]]:
        """Divide fontes em batches."""
        return self._processor.partition_sources(sources)

    def run_batch(
        self,
        batch: Iterable[T],
        processor_fn: Callable[[T], SimulationResult],
    ) -> list[SimulationResult]:
        """Executa batch com scheduler."""
        return self._processor.run_batch(batch, processor_fn)

    def close(self) -> None:
        """Limpa recursos."""
        self._processor.close()


__all__ = ["ResourceAwareProcessorAdapter"]
