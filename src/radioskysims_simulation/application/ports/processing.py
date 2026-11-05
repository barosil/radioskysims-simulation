"""
Módulo: application/ports/processing.py

Port abstrato para processamento consciente de recursos (scheduler).

Define o contrato que a infraestrutura de processamento deve cumprir,
permitindo que application/services trabalhe com processadores polimórficos
(threads, distributed, etc.) sem conhecer detalhes de implementação.

Copyright BINGO Collaboration
Last modified: 2026-11-05
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Generic, TypeVar

from radioskysims_sources.domain.signal_source import SignalSource

from radioskysims_simulation.application.processing_resources import (
    ProcessingResources,
)
from radioskysims_simulation.domain.simulation import SimulationResult

T = TypeVar("T")


class ProcessorPort(ABC, Generic[T]):
    """
    Port abstrato para um processador consciente de recursos.

    Contrato para execução de simulações em batches, respeitando limites
    de CPU e memória. Implementações concretas podem usar threads, processos,
    clusters distribuídos, etc.
    """

    @property
    @abstractmethod
    def resources(self) -> ProcessingResources:
        """Recursos disponíveis (CPUs, memória)."""
        ...

    @property
    @abstractmethod
    def progress_callback(self) -> Callable[[int, int], None] | None:
        """Callback opcional para atualizar progresso (itens_processados, total)."""
        ...

    @property
    @abstractmethod
    def cpu_count(self) -> int:
        """Número de CPUs lógicos disponíveis."""
        ...

    @property
    @abstractmethod
    def max_batch_mb(self) -> float:
        """Tamanho máximo de um batch em MB."""
        ...

    @property
    @abstractmethod
    def parallel_workers(self) -> int:
        """Número de workers paralelos recomendados."""
        ...

    @property
    @abstractmethod
    def dashboard_url(self) -> str | None:
        """URL do dashboard de monitoramento, se disponível (ex: Dask dashboard)."""
        ...

    @abstractmethod
    def batch_budget_mb(self, single_source_mb: float) -> float:
        """
        Calcula orçamento de memória para um batch.

        Parameters
        ----------
        single_source_mb : float
            Estimativa de memória para uma única fonte.

        Returns
        -------
        float
            Orçamento total de memória para o batch.
        """
        ...

    @abstractmethod
    def partition_sources(self, sources: Sequence[SignalSource]) -> list[list[SignalSource]]:
        """
        Divide fontes em batches respeitando limites de memória.

        Parameters
        ----------
        sources : Sequence[SignalSource]
            Lista de fontes a processar.

        Returns
        -------
        list[list[SignalSource]]
            Batches de fontes (cada sublist cabe na memória).
        """
        ...

    @abstractmethod
    def run_batch(
        self,
        batch: Iterable[T],
        processor_fn: Callable[[T], SimulationResult],
    ) -> list[SimulationResult]:
        """
        Executa um batch de itens usando scheduler.

        Parameters
        ----------
        batch : Iterable[T]
            Items a processar.
        processor_fn : Callable[[T], SimulationResult]
            Função que processa um único item.

        Returns
        -------
        list[SimulationResult]
            Resultados processados (mesma ordem que batch).
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Limpa recursos (fecha cluster distribuído, etc.)."""
        ...
