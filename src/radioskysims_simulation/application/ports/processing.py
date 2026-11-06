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
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import xarray as xr
from radioskysims_core.domain.common import FrequencyRange, TimeRange
from radioskysims_sources.domain.signal_source import SignalSource

from radioskysims_simulation.application.processing_resources import (
    ProcessingResources,
)
from radioskysims_simulation.domain.simulation import Simulation


@dataclass
class SourceBatch:
    """Lote de fontes para processamento seguro em memória (contrato do port)."""

    sources: list[SignalSource]
    estimated_mb: float
    batch_index: int


class ProcessorPort(ABC):
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
    def partition_sources(
        self,
        sources: Sequence[SignalSource],
        time_range: TimeRange,
        frequency_range: FrequencyRange | None = None,
        num_channels: int | None = None,
    ) -> list[SourceBatch]:
        """
        Divide fontes em batches respeitando limites de memória.

        Parameters
        ----------
        sources : Sequence[SignalSource]
            Lista de fontes a processar.
        time_range : TimeRange
            Intervalo temporal.
        frequency_range : FrequencyRange | None
            Range de frequências.
        num_channels : int | None
            Número de canais.

        Returns
        -------
        list[SourceBatch]
            Batches de fontes com estimativa de memória.
        """
        ...

    @abstractmethod
    def run_batches(
        self,
        batches: list[SourceBatch],
        simulation_factory: Callable[[list[SignalSource]], Simulation],
        *,
        batch_output_dir: str | None = None,
        part_engine: str = "h5netcdf",
    ) -> xr.Dataset:
        """
        Executa os batches sequencialmente e consolida os resultados.

        Parameters
        ----------
        batches : list[SourceBatch]
            Batches de fontes a processar.
        simulation_factory : Callable[[list[SignalSource]], Simulation]
            Função que recebe uma lista de fontes e retorna uma
            :class:`Simulation` pronta para executar.
        batch_output_dir : str | None
            Diretório opcional para streaming por batches (grava parcial).
        part_engine : str
            Backend xarray para os arquivos parciais (padrão "h5netcdf").

        Returns
        -------
        xr.Dataset
            Dataset consolidado com todas as fontes processadas.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Limpa recursos (fecha cluster distribuído, etc.)."""
        ...
