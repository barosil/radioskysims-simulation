"""
Módulo: bootstrap/dependencies.py

Dependency Injection Container para o simulador.

Define a estratégia de injeção de dependências através de um container simples.
Permite configurar diferentes implementações de ports abstratos (ex: ProcessorPort)
sem modificar o código de application/services.

Padrão: Composition Root com factory functions.

Copyright BINGO Collaboration
Last modified: 2026-11-05
"""

from __future__ import annotations

from typing import Any

from radioskysims_simulation.application.ports.processing import ProcessorPort
from radioskysims_simulation.application.processing_resources import (
    ProcessingResources,
)
from radioskysims_simulation.infrastructure.processing.processor_adapter import (
    ResourceAwareProcessorAdapter,
)
from radioskysims_simulation.infrastructure.processing.resource_processor import (
    ResourceAwareProcessor,
)


class DIContainer:
    """Container de injeção de dependências para o simulador."""

    def __init__(self) -> None:
        """Inicializa o container com defaults."""
        self._processor_factory: type[ProcessorPort] | None = None
        self._processor_instance: ProcessorPort | None = None

    def set_processor_factory(
        self,
        factory: type[ProcessorPort],
    ) -> None:
        """Define a factory para criar ProcessorPort.

        Parameters
        ----------
        factory : type[ProcessorPort]
            Classe que implementa ProcessorPort.
        """
        self._processor_factory = factory
        self._processor_instance = None  # Reset cached instance

    def get_processor(
        self,
        *,
        max_batch_mb: float | None = None,
        cpu_count: int | None = None,
        memory_mb: float | None = None,
        dask_scheduler: str | None = None,
        dask_processes: bool | None = None,
        diagnostics_enabled: bool = False,
        logger: Any | None = None,
    ) -> ProcessorPort:
        """Cria ou retorna uma instância de ProcessorPort.

        Parameters
        ----------
        max_batch_mb : float | None
            Tamanho máximo de batch em MB.
        cpu_count : int | None
            Número de CPUs a usar.
        memory_mb : float | None
            Memória máxima em MB.
        dask_scheduler : str | None
            Scheduler dask ("threads", "processes", "synchronous").
        dask_processes : bool | None
            Se True, usar processos; se False, threads.
        diagnostics_enabled : bool
            Se True, ativar logging de diagnóstico.
        logger : Any | None
            Logger para o processador.

        Returns
        -------
        ProcessorPort
            Instância do processador configurado.
        """
        if self._processor_instance is not None:
            return self._processor_instance

        # Detectar recursos se não fornecidos
        if cpu_count is None or memory_mb is None:
            resources = ProcessingResources.detect(
                cpu_count=cpu_count,
                memory_mb=memory_mb,
            )
        else:
            resources = ProcessingResources(
                cpu_count=cpu_count,
                available_memory_mb=memory_mb,
            )

        # Criar instância de ResourceAwareProcessor
        processor_impl = ResourceAwareProcessor(
            resources=resources,
            max_batch_mb=max_batch_mb,
            dask_scheduler=dask_scheduler or "threads",
            dask_processes=dask_processes,
            diagnostics_enabled=diagnostics_enabled,
            logger=logger,
        )

        # Wrap com adapter para cumprir ProcessorPort
        processor_port = ResourceAwareProcessorAdapter(processor_impl)

        self._processor_instance = processor_port
        return processor_port

    def clear_processor(self) -> None:
        """Limpa a instância de processador (para testes ou reload)."""
        if self._processor_instance is not None:
            self._processor_instance.close()
        self._processor_instance = None


# Singleton global do container
_global_container: DIContainer | None = None


def get_container() -> DIContainer:
    """Retorna o container global (cria se não existir)."""
    global _global_container
    if _global_container is None:
        _global_container = DIContainer()
    return _global_container


def reset_container() -> None:
    """Reseta o container global (útil para testes)."""
    global _global_container
    if _global_container is not None:
        _global_container.clear_processor()
    _global_container = None


def configure_default_processor(
    *,
    max_batch_mb: float | None = None,
    cpu_count: int | None = None,
    memory_mb: float | None = None,
    dask_scheduler: str = "threads",
    dask_processes: bool = False,
    diagnostics_enabled: bool = False,
    logger: Any | None = None,
) -> ProcessorPort:
    """Factory function para criar o processador padrão (ResourceAwareProcessor).

    Parameters
    ----------
    max_batch_mb : float | None
        Tamanho máximo de batch.
    cpu_count : int | None
        Número de CPUs.
    memory_mb : float | None
        Memória máxima.
    dask_scheduler : str
        Scheduler ("threads", "processes", "synchronous").
    dask_processes : bool
        Se usar processos.
    diagnostics_enabled : bool
        Se ativar diagnósticos.
    logger : Any | None
        Logger.

    Returns
    -------
    ProcessorPort
        Instância do processador.
    """
    container = get_container()
    return container.get_processor(
        max_batch_mb=max_batch_mb,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        dask_scheduler=dask_scheduler,
        dask_processes=dask_processes,
        diagnostics_enabled=diagnostics_enabled,
        logger=logger,
    )


__all__ = [
    "DIContainer",
    "configure_default_processor",
    "get_container",
    "reset_container",
]
