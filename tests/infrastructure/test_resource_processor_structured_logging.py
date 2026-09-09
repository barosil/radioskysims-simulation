"""\
Módulo: tests/infrastructure/test_resource_processor_structured_logging.py

Regressão do logger no ``ResourceAwareProcessor`` (rota dask distributed).

O processador usava ``logging.getLogger`` (stdlib) com chamadas no estilo
estruturado (``.info("event", chave=valor)``), o que levantava
``TypeError: Logger._log() got an unexpected keyword argument`` ao subir o
scheduler distributed (dashboard). Após a troca para
``radioskysims_core.logging_config.get_logger`` (``StructuredLogger``),
a rota distributed não deve mais falhar.

Copyright BINGO Collaboration
Last modified: 2026-09-08
"""

from __future__ import annotations

import pytest

from radioskysims_simulation.infrastructure.processing.resource_processor import (
    ProcessingResources,
    ResourceAwareProcessor,
)


@pytest.mark.parametrize("processes", [True, False])
def test_distributed_start_logs_and_sets_dashboard(processes: bool) -> None:
    """Sobe o scheduler distributed (LocalCluster + dashboard) sem TypeError."""
    pytest.importorskip("distributed")
    proc = ResourceAwareProcessor(
        resources=ProcessingResources(cpu_count=2, available_memory_mb=512.0),
        dask_scheduler="distributed",
        dask_processes=processes,
    )
    try:
        # Antes do fix: TypeError — chamada estilo estruturado em logger stdlib.
        proc._configure_dask()
        assert proc.dashboard_url is not None
    finally:
        proc.shutdown()
