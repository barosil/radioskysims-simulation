"""\
Módulo: application/processing_resources.py

Descrição:
    Value objects e utilitários de estimativa de recursos de processamento,
    movidos para a camada de Application para quebrar o ciclo de import
    ``application → infrastructure``.

    Aqui vivem:
    - ``ProcessingResources`` (detecção de CPU/memória e margens de segurança);
    - ``MemoryEstimator`` (estimativa de footprint de memória por fonte/batch).

    A implementação concreta do processador (``ResourceAwareProcessor``)
    permanece em ``infrastructure.processing.resource_processor`` e importa
    este módulo — a direção correta do hexagonal (infra depende de application,
    nunca o contrário).

Copyright BINGO Collaboration
Last modified: 2026-11-05
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from radioskysims_core.domain.common import FrequencyRange, TimeRange
from radioskysims_core.logging_config import get_logger
from radioskysims_sources.domain.signal_source import SignalSource

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constantes de estimativa de footprint de memória
# ---------------------------------------------------------------------------

# Margem de segurança multiplicativa sobre a estimativa teórica
_SAFETY_FACTOR = 3.0

# Memória reservada para o sistema/sobrecarga Python (MB)
_SYSTEM_RESERVE_MB = 512.0

# Bytes por elemento para float64
_F64_BYTES = 8

# Fatores de footprint por fonte, em unidades de "tensor principal"
# (time × frequency × source=1 × polarization × pointing=1):
#   - 2 tensores de saída acumulados por fonte (T_antenna + power);
#   - ~6 tensores intermediários no pico de um único ``TODSimulator.simulate()``
#     (beam_response, beam_power, flux_at_observer, equivalent_temperature,
#     antenna_temperature, power) — tratado como constante por batch
#     (conservador).
_OUTPUT_TENSORS = 2
_PEAK_TENSORS = 6

# Nº de polarizações assumido na estimativa. O feixe do horn do BINGO tem 4
# polarizações (H/V COPOL/CROSS) — assumir 2 subestimava a saída em 2× e impedia
# a decisão de streaming a disco (dataset tentava caber inteiro em RAM).
_DEFAULT_POLARIZATIONS = 4
"""Polarizações por padrão nas estimativas de memória.

O feixe do horn traz H/V COPOL/CROSS (4 correlações), não 2 — subestimar
por 2× fazia a saída "caber" na memória e desligava o streaming a disco.
"""

_DEFAULT_PARALLEL_WORKERS: Final[int] = 8
"""Nº alvo de chunks executados SIMULTANEAMENTE (padrão).

Ajustável pelo parâmetro ``parallel_workers`` ou pela variável de ambiente
``RSS_SIM_PARALLEL_WORKERS``; o valor efetivo é sempre limitado pela memória
orçamentada (ver :meth:`ResourceAwareProcessor.batch_budget_mb`).
"""


def _n_time_steps(time_range: TimeRange) -> int:
    """Nº real de passos temporais do TimeRange (inclusive extremos).

    Usa a duração efetiva de ``time_range`` (e não 24h fixo, como a versão
    antiga). Alinha com ``TimeRange.to_datetime64()``:
    ``round(duration_s / step_s) + 1``.
    """
    try:
        start, end, step_s = time_range.to_tuple()
    except Exception:
        return 1
    if step_s <= 0:
        return 1
    duration_s = float((end - start).sec)
    if duration_s <= 0:
        duration_s = 24 * 3600.0  # fallback defensivo
    return max(1, int(round(duration_s / step_s)) + 1)


def _n_freq_steps(
    frequency_range: FrequencyRange | None,
    num_channels: int | None,
) -> int:
    """Nº de canais de frequência (num_channels ou derivado do range)."""
    n_freq = num_channels or 1
    if frequency_range is not None and n_freq == 1:
        # Deriva o nº de canais da grade real do range (ex.: 1 bin com
        # min == max → value.size == 1). Evita chamar `to_tuple()`, que
        # não existe em FrequencyRange (bug quando num_channels == 1).
        n_freq = max(1, int(frequency_range.value.size))
    return n_freq


def _tensor_mb(
    n_time: int,
    n_freq: int,
    n_sources: int = 1,
    n_pol: int = _DEFAULT_POLARIZATIONS,
) -> float:
    """MB de UM tensor float64 (time, freq, source, polarization, pointing)."""
    shape = (n_time, n_freq, n_sources, n_pol, 1)
    bytes_ = int(np.prod(shape, dtype=np.int64)) * _F64_BYTES
    return float(bytes_) / (1024 * 1024)


@dataclass(frozen=True)
class ProcessingResources:
    """Recursos disponíveis detectados pelo sistema."""

    cpu_count: int
    available_memory_mb: float
    memory_safety_margin: float = _SAFETY_FACTOR
    system_reserve_mb: float = _SYSTEM_RESERVE_MB

    @property
    def usable_memory_mb(self) -> float:
        """Memória utilizável após reserva de sistema e margem de segurança."""
        return max(
            0.0,
            (self.available_memory_mb - self.system_reserve_mb) / self.memory_safety_margin,
        )

    @property
    def usable_cpu_count(self) -> int:
        """CPUs utilizáveis (pelo menos 1)."""
        return max(1, self.cpu_count)

    @classmethod
    def detect(
        cls,
        *,
        cpu_count: int | None = None,
        memory_mb: float | None = None,
        safety_margin: float | None = None,
    ) -> ProcessingResources:
        """Detecta recursos do sistema com fallbacks e overrides.

        Parameters
        ----------
        cpu_count:
            Sobrescreve a detecção automática de CPUs.
        memory_mb:
            Sobrescreve a detecção automática de memória (MB).
        safety_margin:
            Sobrescreve o fator de segurança padrão.

        Returns
        -------
        ProcessingResources
        """
        n_cpus = cpu_count
        if n_cpus is None:
            try:
                n_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                n_cpus = os.cpu_count() or 4

        mem_mb = memory_mb
        if mem_mb is None:
            try:
                import psutil

                mem = psutil.virtual_memory()
                mem_mb = float(mem.available) / (1024 * 1024)
            except ImportError:
                # Fallback: assume 4GB disponível
                mem_mb = 4096.0
                log.warning(
                    "psutil não disponível. Assumindo %.0f MB de memória "
                    "disponível. Instale psutil para detecção automática.",
                    mem_mb,
                )

        return cls(
            cpu_count=n_cpus,
            available_memory_mb=mem_mb,
            memory_safety_margin=safety_margin or _SAFETY_FACTOR,
        )


class MemoryEstimator:
    """Estima o consumo de memória de uma simulação baseada nas fontes."""

    @staticmethod
    def _estimate_n_time(
        time_range: TimeRange,
    ) -> int:
        """Retorna o número real de amostras temporais do TimeRange."""
        _, _, step_s = time_range.to_tuple()

        if step_s <= 0:
            return 1

        duration_s = (time_range.end_jd - time_range.start_jd) * 86400.0

        return max(
            1,
            int(round(duration_s / step_s)) + 1,
        )

    @staticmethod
    def estimate_source_mb(
        sources: Sequence[SignalSource],
        time_range: TimeRange,
        frequency_range: FrequencyRange | None,
        num_channels: int | None,
    ) -> float:
        """Estima a memória de pico para processar N fontes em um batch.

        O cálculo considera:
        - Nº real de passos temporais (duração efetiva do ``time_range``);
        - Nº de canais de frequência;
        - Saída acumulada por fonte (T_antenna + power);
        - Pico de arrays intermediários de um ``simulate()`` (≈6 tensores).

        Parameters
        ----------
        sources:
            Lista de fontes a processar.
        time_range:
            Intervalo temporal da simulação.
        frequency_range:
            Range de frequências.
        num_channels:
            Número de canais de frequência.

        Returns
        -------
        float
            Estimativa em MB.
        """
        n_time = _n_time_steps(time_range)
        n_freq = _n_freq_steps(frequency_range, num_channels)
        tensor_mb = _tensor_mb(n_time, n_freq, n_sources=1)

        n_sources = len(sources)

        # Saída (T+power) acumulada por fonte + pico intermediário (constante).
        total_mb = tensor_mb * (_OUTPUT_TENSORS * n_sources + _PEAK_TENSORS)
        return total_mb

    @staticmethod
    def estimate_single_source_mb(
        time_range: TimeRange,
        frequency_range: FrequencyRange | None = None,
        num_channels: int | None = None,
    ) -> float:
        """Estima a memória para uma única fonte (saída + pico de simulação)."""
        n_time = _n_time_steps(time_range)
        n_freq = _n_freq_steps(frequency_range, num_channels)
        tensor_mb = _tensor_mb(n_time, n_freq, n_sources=1)
        return tensor_mb * (_OUTPUT_TENSORS + _PEAK_TENSORS)

    @staticmethod
    def estimate_total_output_mb(
        sources: Sequence[SignalSource],
        time_range: TimeRange,
        frequency_range: FrequencyRange | None = None,
        num_channels: int | None = None,
    ) -> float:
        """Só a saída final (T_antenna + power) para TODAS as fontes.

        Útil para decidir se o dataset consolidado cabe em RAM ou precisa ser
        gravado/reamostrado em disco (streaming por batches).
        """
        n_time = _n_time_steps(time_range)
        n_freq = _n_freq_steps(frequency_range, num_channels)
        tensor_mb = _tensor_mb(n_time, n_freq, n_sources=1)
        return tensor_mb * _OUTPUT_TENSORS * max(1, len(sources))
