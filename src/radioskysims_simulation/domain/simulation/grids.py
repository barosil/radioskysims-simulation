"""
Módulo: src/radioskysims/domain/simulation/grids.py

Helpers puros de grade para simulação multi-fonte (tempo, frequência e FWHM).

Extraídos de ``application/services/simulation_service.py`` (E1/E5 — remoção
da fronteira domain→application e decomposição do hotspot): são funções puras
de domínio usadas pelo SimulationService, SimOrchestrator e pelo próprio
domínio (normalização do eixo temporal). A camada de application as reexporta
por compatibilidade.

Copyright BINGO Collaboration
Last modified: 2026-09-03
"""

from __future__ import annotations

from typing import Any

import numpy as np
from radioskysims_core.domain.common import FrequencyRange, TimeRange


def _compute_time_grid(time_range: TimeRange) -> np.ndarray:
    """Gera grade temporal datetime64[ns] a partir do TimeRange.

    Evita ruído de ponto flutuante na conversão JD → datetime64, usando um
    passo fixo em nanossegundos a partir do início do intervalo.
    """
    tr_start, _, step_s = time_range.to_tuple()
    start_dt = tr_start.utc.datetime64.astype("datetime64[ns]")
    if step_s <= 0:
        return np.array([start_dt], dtype="datetime64[ns]")

    n_steps = int(np.round((time_range.end_jd - time_range.start_jd) / (step_s / 86400.0))) + 1
    step_ns = np.timedelta64(int(round(step_s * 1e9)), "ns")
    return start_dt + np.arange(n_steps, dtype=np.int64) * step_ns


def _compute_freq_mhz(frequency_range: FrequencyRange | None) -> np.ndarray:
    """Gera grade de frequências em MHz."""
    if frequency_range is None:
        return np.array([], dtype=float)
    return np.asarray(frequency_range.value, dtype=float)


def _beam_fwhm_deg(beam: Any | None) -> float | None:
    """Extrai a largura à meia-potência (FWHM) do feixe em graus.

    Usa ``np.nanmean`` porque ``beam.params().fwhm`` pode conter NaN
    (ex.: beam CSV do BINGO). Retorna ``None`` se o feixe não expor
    ``params().fwhm`` ou se o valor não for finito/positivo.
    """
    if beam is None:
        return None
    try:
        fwhm_vals = beam.params().fwhm
        if hasattr(fwhm_vals, "mean"):
            fwhm_mean = float(np.nanmean(fwhm_vals))
        else:
            fwhm_mean = float(fwhm_vals)
        if np.isfinite(fwhm_mean) and fwhm_mean > 0:
            return fwhm_mean
        return None
    except Exception:
        return None
