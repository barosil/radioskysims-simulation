"""
domain/rfi/detector.py

RFIDetector - Serviço de domínio para detectar interferência de radiofrequência em TOD.

O RFIDetector implementa um algoritmo robusto de detecção estatística de RFI
baseado em desvios do espectro de referência.

Algoritmo:
----------
1. Calcula espectro de referência como mediana ao longo do tempo
2. Calcula desvio normalizado: deviation = (power - baseline) / baseline.std()
3. Marca pixels onde deviation > threshold_sigma
4. Converte cada detecção em RFIEvent com informações completas

Referências:
-----------
- Offringa et al. (2010): "Post-Correlation Radio Frequency Interference"
  Detection Methods
- LOFAR RFI detection strategies
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr


@dataclass(frozen=True)
class RFIEvent:
    """
    Representa uma detecção de RFI.

    Attributes:
        time_idx: Índice temporal do pixel detectado
        freq_idx: Índice de frequência do pixel detectado
        power: Potência medida (W/Hz ou unidade original)
        severity: Severidade normalizada (sigma deviations)
        timestamp: Timestamp do evento (opcional)
        frequency: Frequência central da banda (Hz)
    """

    time_idx: int
    freq_idx: int
    power: float
    severity: float
    timestamp: datetime | None = None
    frequency: float | None = None


class RFIDetector:
    """
    Detecta interferência de radiofrequência (RFI) em Time-Ordered Data.

    Implementa detecção estatística baseada em desvios do espectro de referência,
    adequado para radiossondagem e radioastronomia.

    Attributes:
        threshold_sigma: Número de desvios-padrão para marcar como RFI (default: 5.0)

    Example:
        >>> detector = RFIDetector(threshold_sigma=3.0)
        >>> events = detector.detect(tod)
        >>> for event in events:
        ...     print(f"RFI em t={event.time_idx}, f={event.freq_idx}: "
        ...           f"severity={event.severity:.2f}sigma")
    """

    def __init__(self, threshold_sigma: float = 5.0) -> None:
        """
        Inicializa o detector.

        Args:
            threshold_sigma: Limiar em desvios-padrão para detecção de RFI.
                Default é 5.0 (muito conservador).
                Valores comuns: 3.0 (sensível), 5.0 (robusto), 10.0 (muito robusto)

        Raises:
            ValueError: Se threshold_sigma <= 0
        """
        if threshold_sigma <= 0:
            msg = f"threshold_sigma deve ser > 0, recebido {threshold_sigma}"
            raise ValueError(msg)

        self.threshold_sigma = float(threshold_sigma)

    def detect(
        self,
        tod: xr.Dataset,
        reference_spectrum: xr.DataArray | None = None,
    ) -> list[RFIEvent]:
        """
        Detecta RFI em Time-Ordered Data.

        Processa:
        1. Calcula espectro de referência (mediana temporal se não fornecido)
        2. Calcula desvio normalizado para cada pixel
        3. Marca pixels onde deviation > threshold_sigma
        4. Retorna lista de RFIEvent com metadados completos

        Args:
            tod: xarray.Dataset com coordenadas 'time', 'frequency' e variável 'power'
                Esperado shape: (n_time, n_frequency)
            reference_spectrum: xarray.DataArray com espectro de referência (opcional).
                Se não fornecido, calcula mediana ao longo do tempo.
                Shape: (n_frequency,)

        Returns:
            list[RFIEvent]: Lista de detecções ordenadas por severidade (descendente)

        Raises:
            KeyError: Se 'power' não estiver em tod
            ValueError: Se dimensões inconsistentes

        Example:
            >>> detector = RFIDetector(threshold_sigma=3.0)
            >>> tod = xr.Dataset({
            ...     'power': (('time', 'frequency'), power_array),
            ... }, coords={'time': time, 'frequency': freq})
            >>> events = detector.detect(tod)
        """
        # Extrair dados de potência
        if "power" not in tod:
            msg = "Dataset deve conter variável 'power'"
            raise KeyError(msg)

        power = tod["power"].to_numpy()  # shape: (n_time, n_frequency)

        if power.ndim != 2:
            msg = f"power deve ser 2D (time, frequency), recebido shape {power.shape}"
            raise ValueError(msg)

        n_time, n_freq = power.shape

        # Lidar com dataset vazio
        if n_time == 0 or n_freq == 0:
            return []

        # Obter coordenadas
        time_coords = tod.coords.get("time", np.arange(n_time))
        freq_coords = tod.coords.get("frequency", np.arange(n_freq))

        # Calcular espectro de referência (mediana no tempo)
        if reference_spectrum is None:
            baseline = np.median(power, axis=0)  # shape: (n_frequency,)
        else:
            baseline = (
                reference_spectrum.to_numpy()
                if hasattr(reference_spectrum, "to_numpy")
                else np.asarray(reference_spectrum)
            )

        # Calcular desvio-padrão da baseline
        baseline_std = np.std(baseline)

        if baseline_std == 0:
            # Dados uniformes - retorna lista vazia
            return []

        # Calcular desvio normalizado: deviation = (power - baseline) / baseline.std()
        deviation = (power - baseline[np.newaxis, :]) / baseline_std  # shape: (n_time, n_frequency)

        # Encontrar pixels acima do limiar
        mask = deviation > self.threshold_sigma  # shape: (n_time, n_frequency)

        # Converter detecções em RFIEvent
        events = []
        time_indices, freq_indices = np.where(mask)

        for t_idx, f_idx in zip(time_indices, freq_indices, strict=False):
            event = RFIEvent(
                time_idx=int(t_idx),
                freq_idx=int(f_idx),
                power=float(power[t_idx, f_idx]),
                severity=float(deviation[t_idx, f_idx]),
                timestamp=self._get_timestamp(time_coords, t_idx),
                frequency=float(freq_coords[f_idx])
                if hasattr(freq_coords, "__getitem__")
                else None,
            )
            events.append(event)

        # Ordenar por severidade (descendente)
        events.sort(key=lambda e: e.severity, reverse=True)

        return events

    @staticmethod
    def _get_timestamp(time_coords: np.ndarray | list, idx: int) -> datetime | None:
        """
        Tenta extrair timestamp dos coordenadas de tempo.

        Se time_coords contém objetos datetime, retorna o timestamp.
        Caso contrário, retorna None.

        Args:
            time_coords: Coordenadas de tempo do dataset
            idx: Índice do tempo

        Returns:
            datetime or None
        """
        try:
            # Tentar converter para datetime se possível
            time_val = time_coords[idx]
            if isinstance(time_val, datetime | np.datetime64):
                return pd.Timestamp(time_val).to_pydatetime()
        except (ImportError, IndexError, TypeError, AttributeError):
            pass

        return None
