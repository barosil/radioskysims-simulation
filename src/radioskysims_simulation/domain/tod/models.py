"""
Módulo: src/radioskysims/domain/tod/models.py

Descrição:
    Modelo de dados TOD (Time-Ordered Data) no domínio tempo x frequência.
    Representa dados medidos ou simulados organizados temporalmente.
"""

# domain/tod/models.py
from __future__ import annotations

import xarray as xr


class TOD:
    """
    Time-Ordered Data (TOD) no domínio tempo x frequência.

    Representa medidas ou simulações de sinais ordenados no tempo,
    tipicamente em múltiplas frequências.

    Attributes:
        data (xr.DataArray): Array de dados com dimensões time e freq.
    """

    def __init__(self, data: xr.DataArray) -> None:
        """
        Inicializa TOD com dados temporais.

        Args:
            data (xr.DataArray): DataArray com dimensões (time, freq).
        """
        self.data: xr.DataArray = data
