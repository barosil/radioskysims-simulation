"""
domain/rfi/services.py

Serviços de domínio para RFI (Radio Frequency Interference).

Serviços de domínio encapsulam lógica que envolve múltiplas entidades ou
value objects, mas não pertencem a uma entidade específica.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

from .exceptions import RFIApplicationError, RFICombinationError
from .models import RFI


class RFIService:
    """
    Serviço de domínio para operações com RFI.

    Responsabilidades:
    - Combinar múltiplas instâncias de RFI
    - Aplicar RFI compostos em TOD
    - Validações e garantias de regras físicas

    Características:
    - Stateless (sem estado)
    - Métodos estáticos para operações puras
    - Sem efeitos colaterais
    - Sem dependências de infraestrutura
    """

    @staticmethod
    def combine_rfis(rfi_list: list[RFI]) -> RFI:
        """
        Combina múltiplas instâncias de RFI em uma única entidade RFI.

        Estratégia:
        - Coleta todos os FrequencyRange
        - Remove duplicatas
        - Mescla sobrepostos
        - Retorna novo RFI com intervalos combinados

        Args:
            rfi_list: Lista de instâncias RFI

        Returns:
            Nova instância RFI com todos os intervalos combinados

        Raises:
            RFICombinationError: Se lista vazia ou inválida

        Exemplo:
            rfi1 = RFI([FrequencyRange(1.2, 1.4)], name="GSM")
            rfi2 = RFI([FrequencyRange(1.6, 1.8)], name="WiFi")
            combined = RFIService.combine_rfis([rfi1, rfi2])
        """
        if not rfi_list:
            msg = "rfi_list não pode estar vazio"
            raise RFICombinationError(msg)

        if not all(isinstance(rfi, RFI) for rfi in rfi_list):
            msg = "Todos os items devem ser instâncias de RFI"
            raise RFICombinationError(msg)

        # Coleta todos os frequency_ranges
        all_ranges = []
        for rfi in rfi_list:
            all_ranges.extend(rfi.frequency_ranges)

        if not all_ranges:
            msg = "Nenhum frequency_range encontrado em rfi_list"
            raise RFICombinationError(msg)

        # Cria novo RFI com todos os ranges
        # A classe RFI cuida da deduplicação e mesclagem
        return RFI(
            frequency_ranges=all_ranges,
            name="Combined RFI",
            description=f"Combinação de {len(rfi_list)} fontes de RFI",
        )

    @staticmethod
    def apply_multiple_rfis(
        tod: Any,
        rfi_list: list[RFI],
        frequencies: np.ndarray | None = None,
    ) -> Any:
        """
        Aplica múltiplas instâncias de RFI sequencialmente em TOD.

        Args:
            tod: xarray.Dataset
            rfi_list: Lista de instâncias RFI
            frequencies: Array de frequências em GHz (opcional)

        Returns:
            TOD filtrado com todas as RFIs aplicadas

        Raises:
            RFIApplicationError: Se aplicação falha
        """
        if not rfi_list:
            return tod

        # Combina todas em uma
        combined = RFIService.combine_rfis(rfi_list)

        # Aplica a combinada
        return combined.apply_rfi_filter(tod, frequencies=frequencies)

    @staticmethod
    def validate_tod_filtered(
        original_tod: Any,
        filtered_tod: Any,
        _rfi: RFI,
    ) -> dict[str, Any]:
        """
        Valida que o filtro RFI foi aplicado corretamente.

        Checklist:
        - Shape preservado
        - Sem NaNs introduzidos
        - Dimensões intactas
        - Coordenadas preservadas

        Args:
            original_tod: TOD antes do filtro
            filtered_tod: TOD após o filtro
            rfi: Instância RFI aplicada

        Returns:
            Dicionário com resultados da validação

        Raises:
            RFIApplicationError: Se validação falha
        """
        if xr is None:
            msg = "xarray não disponível para validação"
            raise RFIApplicationError(msg)

        validation_results = {
            "shape_preserved": False,
            "no_new_nans": False,
            "dims_preserved": False,
            "coords_preserved": False,
            "all_valid": False,
        }

        try:
            # Shape preservado
            if original_tod.dims == filtered_tod.dims:
                validation_results["shape_preserved"] = True
            else:
                msg = f"Shape mudou: {original_tod.dims} → {filtered_tod.dims}"
                raise RFIApplicationError(msg)

            # Dimensões preservadas
            if set(original_tod.data_vars) == set(filtered_tod.data_vars):
                validation_results["dims_preserved"] = True
            else:
                msg = "Variáveis de dados mudaram"
                raise RFIApplicationError(msg)

            # Coordenadas preservadas
            if set(original_tod.coords) == set(filtered_tod.coords):
                validation_results["coords_preserved"] = True
            else:
                msg = "Coordenadas mudaram"
                raise RFIApplicationError(msg)

            # Sem NaNs novos
            original_nans = sum(
                original_tod[var].isnull().sum().item() for var in original_tod.data_vars
            )
            filtered_nans = sum(
                filtered_tod[var].isnull().sum().item() for var in filtered_tod.data_vars
            )

            if filtered_nans <= original_nans:
                validation_results["no_new_nans"] = True
            else:
                new_nans = filtered_nans - original_nans
                msg = f"Novos NaNs introduzidos: {new_nans}"
                raise RFIApplicationError(msg)

            # Todos os testes passaram
            if all(validation_results.values()):
                validation_results["all_valid"] = True

        except RFIApplicationError:
            raise
        except Exception as e:
            msg = f"Erro durante validação: {e}"
            raise RFIApplicationError(msg) from e

        return validation_results

    @staticmethod
    def estimate_contamination_ratio(
        tod: Any,
        frequencies: np.ndarray | None = None,
        rfi: RFI | None = None,
    ) -> float:
        """
        Estima razão de contaminar em TOD (0.0 a 1.0).

        Args:
            tod: xarray.Dataset
            frequencies: Array de frequências (opcional)
            rfi: Instância RFI. Se None, assume que dados já estão filtrados

        Returns:
            Razão de contaminar (0.0 = sem RFI, 1.0 = totalmente contaminado)
        """
        if xr is None:
            return 0.0

        if not isinstance(tod, xr.Dataset):
            return 0.0

        try:
            # Se RFI fornecida, calcula razão de frequências contaminadas
            if rfi is not None:
                if frequencies is None:
                    if "frequency" not in tod.coords:
                        return 0.0
                    frequencies = tod.coords["frequency"].values

                mask = rfi.create_mask(frequencies)
                return float(np.mean(mask))

            # Caso contrário, retorna 0 (assume dados já filtrados)
            return 0.0

        except Exception:
            return 0.0
