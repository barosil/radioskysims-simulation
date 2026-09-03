"""
domain/rfi/value_objects.py

Value Objects para o domínio RFI.

Value Objects são imutáveis e comparáveis apenas por valor (não por identidade).
Representam conceitos de negócio como FrequencyRange.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import InvalidFrequencyRangeError


class FrequencyRange:
    """
    Value Object que representa um intervalo de frequência em GHz.

    Características:
    - Imutável (frozenset-like)
    - Comparável apenas por valor
    - Validações de domínio no construtor
    - Restrição: f_min < f_max, ambas > 0

    Exemplo:
        fr = FrequencyRange(1.4, 1.8)
        assert 1.5 in fr
        assert 2.0 not in fr
    """

    __slots__ = ("_f_max", "_f_min")

    def __init__(self, f_min: float, f_max: float) -> None:
        """
        Cria um FrequencyRange.

        Args:
            f_min: Frequência mínima em GHz
            f_max: Frequência máxima em GHz

        Raises:
            InvalidFrequencyRangeError: Se f_min >= f_max ou valores não-positivos
        """
        self._validate(f_min, f_max)
        self._f_min = float(f_min)
        self._f_max = float(f_max)

    @staticmethod
    def _validate(f_min: Any, f_max: Any) -> None:
        """Valida os limites de frequência."""
        # Tipo
        try:
            f_min_float = float(f_min)
            f_max_float = float(f_max)
        except (TypeError, ValueError) as e:
            msg = f"Frequências devem ser numéricos: {e}"
            raise InvalidFrequencyRangeError(msg) from e

        # Positivas
        if f_min_float <= 0 or f_max_float <= 0:
            msg = f"Frequências devem ser > 0 GHz: f_min={f_min_float}, f_max={f_max_float}"
            raise InvalidFrequencyRangeError(msg)

        # Ordem
        if f_min_float >= f_max_float:
            msg = f"f_min deve ser < f_max: {f_min_float} >= {f_max_float}"
            raise InvalidFrequencyRangeError(msg)

    @property
    def f_min(self) -> float:
        """Frequência mínima em GHz."""
        return self._f_min

    @property
    def f_max(self) -> float:
        """Frequência máxima em GHz."""
        return self._f_max

    @property
    def bandwidth(self) -> float:
        """Largura de banda em GHz."""
        return self._f_max - self._f_min

    def __contains__(self, frequency: float) -> bool:
        """
        Verifica se uma frequência está dentro deste intervalo.

        Args:
            frequency: Frequência em GHz

        Returns:
            True se f_min <= frequency <= f_max
        """
        return self._f_min <= float(frequency) <= self._f_max

    def overlaps_with(self, other: FrequencyRange) -> bool:
        """
        Verifica se este intervalo sobrepõe com outro.

        Args:
            other: Outro FrequencyRange

        Returns:
            True se há sobreposição, False caso contrário
        """
        return not (self._f_max < other._f_min or other._f_max < self._f_min)

    def intersection(self, other: FrequencyRange) -> FrequencyRange | None:
        """
        Calcula interseção com outro intervalo.

        Args:
            other: Outro FrequencyRange

        Returns:
            FrequencyRange da interseção, ou None se não há sobreposição
        """
        if not self.overlaps_with(other):
            return None

        f_min = max(self._f_min, other._f_min)
        f_max = min(self._f_max, other._f_max)

        return FrequencyRange(f_min, f_max)

    def union(self, other: FrequencyRange) -> FrequencyRange | tuple[FrequencyRange, ...]:
        """
        Calcula união com outro intervalo.

        Args:
            other: Outro FrequencyRange

        Returns:
            Um novo FrequencyRange se contíguo, ou tupla se disjuntos
        """
        if self.overlaps_with(other):
            f_min = min(self._f_min, other._f_min)
            f_max = max(self._f_max, other._f_max)
            return FrequencyRange(f_min, f_max)

        # Retorna ambos em ordem
        if self._f_max < other._f_min:
            return (self, other)
        return (other, self)

    def __eq__(self, other: object) -> bool:
        """Igualdade por valor."""
        if not isinstance(other, FrequencyRange):
            return NotImplemented
        return np.isclose(self._f_min, other._f_min) and np.isclose(self._f_max, other._f_max)

    def __hash__(self) -> int:
        """Hash para usar em conjuntos/dicionários."""
        return hash((round(self._f_min, 10), round(self._f_max, 10)))

    def __repr__(self) -> str:
        return f"FrequencyRange({self._f_min:.3f}-{self._f_max:.3f} GHz)"

    def __str__(self) -> str:
        return f"{self._f_min:.3f}-{self._f_max:.3f} GHz"
