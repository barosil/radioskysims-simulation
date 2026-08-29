"""
domain/noise/value_objects.py

Value Objects para o domínio Noise.

Value Objects encapsulam conceitos do domínio (temperatura, eficiência, etc.)
com validação e imutabilidade. Dois value objects com mesmo valor são iguais.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import InvalidNoiseParameterError


class SystemTemperature:
    """
    Value Object representando temperatura de sistema (em Kelvin).

    Invariantes:
    - Sempre > 0
    - Imutável

    Representação:
    - Kelvin (K)
    - Intervalo típico: 20-300 K (depende da frequência e design)

    Attributes:
        value: Temperatura em Kelvin (float > 0)

    Examples:
        t_sys = SystemTemperature(50.0)  # 50 K
        assert t_sys.value == 50.0
    """

    __slots__ = ("_value",)

    def __init__(self, value: float) -> None:
        """
        Cria SystemTemperature.

        Args:
            value: Temperatura em Kelvin (deve ser > 0)

        Raises:
            InvalidNoiseParameterError: Se value <= 0 ou tipo inválido
        """
        self._validate(value)
        self._value = float(value)

    @staticmethod
    def _validate(value: Any) -> None:
        """Valida temperatura."""
        # Rejeitar strings explicitamente
        if isinstance(value, str):
            msg = f"t_sys deve ser float > 0, recebido {type(value).__name__}"
            raise InvalidNoiseParameterError(msg)

        try:
            v = float(value)
        except (TypeError, ValueError) as err:
            msg = f"t_sys deve ser float > 0, recebido {type(value).__name__}"
            raise InvalidNoiseParameterError(msg) from err

        if v <= 0:
            msg = f"t_sys deve ser > 0 K, recebido {v}"
            raise InvalidNoiseParameterError(msg)

    @property
    def value(self) -> float:
        """Valor da temperatura em Kelvin."""
        return self._value

    def __eq__(self, other: Any) -> bool:
        """Igualdade baseada em valor."""
        if not isinstance(other, SystemTemperature):
            return False
        return np.isclose(self._value, other._value)

    def __hash__(self) -> int:
        """Hash baseado em valor (com tolerância)."""
        return hash(round(self._value, 10))

    def __repr__(self) -> str:
        """Representação técnica."""
        return f"SystemTemperature({self._value:.2f} K)"

    def __str__(self) -> str:
        """Representação amigável."""
        return f"{self._value:.2f} K"


class Efficiency:
    """
    Value Object representando eficiência do sistema (0-1).

    Invariantes:
    - 0 < value <= 1
    - Imutável

    Representação:
    - Fração (0.0 a 1.0)
    - Exemplo: 0.95 = 95% de eficiência

    Attributes:
        value: Eficiência (float em (0, 1])

    Examples:
        eff = Efficiency(0.95)
        assert eff.value == 0.95
    """

    __slots__ = ("_value",)

    def __init__(self, value: float) -> None:
        """
        Cria Efficiency.

        Args:
            value: Eficiência (deve estar em (0, 1])

        Raises:
            InvalidNoiseParameterError: Se value <= 0, value > 1 ou tipo inválido
        """
        self._validate(value)
        self._value = float(value)

    @staticmethod
    def _validate(value: Any) -> None:
        """Valida eficiência."""
        try:
            v = float(value)
        except (TypeError, ValueError) as err:
            msg = (
                f"efficiency deve ser float em (0, 1], recebido {type(value).__name__}"
            )
            raise InvalidNoiseParameterError(msg) from err

        if v <= 0 or v > 1:
            msg = f"efficiency deve estar em (0, 1], recebido {v}"
            raise InvalidNoiseParameterError(msg)

    @property
    def value(self) -> float:
        """Valor da eficiência (0-1)."""
        return self._value

    def __eq__(self, other: Any) -> bool:
        """Igualdade baseada em valor."""
        if not isinstance(other, Efficiency):
            return False
        return np.isclose(self._value, other._value)

    def __hash__(self) -> int:
        """Hash baseado em valor (com tolerância)."""
        return hash(round(self._value, 10))

    def __repr__(self) -> str:
        """Representação técnica."""
        return f"Efficiency({self._value:.4f})"

    def __str__(self) -> str:
        """Representação amigável (percentual)."""
        return f"{self._value * 100:.1f}%"


class Bandwidth:
    """
    Value Object representando largura de banda (em Hz).

    Invariantes:
    - Sempre > 0
    - Imutável

    Representação:
    - Hertz (Hz)
    - Pode ser representado em MHz, GHz, etc. para legibilidade

    Attributes:
        value: Largura de banda em Hz (float > 0)

    Examples:
        bw = Bandwidth(1e6)  # 1 MHz
        assert bw.value == 1e6
    """

    __slots__ = ("_value",)

    def __init__(self, value: float) -> None:
        """
        Cria Bandwidth.

        Args:
            value: Largura de banda em Hz (deve ser > 0)

        Raises:
            InvalidNoiseParameterError: Se value <= 0 ou tipo inválido
        """
        self._validate(value)
        self._value = float(value)

    @staticmethod
    def _validate(value: Any) -> None:
        """Valida largura de banda."""
        try:
            v = float(value)
        except (TypeError, ValueError) as err:
            msg = f"bandwidth deve ser float > 0, recebido {type(value).__name__}"
            raise InvalidNoiseParameterError(msg) from err

        if v <= 0:
            msg = f"bandwidth deve ser > 0 Hz, recebido {v}"
            raise InvalidNoiseParameterError(msg)

    @property
    def value(self) -> float:
        """Valor da largura de banda em Hz."""
        return self._value

    def in_mhz(self) -> float:
        """Retorna largura de banda em MHz."""
        return self._value / 1e6

    def in_ghz(self) -> float:
        """Retorna largura de banda em GHz."""
        return self._value / 1e9

    def __eq__(self, other: Any) -> bool:
        """Igualdade baseada em valor."""
        if not isinstance(other, Bandwidth):
            return False
        return np.isclose(self._value, other._value)

    def __hash__(self) -> int:
        """Hash baseado em valor (com tolerância)."""
        return hash(round(self._value, 1))

    def __repr__(self) -> str:
        """Representação técnica."""
        return f"Bandwidth({self._value:.2e} Hz)"

    def __str__(self) -> str:
        """Representação amigável."""
        if self._value >= 1e9:
            return f"{self._value / 1e9:.2f} GHz"
        if self._value >= 1e6:
            return f"{self._value / 1e6:.2f} MHz"
        if self._value >= 1e3:
            return f"{self._value / 1e3:.2f} kHz"
        return f"{self._value:.2f} Hz"


class IntegrationTime:
    """
    Value Object representando tempo de integração (em segundos).

    Invariantes:
    - Sempre > 0
    - Imutável

    Representação:
    - Segundos (s)
    - Pode ser representado em ms, μs, etc. para legibilidade

    Attributes:
        value: Tempo de integração em segundos (float > 0)

    Examples:
        t_int = IntegrationTime(1.0)  # 1 segundo
        assert t_int.value == 1.0
    """

    __slots__ = ("_value",)

    def __init__(self, value: float) -> None:
        """
        Cria IntegrationTime.

        Args:
            value: Tempo de integração em segundos (deve ser > 0)

        Raises:
            InvalidNoiseParameterError: Se value <= 0 ou tipo inválido
        """
        self._validate(value)
        self._value = float(value)

    @staticmethod
    def _validate(value: Any) -> None:
        """Valida tempo de integração."""
        try:
            v = float(value)
        except (TypeError, ValueError) as err:
            msg = (
                f"integration_time deve ser float > 0, recebido {type(value).__name__}"
            )
            raise InvalidNoiseParameterError(msg) from err

        if v <= 0:
            msg = f"integration_time deve ser > 0 s, recebido {v}"
            raise InvalidNoiseParameterError(msg)

    @property
    def value(self) -> float:
        """Valor do tempo de integração em segundos."""
        return self._value

    def in_milliseconds(self) -> float:
        """Retorna tempo em milissegundos."""
        return self._value * 1e3

    def in_microseconds(self) -> float:
        """Retorna tempo em microssegundos."""
        return self._value * 1e6

    def __eq__(self, other: Any) -> bool:
        """Igualdade baseada em valor."""
        if not isinstance(other, IntegrationTime):
            return False
        return np.isclose(self._value, other._value)

    def __hash__(self) -> int:
        """Hash baseado em valor (com tolerância)."""
        return hash(round(self._value, 10))

    def __repr__(self) -> str:
        """Representação técnica."""
        return f"IntegrationTime({self._value:.4f} s)"

    def __str__(self) -> str:
        """Representação amigável."""
        if self._value >= 1:
            return f"{self._value:.2f} s"
        if self._value >= 1e-3:
            return f"{self._value * 1e3:.2f} ms"
        if self._value >= 1e-6:
            return f"{self._value * 1e6:.2f} μs"
        return f"{self._value * 1e9:.2f} ns"
