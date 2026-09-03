"""
domain/rfi/exceptions.py

Exceções de domínio para o contexto RFI (Radio Frequency Interference).

Estas exceções representam violações de regras de negócio do domínio,
não exceções técnicas de infraestrutura.
"""

from __future__ import annotations


class RFIDomainError(Exception):
    """Exceção base para erros de domínio RFI."""


class InvalidFrequencyRangeError(RFIDomainError):
    """
    Exceção levantada quando um intervalo de frequência viola regras de domínio.

    Violações:
    - f_min >= f_max
    - Frequências negativas ou zero
    - Tipos inválidos
    """


class InvalidRFIError(RFIDomainError):
    """
    Exceção levantada quando RFI viola regras de domínio.

    Violações:
    - Sem frequency_ranges
    - Frequências não em GHz
    """


class RFIApplicationError(RFIDomainError):
    """
    Exceção levantada quando aplicação de filtro RFI falha.

    Causas:
    - TOD malformado (shape incorreto)
    - Frequências inválidas
    - Dataset corrompido
    """


class RFICombinationError(RFIDomainError):
    """Exceção levantada quando combinação de múltiplas instâncias RFI falha."""
