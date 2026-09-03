"""
domain/noise/exceptions.py

Exceções de domínio para Noise (ruído do sistema de radiotelescópio).

As exceções abaixo representam violações de regras de negócio e invariantes
do domínio Noise. Cada exceção corresponde a uma situação que não deveria
ocorrer em operação normal.
"""

from __future__ import annotations


class NoiseDomainError(Exception):
    """
    Exceção base para todas as exceções de domínio Noise.

    Herança:
        Exception
    """


class InvalidNoiseParameterError(NoiseDomainError):
    """
    Erro: Parâmetro de ruído violou uma restrição do domínio.

    Violations:
    - t_sys <= 0 (temperatura de sistema deve ser positiva)
    - efficiency <= 0 ou efficiency > 1 (eficiência deve estar em (0, 1])
    - bandwidth <= 0 (largura de banda deve ser positiva)
    - integration_time <= 0 (tempo de integração deve ser positivo)
    - Type mismatch (esperado float, recebido outro tipo)

    Examples:
        InvalidNoiseParameterError("t_sys deve ser > 0, recebido -10.5")
        InvalidNoiseParameterError("efficiency deve estar em (0, 1], recebido 1.5")
    """


class NoiseGenerationError(NoiseDomainError):
    """
    Erro: Falha ao gerar série temporal de ruído.

    Violations:
    - Array de frequências vazio
    - Array de tempos vazio
    - Array de elevação com shape incompatível
    - Parâmetros inconsistentes (ex: frequencies.shape != elevations.shape)
    - Operação matemática falhou (resultado contém NaN ou inf)

    Examples:
        NoiseGenerationError("frequencies não pode estar vazio")
        NoiseGenerationError("Shape incompatível: frequencies (100,) vs elevations (50,)")
    """
