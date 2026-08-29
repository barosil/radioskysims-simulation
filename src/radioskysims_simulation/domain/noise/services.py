"""
domain/noise/services.py

Serviço de domínio para operações com Noise.

Serviços de domínio encapsulam lógica que envolve múltiplas entidades
ou value objects, mas não pertencem a uma entidade específica.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import InvalidNoiseParameterError, NoiseGenerationError
from .models import Noise
from .value_objects import Bandwidth, IntegrationTime


class NoiseService:
    """
    Serviço de domínio para operações com múltiplas instâncias de Noise.

    Responsabilidades:
    - Compor ruído de múltiplas fontes
    - Aplicar eficiência do sistema
    - Garantir coerência física

    Características:
    - Stateless (sem estado)
    - Métodos estáticos para operações puras
    - Sem efeitos colaterais
    - Sem dependências de infraestrutura
    """

    @staticmethod
    def compose_noise(
        thermal: np.ndarray,
        atmospheric: np.ndarray,
        add_quadrature: bool = True,
    ) -> np.ndarray:
        """
        Compõe ruído térmico com ruído atmosférico.

        Estratégia:
        - Se add_quadrature=True: composição em quadratura (raiz da soma de quadrados)
        - Se add_quadrature=False: simples adição linear

        Args:
            thermal: Array de ruído térmico em Kelvin
            atmospheric: Array de ruído atmosférico em Kelvin
            add_quadrature: Se True, usa soma quadrática; se False, adição linear

        Returns:
            Array de ruído total em Kelvin (shape = thermal.shape)

        Raises:
            NoiseGenerationError: Se operação falha

        Notes:
            - add_quadrature=True é mais preciso fisicamente
            - add_quadrature=False é mais conservador
        """
        try:
            if thermal.shape != atmospheric.shape:
                msg = f"Shapes inconsistentes: thermal {thermal.shape} vs atmospheric {atmospheric.shape}"
                raise NoiseGenerationError(msg)

            if add_quadrature:
                # Soma quadrática: sqrt(T_th^2 + T_atm^2)
                t_total = np.sqrt(thermal**2 + atmospheric**2)
            else:
                # Adição linear
                t_total = thermal + atmospheric

            # Garantir sem NaNs
            if np.any(np.isnan(t_total)) or np.any(np.isinf(t_total)):
                msg = "Composição resultou em NaN ou inf"
                raise NoiseGenerationError(msg)

            if np.any(t_total < 0):
                msg = "Ruído composto não pode ser negativo"
                raise NoiseGenerationError(msg)

            return t_total

        except NoiseGenerationError:
            raise
        except Exception as e:
            msg = f"Erro ao compor ruído: {e}"
            raise NoiseGenerationError(msg) from e

    @staticmethod
    def estimate_sensitivity(
        noise: Noise,
        bandwidth: float | Bandwidth,
        integration_time: float | IntegrationTime,
    ) -> float:
        """
        Estima sensibilidade de detecção (NEFD - Noise Equivalent Flux Density).

        Fórmula:
            NEFD = T_sys / sqrt(BW * tau)

        Onde:
            T_sys: Temperatura de sistema
            BW: Largura de banda
            tau: Tempo de integração

        Args:
            noise: Instância de Noise
            bandwidth: Largura de banda em Hz (float ou Bandwidth)
            integration_time: Tempo de integração em segundos (float ou IntegrationTime)

        Returns:
            NEFD em Kelvin (sensibilidade mínima detectável)

        Raises:
            InvalidNoiseParameterError: Se parâmetros inválidos

        Notes:
            - NEFD menor = sistema mais sensível
            - Reduz inversamente com sqrt(BW * tau)
            - Importante para design de experimentos
        """
        # Converter value objects
        bw = bandwidth.value if isinstance(bandwidth, Bandwidth) else float(bandwidth)

        if isinstance(integration_time, IntegrationTime):
            tau = integration_time.value
        else:
            tau = float(integration_time)

        # Validar
        if bw <= 0:
            msg = "bandwidth deve ser > 0"
            raise InvalidNoiseParameterError(msg)
        if tau <= 0:
            msg = "integration_time deve ser > 0"
            raise InvalidNoiseParameterError(msg)

        try:
            nefd = noise.t_sys / np.sqrt(bw * tau)
            return float(nefd)
        except Exception as e:
            msg = f"Erro ao estimar sensibilidade: {e}"
            raise InvalidNoiseParameterError(msg) from e

    @staticmethod
    def verify_noise_properties(
        noise_array: np.ndarray,
        t_sys_expected: float,
        tolerance: float = 0.2,
    ) -> dict[str, Any]:
        """
        Verifica propriedades físicas de um array de ruído.

        Checklist:
        - Sem valores negativos
        - Sem NaN ou inf
        - Ordem de grandeza coerente

        Args:
            noise_array: Array de ruído em Kelvin
            t_sys_expected: Temperatura de sistema esperada (Kelvin)
            tolerance: Tolerância relativa (padrão: 20%)

        Returns:
            Dicionário com resultados da verificação

        Raises:
            NoiseGenerationError: Se verificação falha criticamente
        """
        results = {
            "is_positive": False,
            "is_finite": False,
            "is_reasonable_magnitude": False,
            "all_valid": False,
        }

        try:
            # Verificar positividade
            if np.all(noise_array >= 0):
                results["is_positive"] = True
            else:
                msg = f"Array contém valores negativos: mín={noise_array.min():.4f}"
                raise NoiseGenerationError(msg)

            # Verificar finitude
            if np.all(np.isfinite(noise_array)):
                results["is_finite"] = True
            else:
                msg = "Array contém NaN ou inf"
                raise NoiseGenerationError(msg)

            # Verificar ordem de grandeza
            mean_noise = np.mean(noise_array)
            lower_bound = t_sys_expected * (1 - tolerance)
            upper_bound = t_sys_expected * (1 + tolerance)

            if lower_bound <= mean_noise <= upper_bound:
                results["is_reasonable_magnitude"] = True

            # Resultado final: todos os checks devem passar
            results["all_valid"] = (
                results["is_positive"]
                and results["is_finite"]
                and results["is_reasonable_magnitude"]
            )

            return results

        except NoiseGenerationError:
            raise
        except Exception as e:
            msg = f"Erro ao verificar propriedades: {e}"
            raise NoiseGenerationError(msg) from e

    @staticmethod
    def create_noise_system(
        t_sys: float,
        efficiency: float = 0.95,
        t_cmb: float = 2.73,
        t_ground: float = 300.0,
    ) -> Noise:
        """
        Factory method para criar instância de Noise com parâmetros típicos.

        Args:
            t_sys: Temperatura de sistema em Kelvin
            efficiency: Eficiência do sistema (padrão: 0.95)
            t_cmb: Temperatura CMB (padrão: 2.73 K, não usado mas documentado)
            t_ground: Temperatura do solo (padrão: 300 K, não usada mas documentada)

        Returns:
            Instância de Noise configurada

        Raises:
            InvalidNoiseParameterError: Se parâmetros inválidos
        """
        _ = (t_cmb, t_ground)
        try:
            return Noise(t_sys=t_sys, efficiency=efficiency)
        except InvalidNoiseParameterError:
            raise
        except Exception as e:
            msg = f"Erro ao criar sistema de ruído: {e}"
            raise InvalidNoiseParameterError(msg) from e
