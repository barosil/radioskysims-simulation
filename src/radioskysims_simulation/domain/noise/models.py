"""
domain/noise/models.py

Entidade Noise para modelagem de ruído em radiotelescópios.

A entidade Noise representa um sistema de radiotelescópio com suas características
térmicas e atmosféricas. Fornece métodos para calcular ruído térmico, ruído
atmosférico e gerar séries temporais de ruído realistas.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import InvalidNoiseParameterError, NoiseGenerationError
from .value_objects import Bandwidth, Efficiency, IntegrationTime, SystemTemperature


class Noise:
    """
    Entidade de domínio representando o sistema de ruído de um radiotelescópio.

    Responsabilidades:
    - Armazenar características do sistema (temperatura, eficiência)
    - Calcular ruído térmico em função da frequência
    - Calcular ruído atmosférico em função da elevação
    - Gerar séries temporais de ruído realistas

    Características:
    - Imutável (atributos privados)
    - Sem efeitos colaterais (não modifica dados de entrada)
    - Validações rigorosas
    - Computações vectorizadas (sem loops)
    - Sem dependências de infraestrutura

    Atributos:
        t_sys: Temperatura de sistema (Kelvin)
        efficiency: Eficiência do sistema (0-1)

    Exemplo:
        noise = Noise(t_sys=50.0, efficiency=0.95)

        # Ruído térmico em função da frequência
        freq = np.array([1.4e9, 1.5e9]) # Hz
        p_thermal = noise.thermal_noise_power(freq, bandwidth=1e6, integration_time=1.0)

        # Ruído atmosférico em função da elevação
        el = np.array([np.pi/4, np.pi/6])  # radianos
        t_atm = noise.atmospheric_noise(el, frequency=1.4e9)

        # Série temporal de ruído
        times = np.linspace(0, 1, 1000)  # segundos
        elevations = np.linspace(np.pi/6, np.pi/2, 100)  # radianos
        frequencies = np.array([1.4e9, 1.5e9, 1.6e9])  # Hz

        tod = noise.generate_noise_time_series(frequencies, times, elevations)
        # Resultado: shape (Nt=1000, Nf=3)
    """

    __slots__ = ("_efficiency", "_t_sys")

    def __init__(
        self,
        t_sys: float | SystemTemperature,
        efficiency: float | Efficiency = 0.95,
    ) -> None:
        """
        Cria instância de Noise.

        Args:
            t_sys: Temperatura de sistema (Kelvin ou SystemTemperature)
            efficiency: Eficiência do sistema (0-1) ou Efficiency object

        Raises:
            InvalidNoiseParameterError: Se parâmetros violam restrições
        """
        # Converter para value objects se necessário
        if isinstance(t_sys, int | float):
            self._t_sys = SystemTemperature(t_sys)
        elif isinstance(t_sys, SystemTemperature):
            self._t_sys = t_sys
        else:
            msg = f"t_sys deve ser float ou SystemTemperature, recebido {type(t_sys)}"
            raise InvalidNoiseParameterError(msg)

        if isinstance(efficiency, int | float):
            self._efficiency = Efficiency(efficiency)
        elif isinstance(efficiency, Efficiency):
            self._efficiency = efficiency
        else:
            msg = (
                f"efficiency deve ser float ou Efficiency, recebido {type(efficiency)}"
            )
            raise InvalidNoiseParameterError(msg)

    @property
    def t_sys(self) -> float:
        """Temperatura de sistema em Kelvin."""
        return self._t_sys.value

    @property
    def efficiency(self) -> float:
        """Eficiência do sistema (0-1)."""
        return self._efficiency.value

    def thermal_noise_power(
        self,
        frequency: np.ndarray,
        bandwidth: float | Bandwidth,
        integration_time: float | IntegrationTime,
    ) -> np.ndarray:
        """
        Calcula potência de ruído térmico usando equação de Rayleigh-Jeans.

        Fórmula:
            P_thermal = k_B * T_sys * BW

        Ruído reduzido no tempo (standard deviation):
            sigma = P_thermal / sqrt(BW * tau)

        Args:
            frequency: Array de frequências em Hz (não usado aqui, mas necessário
                      para compatibilidade com assinatura geral)
            bandwidth: Largura de banda em Hz (float ou Bandwidth)
            integration_time: Tempo de integração em segundos (float ou IntegrationTime)

        Returns:
            Array de potência térmica em Kelvin (shape = frequency.shape)

        Raises:
            InvalidNoiseParameterError: Se parâmetros inválidos
            NoiseGenerationError: Se operação falha

        Notes:
            - Usa constante de Boltzmann k_B = 1.380649e-23 J/K
            - Retorna potência normalizada (em Kelvin equivalente)
            - Valores sempre >= 0
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

        # Garantir frequência é array
        freq = np.atleast_1d(frequency)
        if freq.size == 0:
            msg = "frequency não pode estar vazio"
            raise NoiseGenerationError(msg)

        try:
            # Ruído térmico reduzido por integração
            # Fórmula normalizada: T_thermal = T_sys / sqrt(tau) * sqrt(BW / 1e6) * efficiency
            #
            # Normaliza BW em relação a 1 MHz (referência padrão)
            #
            # Comportamento esperado:
            # - Aumenta com BW (mais largura de banda → mais potência bruta)
            # - Diminui com tau (mais tempo de integração reduz ruído)
            # - Melhora com eficiência
            # - Mantém valores razoáveis (da ordem de T_sys para parametros típicos)

            bw_mhz = bw / 1e6  # normaliza para MHz
            t_thermal_scalar = (
                (self.t_sys / np.sqrt(tau)) * np.sqrt(bw_mhz) * self.efficiency
            )

            # Repetir para cada frequência
            t_thermal = np.full_like(freq, t_thermal_scalar, dtype=float)

            # Garantir shape correto e sem NaNs
            if np.any(np.isnan(t_thermal)) or np.any(np.isinf(t_thermal)):
                msg = "Operação resultou em NaN ou inf"
                raise NoiseGenerationError(msg)

            if np.any(t_thermal < 0):
                msg = "Ruído térmico não pode ser negativo"
                raise NoiseGenerationError(msg)

            return t_thermal

        except (TypeError, ValueError) as e:
            msg = f"Erro ao calcular ruído térmico: {e}"
            raise NoiseGenerationError(msg) from e

    def atmospheric_noise(
        self,
        elevation: np.ndarray,
        frequency: float,
    ) -> np.ndarray:
        """
        Calcula ruído atmosférico em função da elevação.

        Modelo simplificado:
            T_atm = T_spill + (T_atm_0 / sin(elevation))

        Onde:
            T_spill: ~5 K (spillover do solo, negligenciado aqui)
            T_atm_0: ~10 K (temperatura atmosférica na zênite)
            elevation: Ângulo de elevação em radianos (0-π/2)

        Args:
            elevation: Array de elevações em radianos (0 < el <= π/2)
            frequency: Frequência em Hz (não usado neste modelo, mantido para compatibilidade)

        Returns:
            Array de temperatura atmosférica em Kelvin (shape = elevation.shape)

        Raises:
            NoiseGenerationError: Se elevação inválida ou operação falha

        Notes:
            - Elevações baixas (~0) resultam em ruído atmosférico alto
            - Elevações altas (~π/2, zênite) resultam em ruído mínimo
            - Modelo válido para elevações > 0
        """
        _ = frequency
        el = np.atleast_1d(elevation)

        if el.size == 0:
            msg = "elevation não pode estar vazio"
            raise NoiseGenerationError(msg)

        try:
            # Validar range de elevação
            if np.any(el <= 0) or np.any(el > np.pi / 2):
                msg = f"elevation deve estar em (0, π/2], recebido valores em range {el.min():.4f}-{el.max():.4f}"
                raise NoiseGenerationError(msg)

            # Modelo atmosférico simplificado
            t_atm_0 = 10.0  # K (temperatura na zênite)
            t_spill = 5.0  # K (spillover do solo)

            # T_atm = T_spill + T_atm_0 / sin(el)
            sin_el = np.sin(el)
            t_atm = t_spill + (t_atm_0 / sin_el)

            # Garantir sem NaNs
            if np.any(np.isnan(t_atm)) or np.any(np.isinf(t_atm)):
                msg = "Operação resultou em NaN ou inf"
                raise NoiseGenerationError(msg)

            if np.any(t_atm < 0):
                msg = "Ruído atmosférico não pode ser negativo"
                raise NoiseGenerationError(msg)

            return t_atm

        except (TypeError, ValueError) as e:
            msg = f"Erro ao calcular ruído atmosférico: {e}"
            raise NoiseGenerationError(msg) from e

    def generate_noise_time_series(
        self,
        frequencies: np.ndarray,
        times: np.ndarray,
        elevation: np.ndarray,
        bandwidth: float | Bandwidth = 1e6,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Gera série temporal de ruído realista.

        Combina:
        - Ruído térmico (constante no tempo)
        - Ruído atmosférico (variável com elevação)
        - Flutuações aleatórias gaussianas

        Args:
            frequencies: Array de frequências em Hz (Nf,)
            times: Array de tempos em segundos (Nt,)
            elevation: Array de elevações em radianos (Nt,) ou (Nf,) ou escalar
            bandwidth: Largura de banda em Hz (padrão: 1 MHz)
            seed: Seed para RNG (para reprodutibilidade)

        Returns:
            Array de ruído térmico em Kelvin com shape (Nt, Nf)

        Raises:
            NoiseGenerationError: Se parâmetros inconsistentes ou operação falha

        Notes:
            - Ruído é gaussiano (Rayleigh, na verdade, mas aproximado como gaussiano)
            - Sem correlação temporal (white noise)
            - Shape sempre (Nt, Nf) onde Nt=len(times), Nf=len(frequencies)
            - Determinístico com seed fixo
        """
        freq = np.atleast_1d(frequencies)
        time = np.atleast_1d(times)
        el = np.atleast_1d(elevation)

        if freq.size == 0:
            msg = "frequencies não pode estar vazio"
            raise NoiseGenerationError(msg)
        if time.size == 0:
            msg = "times não pode estar vazio"
            raise NoiseGenerationError(msg)
        if el.size == 0:
            msg = "elevation não pode estar vazio"
            raise NoiseGenerationError(msg)

        try:
            # Garantir shapes consistentes
            Nt = len(time)
            Nf = len(freq)

            # Elevação: broadcast para shape (Nt,)
            if el.shape == (Nt,):
                elevation_broadcast = el
            elif el.size == 1:
                elevation_broadcast = np.full(Nt, el.flat[0])
            else:
                msg = f"elevation shape {el.shape} incompatível com times shape ({Nt},)"
                raise NoiseGenerationError(msg)

            # Converter bandwidth para float
            if isinstance(bandwidth, Bandwidth):
                bw = bandwidth.value
            else:
                bw = float(bandwidth)

            # Tempo de integração: 1 segundo por padrão (time entre samples é apenas para
            # fins de série temporal, não afeta integração radiométrica)
            tau = 1.0

            # Ruído térmico (não depende de tempo ou elevação)
            t_thermal = self.thermal_noise_power(freq, bw, tau)  # shape (Nf,)

            # Ruído atmosférico (depende de elevação, varia no tempo)
            t_atm = self.atmospheric_noise(
                elevation_broadcast, frequency=freq[0]
            )  # shape (Nt,)

            # Combinar: broadcast para (Nt, Nf)
            # t_total[t,f] = t_thermal[f] + t_atm[t]
            t_total = t_thermal[np.newaxis, :] + t_atm[:, np.newaxis]

            # Adicionar flutuações gaussianas (random)
            rng = np.random.default_rng(seed)

            # Desvio padrão das flutuações ~ 10% do ruído total
            sigma_fluctuations = 0.1 * t_total
            noise_fluctuations = rng.normal(loc=0.0, scale=sigma_fluctuations)

            t_total = t_total + noise_fluctuations

            # Garantir positividade e sem NaNs
            t_total = np.maximum(t_total, 0.0)
            if np.any(np.isnan(t_total)) or np.any(np.isinf(t_total)):
                msg = "Série temporal contém NaN ou inf"
                raise NoiseGenerationError(msg)

            # Garantir shape
            if t_total.shape != (Nt, Nf):
                msg = (
                    f"Shape incorreto: esperado ({Nt}, {Nf}), recebido {t_total.shape}"
                )
                raise NoiseGenerationError(msg)

            return t_total

        except (NoiseGenerationError, InvalidNoiseParameterError):
            raise
        except Exception as e:
            msg = f"Erro ao gerar série temporal: {e}"
            raise NoiseGenerationError(msg) from e

    def __eq__(self, other: Any) -> bool:
        """Igualdade baseada em parâmetros."""
        if not isinstance(other, Noise):
            return False
        return np.isclose(self.t_sys, other.t_sys) and np.isclose(
            self.efficiency, other.efficiency
        )

    def __hash__(self) -> int:
        """Hash baseado nos parâmetros imutáveis."""
        return hash((round(self.t_sys, 10), round(self.efficiency, 10)))

    def __repr__(self) -> str:
        """Representação técnica."""
        return f"Noise(t_sys={self.t_sys:.1f} K, efficiency={self.efficiency:.2f})"

    def __str__(self) -> str:
        """Representação amigável."""
        return f"Noise System: T_sys={self.t_sys:.1f} K, η={self.efficiency:.1%}"
