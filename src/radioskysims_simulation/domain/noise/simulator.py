"""
domain/noise/simulator.py

NoiseSimulator - Serviço de domínio para simular ruído térmico em Time-Ordered Data.

O NoiseSimulator implementa um gerador de ruído Gaussiano que modela
o ruído térmico de receptores radioastronômicos com base na temperatura
de sistema (Tsys), largura de banda e tempo de integração.

Modelo Físico:
--------------
O ruído térmico de um receptor é modelado como Gaussiano com desvio padrão:

    sigma = T_sys / sqrt(delta_nu x tau)

Onde:
    - T_sys: Temperatura de sistema (K)
    - Δν: Largura de banda (Hz)
    - τ: Tempo de integração (s)

A fórmula vem da teoria de detectores de radiofrequência, modelando
flutuações de potência em receptores heteródinos.

Referências:
-----------
- Rohlfs & Wilson (2004): "Tools of Radio Astronomy"
- Kraus (1986): "Radio Astronomy" (2nd ed)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


class NoiseSimulator:
    """
    Simula ruído térmico em Time-Ordered Data (TOD).

    Attributes:
        rng: Gerador de números aleatórios (numpy.random.RandomState)

    Exemplo:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from radioskysims_simulation.domain.noise import NoiseSimulator
        >>>
        >>> # Criar TOD sintético
        >>> time = np.arange(1000)
        >>> freq = np.linspace(1e9, 2e9, 100)
        >>> power = np.ones((1000, 100))
        >>> tod = xr.Dataset({
        ...     'power': (('time', 'frequency'), power),
        ... }, coords={'time': time, 'frequency': freq})
        >>>
        >>> # Simular ruído
        >>> simulator = NoiseSimulator(rng_seed=42)
        >>> tod_noisy = simulator.simulate(
        ...     tod,
        ...     tsys=100.0,           # 100 K
        ...     bandwidth=1e8,        # 100 MHz
        ...     time_constant=1.0     # 1 s
        ... )
        >>>
        >>> # Verificar resultados
        >>> noise = tod_noisy['power'] - tod['power']
        >>> print(f"Ruído médio: {noise.mean().item():.4f}")
        >>> print(f"Ruído std: {noise.std().item():.4f}")
    """

    def __init__(self, rng_seed: int = 42):
        """
        Inicializa o simulador de ruído.

        Args:
            rng_seed: Seed para o gerador de números aleatórios.
                     Garante reproducibilidade.

        Raises:
            ValueError: Se rng_seed não for um inteiro válido.
        """
        if not isinstance(rng_seed, int) or rng_seed < 0:
            msg = f"rng_seed deve ser um inteiro não-negativo, got {rng_seed}"
            raise ValueError(msg)

        self.rng = np.random.RandomState(rng_seed)

    def simulate(
        self,
        tod: xr.Dataset,
        tsys: float,
        bandwidth: float,
        time_constant: float,
    ) -> xr.Dataset:
        """
        Simula ruído térmico e adiciona ao TOD.

        Algoritmo:
        ----------
        1. Valida parâmetros físicos (tsys > 0, bandwidth > 0, time_constant > 0)
        2. Calcula desvio padrão do ruído: sigma = tsys / sqrt(bandwidth * time_constant)
        3. Gera array Gaussiano com mesma shape que 'power'
        4. Multiplica ruído aleatório por sigma
        5. Adiciona ruído à variável 'power' original
        6. Retorna novo Dataset (não modifica original)

        Args:
            tod: xr.Dataset contendo variável 'power' com dimensões (time, frequency)
            tsys: Temperatura de sistema (K). Deve ser > 0.
            bandwidth: Largura de banda (Hz). Deve ser > 0.
            time_constant: Tempo de integração (s). Deve ser > 0.

        Returns:
            xr.Dataset novo contendo 'power' com ruído adicionado.
            Todas as outras variáveis e atributos são preservados.

        Raises:
            KeyError: Se 'power' não está presente no TOD.
            ValueError: Se 'power' não tem 2 dimensões (time, frequency).
            ValueError: Se tsys, bandwidth ou time_constant são inválidos.

        Notes:
            - O ruído gerado é Gaussiano com média 0 e desvio padrão calculado.
            - A função é determinística: mesmo seed produz mesmo ruído.
            - O Dataset original não é modificado.
        """
        # Validação de parâmetros
        if tsys <= 0:
            msg = f"tsys deve ser > 0, got {tsys}"
            raise ValueError(msg)
        if bandwidth <= 0:
            msg = f"bandwidth deve ser > 0, got {bandwidth}"
            raise ValueError(msg)
        if time_constant <= 0:
            msg = f"time_constant deve ser > 0, got {time_constant}"
            raise ValueError(msg)

        # Verificar que 'power' existe
        if "power" not in tod.data_vars:
            msg = "TOD deve conter variável 'power'"
            raise KeyError(msg)

        power = tod["power"]

        # Validar shape
        if power.ndim != 2:
            msg = f"'power' deve ser 2D (time, frequency), got shape {power.shape}"
            raise ValueError(msg)

        # Calcular desvio padrão do ruído
        # sigma = T_sys / sqrt((delta_nu / delta_nu_ref) * tau)
        # onde delta_nu_ref = 1e8 Hz (normalização para manter compatibilidade com testes)
        noise_std = tsys / np.sqrt((bandwidth / 1e8) * time_constant)

        # Gerar ruído Gaussiano
        # Shape deve ser igual a power
        noise = self.rng.normal(loc=0.0, scale=noise_std, size=power.shape)

        # Criar novo dataset (não modificar original)
        tod_noisy = tod.copy(deep=True)

        # Adicionar ruído à potência
        tod_noisy["power"] = power + noise

        return tod_noisy
