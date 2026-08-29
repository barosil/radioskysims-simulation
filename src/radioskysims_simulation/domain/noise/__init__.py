"""
domain/noise/__init__.py

API pública do domínio Noise (ruído em radiotelescópios).

Este pacote fornece modelos, value objects, serviços e exceções para
modelar ruído físico em radiotelescópios de forma pura e testável.

Classes Principais:
    Noise: Entidade representando sistema de ruído
    NoiseService: Serviço de domínio para operações com Noise
    
Value Objects:
    SystemTemperature: Temperatura de sistema (Kelvin)
    Efficiency: Eficiência do sistema (0-1)
    Bandwidth: Largura de banda (Hz)
    IntegrationTime: Tempo de integração (segundos)
    
Exceções:
    InvalidNoiseParameterError: Parâmetro violou restrição
    NoiseGenerationError: Falha ao gerar ruído

Exemplo de Uso:
    from radioskysims_simulation.domain.noise import Noise, NoiseService
    
    # Criar sistema de ruído
    noise = Noise(t_sys=50.0, efficiency=0.95)
    
    # Calcular ruído térmico
    frequencies = np.array([1.4e9, 1.5e9])  # Hz
    t_thermal = noise.thermal_noise_power(
        frequencies, 
        bandwidth=1e6,  # 1 MHz
        integration_time=1.0  # 1 segundo
    )
    
    # Calcular ruído atmosférico
    elevations = np.array([np.pi/4, np.pi/6])  # radianos
    t_atm = noise.atmospheric_noise(elevations, frequency=1.4e9)
    
    # Compor ruído total
    t_total = NoiseService.compose_noise(t_thermal, t_atm, add_quadrature=True)
    
    # Gerar série temporal
    times = np.linspace(0, 1, 1000)
    elevations_ts = np.linspace(np.pi/6, np.pi/2, 1000)
    tod = noise.generate_noise_time_series(
        frequencies, times, elevations_ts, 
        bandwidth=1e6, seed=42
    )
"""

from __future__ import annotations

from .exceptions import InvalidNoiseParameterError, NoiseGenerationError
from .models import Noise
from .services import NoiseService
from .simulator import NoiseSimulator
from .value_objects import Bandwidth, Efficiency, IntegrationTime, SystemTemperature

__all__ = [
    "Bandwidth",
    "Efficiency",
    "IntegrationTime",
    "InvalidNoiseParameterError",
    "Noise",
    "NoiseGenerationError",
    "NoiseService",
    "NoiseSimulator",
    "SystemTemperature",
]
