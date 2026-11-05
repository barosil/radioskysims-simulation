"""
tests/domain/noise/test_models.py

Testes unitários para modelos do domínio Noise.
"""

import pytest
import numpy as np

from radioskysims_simulation.domain.noise import (
    Noise,
    SystemTemperature,
    Efficiency,
    Bandwidth,
    IntegrationTime,
    InvalidNoiseParameterError,
    NoiseGenerationError,
)


class TestNoiseCreation:
    """Testes de criação de Noise."""

    def test_valid_creation_floats(self):
        """Cria Noise com floats."""
        noise = Noise(t_sys=50.0, efficiency=0.95)
        assert noise.t_sys == 50.0
        assert noise.efficiency == 0.95

    def test_valid_creation_value_objects(self):
        """Cria Noise com value objects."""
        t_sys = SystemTemperature(50.0)
        eff = Efficiency(0.95)
        noise = Noise(t_sys=t_sys, efficiency=eff)
        assert noise.t_sys == 50.0
        assert noise.efficiency == 0.95

    def test_default_efficiency(self):
        """Eficiência padrão é 0.95."""
        noise = Noise(t_sys=50.0)
        assert noise.efficiency == pytest.approx(0.95)

    def test_invalid_t_sys_raises(self):
        """Rejeita t_sys inválida."""
        with pytest.raises(InvalidNoiseParameterError):
            Noise(t_sys=-10.0)

    def test_invalid_efficiency_raises(self):
        """Rejeita eficiência inválida."""
        with pytest.raises(InvalidNoiseParameterError):
            Noise(t_sys=50.0, efficiency=1.5)


class TestThermalNoise:
    """Testes de cálculo de ruído térmico."""

    def test_thermal_noise_positive(self):
        """Ruído térmico é sempre positivo."""
        noise = Noise(t_sys=50.0)
        frequencies = np.array([1.4e9, 1.5e9])
        t_thermal = noise.thermal_noise_power(frequencies, bandwidth=1e6, integration_time=1.0)

        assert np.all(t_thermal >= 0)
        assert t_thermal.shape == frequencies.shape

    def test_thermal_noise_order_of_magnitude(self):
        """Ruído térmico está na ordem de magnitude esperada."""
        noise = Noise(t_sys=50.0)
        frequencies = np.array([1.4e9])
        t_thermal = noise.thermal_noise_power(frequencies, bandwidth=1e6, integration_time=1.0)

        # Ruído térmico reduzido deve estar na ordem de 0.1-1 K para T_sys=50K
        assert t_thermal[0] < noise.t_sys

    def test_thermal_noise_increases_with_bandwidth(self):
        """Ruído térmico aumenta com largura de banda."""
        noise = Noise(t_sys=50.0)
        frequencies = np.array([1.4e9])

        t1 = noise.thermal_noise_power(frequencies, bandwidth=1e6, integration_time=1.0)
        t2 = noise.thermal_noise_power(frequencies, bandwidth=10e6, integration_time=1.0)

        # Maiores largura de banda → mais ruído
        assert t2[0] > t1[0]

    def test_thermal_noise_decreases_with_integration_time(self):
        """Ruído reduz com maior tempo de integração."""
        noise = Noise(t_sys=50.0)
        frequencies = np.array([1.4e9])

        t1 = noise.thermal_noise_power(frequencies, bandwidth=1e6, integration_time=0.1)
        t2 = noise.thermal_noise_power(frequencies, bandwidth=1e6, integration_time=1.0)

        # Maior tempo de integração → menos ruído
        assert t1[0] > t2[0]

    def test_thermal_noise_value_objects(self):
        """Funciona com value objects para bandwidth e integration_time."""
        noise = Noise(t_sys=50.0)
        frequencies = np.array([1.4e9])

        bw = Bandwidth(1e6)
        t_int = IntegrationTime(1.0)

        t_thermal = noise.thermal_noise_power(frequencies, bandwidth=bw, integration_time=t_int)
        assert t_thermal.shape == frequencies.shape


class TestAtmosphericNoise:
    """Testes de cálculo de ruído atmosférico."""

    def test_atmospheric_noise_positive(self):
        """Ruído atmosférico é sempre positivo."""
        noise = Noise(t_sys=50.0)
        elevations = np.array([np.pi / 4, np.pi / 6])
        t_atm = noise.atmospheric_noise(elevations, frequency=1.4e9)

        assert np.all(t_atm >= 0)

    def test_atmospheric_noise_increases_at_low_elevation(self):
        """Ruído atmosférico aumenta com elevação baixa."""
        noise = Noise(t_sys=50.0)

        # Elevação baixa
        t_low = noise.atmospheric_noise(np.array([np.pi / 6]), frequency=1.4e9)

        # Elevação alta (próxima ao zênite)
        t_high = noise.atmospheric_noise(np.array([np.pi / 2 - 0.1]), frequency=1.4e9)

        # Elevação baixa → mais ruído atmosférico
        assert t_low[0] > t_high[0]

    def test_atmospheric_noise_invalid_elevation_raises(self):
        """Rejeita elevações fora do range (0, π/2]."""
        noise = Noise(t_sys=50.0)

        # Elevação zero
        with pytest.raises(NoiseGenerationError):
            noise.atmospheric_noise(np.array([0.0]), frequency=1.4e9)

        # Elevação > π/2
        with pytest.raises(NoiseGenerationError):
            noise.atmospheric_noise(np.array([np.pi]), frequency=1.4e9)

    def test_atmospheric_noise_single_value(self):
        """Funciona com elevação escalar."""
        noise = Noise(t_sys=50.0)
        t_atm = noise.atmospheric_noise(np.pi / 4, frequency=1.4e9)

        assert isinstance(t_atm, np.ndarray)
        assert t_atm.size >= 1


class TestNoiseTimeSeries:
    """Testes de geração de série temporal de ruído."""

    def test_time_series_shape(self):
        """Série temporal tem shape correto."""
        noise = Noise(t_sys=50.0)

        frequencies = np.array([1.4e9, 1.5e9, 1.6e9])  # 3 frequências
        times = np.linspace(0, 1, 100)  # 100 tempos
        elevations = np.linspace(np.pi / 6, np.pi / 2, 100)

        tod = noise.generate_noise_time_series(frequencies, times, elevations)

        assert tod.shape == (len(times), len(frequencies))

    def test_time_series_positive(self):
        """Série temporal tem valores não-negativos."""
        noise = Noise(t_sys=50.0)

        frequencies = np.array([1.4e9, 1.5e9])
        times = np.linspace(0, 1, 50)
        elevations = np.linspace(np.pi / 6, np.pi / 2, 50)

        tod = noise.generate_noise_time_series(frequencies, times, elevations, seed=42)

        assert np.all(tod >= 0)

    def test_time_series_deterministic_with_seed(self):
        """Série temporal é determinística com seed fixo."""
        noise = Noise(t_sys=50.0)

        frequencies = np.array([1.4e9])
        times = np.linspace(0, 1, 100)
        elevations = np.linspace(np.pi / 6, np.pi / 2, 100)

        tod1 = noise.generate_noise_time_series(frequencies, times, elevations, seed=42)
        tod2 = noise.generate_noise_time_series(frequencies, times, elevations, seed=42)

        assert np.allclose(tod1, tod2)

    def test_time_series_different_without_seed(self):
        """Série temporal é diferente sem seed."""
        noise = Noise(t_sys=50.0)

        frequencies = np.array([1.4e9])
        times = np.linspace(0, 1, 100)
        elevations = np.linspace(np.pi / 6, np.pi / 2, 100)

        tod1 = noise.generate_noise_time_series(frequencies, times, elevations)
        tod2 = noise.generate_noise_time_series(frequencies, times, elevations)

        # Extremamente improvável serem iguais
        assert not np.allclose(tod1, tod2)

    def test_time_series_elevation_broadcast(self):
        """Elevação escalar é broadcastada."""
        noise = Noise(t_sys=50.0)

        frequencies = np.array([1.4e9, 1.5e9])
        times = np.linspace(0, 1, 50)
        elevation = np.pi / 4  # escalar

        tod = noise.generate_noise_time_series(frequencies, times, elevation, seed=42)

        assert tod.shape == (50, 2)

    def test_time_series_invalid_elevation_shape_raises(self):
        """Rejeita elevation shape incompatível."""
        noise = Noise(t_sys=50.0)

        frequencies = np.array([1.4e9])
        times = np.linspace(0, 1, 100)
        elevations = np.linspace(np.pi / 6, np.pi / 2, 50)  # shape (50,) vs times (100,)

        with pytest.raises(NoiseGenerationError):
            noise.generate_noise_time_series(frequencies, times, elevations)


class TestNoiseEquality:
    """Testes de igualdade."""

    def test_equal_noise(self):
        """Noise com mesmos parâmetros são iguais."""
        noise1 = Noise(t_sys=50.0, efficiency=0.95)
        noise2 = Noise(t_sys=50.0, efficiency=0.95)

        assert noise1 == noise2

    def test_different_noise(self):
        """Noise com parâmetros diferentes não são iguais."""
        noise1 = Noise(t_sys=50.0)
        noise2 = Noise(t_sys=60.0)

        assert noise1 != noise2


class TestNoiseRepresentation:
    """Testes de representação."""

    def test_repr(self):
        """Representação técnica."""
        noise = Noise(t_sys=50.0, efficiency=0.95)
        r = repr(noise)

        assert "Noise" in r
        assert "50" in r
        assert "0.95" in r

    def test_str(self):
        """Representação amigável."""
        noise = Noise(t_sys=50.0)
        s = str(noise)

        assert "T_sys" in s
        assert "50" in s
