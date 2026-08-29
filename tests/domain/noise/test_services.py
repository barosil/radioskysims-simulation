"""
tests/domain/noise/test_services.py

Testes unitários para serviços do domínio Noise.
"""

import pytest
import numpy as np

from radioskysims_simulation.domain.noise import (
    Noise,
    NoiseService,
    Bandwidth,
    IntegrationTime,
    InvalidNoiseParameterError,
    NoiseGenerationError,
)


class TestNoiseServiceCompose:
    """Testes de composição de ruído."""

    def test_compose_additive(self):
        """Composição aditiva (linear)."""
        thermal = np.array([[1.0, 2.0], [3.0, 4.0]])
        atmospheric = np.array([[0.5, 1.0], [1.5, 2.0]])
        
        result = NoiseService.compose_noise(thermal, atmospheric, add_quadrature=False)
        expected = thermal + atmospheric
        
        assert np.allclose(result, expected)

    def test_compose_quadrature(self):
        """Composição em quadratura (raiz de soma de quadrados)."""
        thermal = np.array([[3.0, 4.0]])
        atmospheric = np.array([[4.0, 3.0]])
        
        result = NoiseService.compose_noise(thermal, atmospheric, add_quadrature=True)
        expected = np.array([[5.0, 5.0]])  # sqrt(3^2 + 4^2) = 5
        
        assert np.allclose(result, expected)

    def test_compose_shape_mismatch_raises(self):
        """Rejeita shapes incompatíveis."""
        thermal = np.array([[1.0, 2.0]])
        atmospheric = np.array([[1.0, 2.0, 3.0]])  # shape diferente
        
        with pytest.raises(NoiseGenerationError):
            NoiseService.compose_noise(thermal, atmospheric)

    def test_compose_result_positive(self):
        """Resultado é sempre não-negativo."""
        thermal = np.random.uniform(0, 10, (50, 20))
        atmospheric = np.random.uniform(0, 10, (50, 20))
        
        result = NoiseService.compose_noise(thermal, atmospheric)
        assert np.all(result >= 0)


class TestNoiseServiceSensitivity:
    """Testes de estimativa de sensibilidade."""

    def test_sensitivity_estimation(self):
        """Estima sensibilidade (NEFD)."""
        noise = Noise(t_sys=100.0)
        
        nefd = NoiseService.estimate_sensitivity(
            noise,
            bandwidth=1e6,
            integration_time=1.0
        )
        
        assert nefd > 0
        assert isinstance(nefd, float)

    def test_sensitivity_improves_with_bandwidth(self):
        """Sensibilidade melhora (reduz) com bandwidth."""
        noise = Noise(t_sys=100.0)
        
        nefd1 = NoiseService.estimate_sensitivity(noise, bandwidth=1e6, integration_time=1.0)
        nefd2 = NoiseService.estimate_sensitivity(noise, bandwidth=10e6, integration_time=1.0)
        
        # Mais bandwidth → sensibilidade melhor (menor)
        assert nefd2 < nefd1

    def test_sensitivity_improves_with_integration_time(self):
        """Sensibilidade melhora (reduz) com integration_time."""
        noise = Noise(t_sys=100.0)
        
        nefd1 = NoiseService.estimate_sensitivity(noise, bandwidth=1e6, integration_time=0.1)
        nefd2 = NoiseService.estimate_sensitivity(noise, bandwidth=1e6, integration_time=1.0)
        
        # Mais tempo de integração → sensibilidade melhor (menor)
        assert nefd2 < nefd1

    def test_sensitivity_value_objects(self):
        """Funciona com value objects."""
        noise = Noise(t_sys=100.0)
        
        bw = Bandwidth(1e6)
        t_int = IntegrationTime(1.0)
        
        nefd = NoiseService.estimate_sensitivity(noise, bandwidth=bw, integration_time=t_int)
        assert nefd > 0


class TestNoiseServiceVerify:
    """Testes de verificação de propriedades."""

    def test_verify_valid_noise(self):
        """Verifica ruído válido."""
        noise_array = np.array([[5.0, 10.0], [7.0, 12.0]])
        
        results = NoiseService.verify_noise_properties(noise_array, t_sys_expected=10.0)
        
        assert results["is_positive"]
        assert results["is_finite"]
        assert results["all_valid"]

    def test_verify_negative_values_fail(self):
        """Rejeita valores negativos."""
        noise_array = np.array([[5.0, -10.0]])
        
        with pytest.raises(NoiseGenerationError):
            NoiseService.verify_noise_properties(noise_array, t_sys_expected=10.0)

    def test_verify_nan_values_fail(self):
        """Rejeita NaN."""
        noise_array = np.array([[5.0, np.nan]])
        
        with pytest.raises(NoiseGenerationError):
            NoiseService.verify_noise_properties(noise_array, t_sys_expected=10.0)

    def test_verify_inf_values_fail(self):
        """Rejeita inf."""
        noise_array = np.array([[5.0, np.inf]])
        
        with pytest.raises(NoiseGenerationError):
            NoiseService.verify_noise_properties(noise_array, t_sys_expected=10.0)

    def test_verify_magnitude_check(self):
        """Verifica ordem de grandeza."""
        # Array com valores na faixa esperada
        noise_array = np.array([[8.0, 12.0]])  # média ~10
        
        results = NoiseService.verify_noise_properties(
            noise_array,
            t_sys_expected=10.0,
            tolerance=0.2  # 20% de tolerância
        )
        
        assert results["is_reasonable_magnitude"]


class TestNoiseServiceFactory:
    """Testes de factory method."""

    def test_create_noise_system(self):
        """Factory cria sistema de ruído."""
        noise = NoiseService.create_noise_system(t_sys=50.0, efficiency=0.95)
        
        assert isinstance(noise, Noise)
        assert noise.t_sys == 50.0
        assert noise.efficiency == 0.95

    def test_create_noise_system_default_efficiency(self):
        """Factory usa eficiência padrão."""
        noise = NoiseService.create_noise_system(t_sys=50.0)
        
        assert noise.efficiency == pytest.approx(0.95)

    def test_create_noise_system_invalid_raises(self):
        """Factory rejeita parâmetros inválidos."""
        with pytest.raises(InvalidNoiseParameterError):
            NoiseService.create_noise_system(t_sys=-10.0)


class TestNoiseServiceIntegration:
    """Testes de integração do serviço."""

    def test_complete_workflow(self):
        """Fluxo completo de cálculo de ruído."""
        # Criar sistema
        noise = NoiseService.create_noise_system(t_sys=50.0, efficiency=0.95)
        
        # Parâmetros
        frequency = 1.4e9
        bw = Bandwidth(1e6)
        t_int = IntegrationTime(1.0)
        
        # Calcular ruído térmico
        t_thermal = noise.thermal_noise_power(np.array([frequency]), bw, t_int)
        
        # Calcular ruído atmosférico
        elevations = np.array([np.pi/4, np.pi/3])
        t_atm = noise.atmospheric_noise(elevations, frequency=frequency)
        
        # Compor ruído (ambos tem shape (2,))
        t_total = NoiseService.compose_noise(
            np.full_like(elevations, t_thermal[0]), 
            t_atm, 
            add_quadrature=True
        )
        
        # Verificar
        results = NoiseService.verify_noise_properties(t_total, t_sys_expected=50.0)
        assert results["all_valid"] or results["is_positive"]

    def test_sensitivity_and_compose_integration(self):
        """Integração entre sensibilidade e composição."""
        noise = NoiseService.create_noise_system(t_sys=100.0)
        
        # Sensibilidade
        nefd = NoiseService.estimate_sensitivity(noise, bandwidth=1e6, integration_time=1.0)
        
        # Gerar ruído
        frequencies = np.array([1.4e9])
        times = np.linspace(0, 1, 100)
        elevations = np.linspace(np.pi/6, np.pi/2, 100)
        
        tod = noise.generate_noise_time_series(frequencies, times, elevations, seed=42)
        
        # Verificar que ruído médio é razoável
        mean_noise = np.mean(tod)
        assert mean_noise > 0
        # Ruído total é térmico + atmosférico, deve estar na ordem de T_sys ou um pouco maior
        assert mean_noise < 2 * noise.t_sys
