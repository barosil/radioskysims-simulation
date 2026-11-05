"""
tests/domain/noise/test_simulator.py

Testes unitários para o NoiseSimulator.
"""

import pytest
import numpy as np
import xarray as xr

from radioskysims_simulation.domain.noise import NoiseSimulator


class TestNoiseSimulatorInitialization:
    """Testes de inicialização do NoiseSimulator."""

    def test_valid_initialization(self):
        """Inicializa com seed válido."""
        simulator = NoiseSimulator(rng_seed=42)
        assert simulator.rng is not None

    def test_custom_seed(self):
        """Aceita seed customizado."""
        simulator = NoiseSimulator(rng_seed=123)
        assert simulator.rng is not None

    def test_zero_seed(self):
        """Aceita seed 0."""
        simulator = NoiseSimulator(rng_seed=0)
        assert simulator.rng is not None

    def test_negative_seed_raises(self):
        """Rejeita seed negativo."""
        with pytest.raises(ValueError, match="rng_seed deve ser um inteiro não-negativo"):
            NoiseSimulator(rng_seed=-1)

    def test_non_integer_seed_raises(self):
        """Rejeita seed não-inteiro."""
        with pytest.raises(ValueError, match="rng_seed deve ser um inteiro não-negativo"):
            NoiseSimulator(rng_seed=3.14)


class TestNoiseSimulatorBasic:
    """Testes básicos da simulação de ruído."""

    def test_smoke_test(self):
        """Teste smoke do requisito."""
        time = np.arange(1000)
        freq = np.linspace(1e9, 2e9, 100)
        power = np.ones((1000, 100))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=0)
        tod_noisy = simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

        noise = tod_noisy["power"] - tod["power"]
        # Média deve estar perto de zero (permitindo flutuação estatística)
        assert np.abs(noise.mean()) < 1.0
        # Desvio padrão deve estar no intervalo esperado pela fórmula normalizada
        assert 50 < noise.std() < 100

    def test_returns_dataset(self):
        """Retorna um xr.Dataset."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.ones((10, 5))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)
        result = simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

        assert isinstance(result, xr.Dataset)

    def test_preserves_original_tod(self):
        """Não modifica o TOD original."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.ones((10, 5))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        original_power = tod["power"].copy()

        simulator = NoiseSimulator(rng_seed=42)
        simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

        # TOD original deve ser inalterado
        assert np.allclose(tod["power"].values, original_power.values)

    def test_output_different_from_input(self):
        """Saída é diferente da entrada (ruído adicionado)."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.ones((10, 5))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)
        tod_noisy = simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

        assert not np.allclose(tod["power"].values, tod_noisy["power"].values)


class TestNoiseSimulatorProperties:
    """Testes de propriedades do ruído gerado."""

    def test_noise_mean_near_zero(self):
        """Média do ruído é próxima de zero."""
        time = np.arange(1000)
        freq = np.linspace(1e9, 2e9, 100)
        power = np.zeros((1000, 100))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)
        tod_noisy = simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

        noise = tod_noisy["power"].values
        assert np.abs(noise.mean()) < 0.1

    def test_noise_std_matches_formula(self):
        """Desvio padrão do ruído segue a fórmula σ = T_sys / sqrt(Δν × τ)."""
        time = np.arange(1000)
        freq = np.linspace(1e9, 2e9, 100)
        power = np.zeros((1000, 100))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        tsys = 100.0
        bandwidth = 1e8
        time_constant = 1.0
        bandwidth_ref = 1e8  # Normalização padrão

        # Fórmula: σ = T_sys / sqrt((bandwidth / bandwidth_ref) × time_constant)
        expected_std = tsys / np.sqrt((bandwidth / bandwidth_ref) * time_constant)

        simulator = NoiseSimulator(rng_seed=42)
        tod_noisy = simulator.simulate(
            tod, tsys=tsys, bandwidth=bandwidth, time_constant=time_constant
        )

        actual_std = tod_noisy["power"].std().item()

        # Tolerância: ~5% (ruído estatístico)
        assert np.abs(actual_std - expected_std) / expected_std < 0.05

    def test_noise_deterministic_with_seed(self):
        """Mesmo seed produz mesmo ruído."""
        time = np.arange(100)
        freq = np.linspace(1e9, 2e9, 50)
        power = np.ones((100, 50))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator1 = NoiseSimulator(rng_seed=42)
        result1 = simulator1.simulate(tod.copy(), tsys=100, bandwidth=1e8, time_constant=1.0)

        simulator2 = NoiseSimulator(rng_seed=42)
        result2 = simulator2.simulate(tod.copy(), tsys=100, bandwidth=1e8, time_constant=1.0)

        assert np.allclose(result1["power"].values, result2["power"].values)

    def test_different_seeds_produce_different_noise(self):
        """Diferentes seeds produzem ruído diferente."""
        time = np.arange(100)
        freq = np.linspace(1e9, 2e9, 50)
        power = np.ones((100, 50))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator1 = NoiseSimulator(rng_seed=42)
        result1 = simulator1.simulate(tod.copy(), tsys=100, bandwidth=1e8, time_constant=1.0)

        simulator2 = NoiseSimulator(rng_seed=123)
        result2 = simulator2.simulate(tod.copy(), tsys=100, bandwidth=1e8, time_constant=1.0)

        assert not np.allclose(result1["power"].values, result2["power"].values)


class TestNoiseSimulatorInputValidation:
    """Testes de validação de entrada."""

    def test_missing_power_raises(self):
        """Rejeita TOD sem 'power'."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)

        tod = xr.Dataset(coords={"time": time, "frequency": freq})

        simulator = NoiseSimulator(rng_seed=42)
        with pytest.raises(KeyError, match="TOD deve conter variável 'power'"):
            simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

    def test_wrong_dimensions_raises(self):
        """Rejeita 'power' com dimensões erradas."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.ones(10)  # 1D, deveria ser 2D

        tod = xr.Dataset(
            {
                "power": ("time", power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)
        with pytest.raises(ValueError, match="'power' deve ser 2D"):
            simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

    def test_invalid_tsys_raises(self):
        """Rejeita T_sys <= 0."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.ones((10, 5))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)

        with pytest.raises(ValueError, match="tsys deve ser > 0"):
            simulator.simulate(tod, tsys=0, bandwidth=1e8, time_constant=1.0)

        with pytest.raises(ValueError, match="tsys deve ser > 0"):
            simulator.simulate(tod, tsys=-100, bandwidth=1e8, time_constant=1.0)

    def test_invalid_bandwidth_raises(self):
        """Rejeita bandwidth <= 0."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.ones((10, 5))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)

        with pytest.raises(ValueError, match="bandwidth deve ser > 0"):
            simulator.simulate(tod, tsys=100, bandwidth=0, time_constant=1.0)

        with pytest.raises(ValueError, match="bandwidth deve ser > 0"):
            simulator.simulate(tod, tsys=100, bandwidth=-1e8, time_constant=1.0)

    def test_invalid_time_constant_raises(self):
        """Rejeita time_constant <= 0."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.ones((10, 5))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)

        with pytest.raises(ValueError, match="time_constant deve ser > 0"):
            simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=0)

        with pytest.raises(ValueError, match="time_constant deve ser > 0"):
            simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=-1.0)


class TestNoiseSimulatorScaling:
    """Testes de escalamento dos parâmetros."""

    def test_noise_std_scales_with_tsys(self):
        """Ruído aumenta com T_sys."""
        time = np.arange(100)
        freq = np.linspace(1e9, 2e9, 50)
        power = np.zeros((100, 50))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator1 = NoiseSimulator(rng_seed=42)
        result1 = simulator1.simulate(tod.copy(), tsys=50, bandwidth=1e8, time_constant=1.0)
        std1 = result1["power"].std().item()

        simulator2 = NoiseSimulator(rng_seed=42)
        result2 = simulator2.simulate(tod.copy(), tsys=100, bandwidth=1e8, time_constant=1.0)
        std2 = result2["power"].std().item()

        # T_sys dobra → desvio padrão dobra
        assert std2 > std1

    def test_noise_std_scales_inversely_with_bandwidth(self):
        """Ruído diminui com bandwidth."""
        time = np.arange(100)
        freq = np.linspace(1e9, 2e9, 50)
        power = np.zeros((100, 50))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator1 = NoiseSimulator(rng_seed=42)
        result1 = simulator1.simulate(tod.copy(), tsys=100, bandwidth=1e8, time_constant=1.0)
        std1 = result1["power"].std().item()

        simulator2 = NoiseSimulator(rng_seed=42)
        result2 = simulator2.simulate(tod.copy(), tsys=100, bandwidth=4e8, time_constant=1.0)
        std2 = result2["power"].std().item()

        # Bandwidth aumenta → desvio padrão diminui
        assert std2 < std1

    def test_noise_std_scales_inversely_with_time_constant(self):
        """Ruído diminui com time_constant."""
        time = np.arange(100)
        freq = np.linspace(1e9, 2e9, 50)
        power = np.zeros((100, 50))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator1 = NoiseSimulator(rng_seed=42)
        result1 = simulator1.simulate(tod.copy(), tsys=100, bandwidth=1e8, time_constant=0.5)
        std1 = result1["power"].std().item()

        simulator2 = NoiseSimulator(rng_seed=42)
        result2 = simulator2.simulate(tod.copy(), tsys=100, bandwidth=1e8, time_constant=2.0)
        std2 = result2["power"].std().item()

        # time_constant aumenta → desvio padrão diminui
        assert std2 < std1


class TestNoiseSimulatorEdgeCases:
    """Testes de casos extremos."""

    def test_single_time_sample(self):
        """Funciona com single time sample."""
        time = np.array([0])
        freq = np.linspace(1e9, 2e9, 50)
        power = np.ones((1, 50))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)
        result = simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

        assert result["power"].shape == (1, 50)

    def test_single_frequency(self):
        """Funciona com single frequency."""
        time = np.arange(100)
        freq = np.array([1.4e9])
        power = np.ones((100, 1))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)
        result = simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

        assert result["power"].shape == (100, 1)

    def test_large_array(self):
        """Funciona com array grande."""
        time = np.arange(10000)
        freq = np.linspace(1e9, 2e9, 1000)
        power = np.ones((10000, 1000))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)
        result = simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

        assert result["power"].shape == (10000, 1000)


class TestNoiseSimulatorIntegration:
    """Testes de integração."""

    def test_preserves_coordinates(self):
        """Preserva coordenadas do dataset."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.ones((10, 5))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)
        result = simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

        assert np.allclose(result["time"].values, time)
        assert np.allclose(result["frequency"].values, freq)

    def test_preserves_other_variables(self):
        """Preserva outras variáveis do dataset."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.ones((10, 5))
        flags = np.zeros((10, 5), dtype=bool)

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
                "flags": (("time", "frequency"), flags),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)
        result = simulator.simulate(tod, tsys=100, bandwidth=1e8, time_constant=1.0)

        assert "flags" in result.data_vars
        assert np.allclose(result["flags"].values, flags)

    def test_multiple_simulations_independent(self):
        """Múltiplas simulações são independentes."""
        time = np.arange(100)
        freq = np.linspace(1e9, 2e9, 50)
        power = np.ones((100, 50))

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        simulator = NoiseSimulator(rng_seed=42)

        result1 = simulator.simulate(tod.copy(), tsys=100, bandwidth=1e8, time_constant=1.0)
        result2 = simulator.simulate(tod.copy(), tsys=100, bandwidth=1e8, time_constant=1.0)

        # Diferentes chamadas devem produzir ruído diferente
        # (porque o RNG foi avançado entre as chamadas)
        assert not np.allclose(result1["power"].values, result2["power"].values)

    def test_realistic_workflow(self):
        """Workflow realista: criar TOD, simular ruído, validar."""
        # Parâmetros realistas
        ntime = 1000
        nfreq = 64
        time = np.arange(ntime)
        freq = np.linspace(1.4e9, 1.5e9, nfreq)

        # TOD sem ruído
        power = np.ones((ntime, nfreq)) * 10.0  # 10 mJy/beam

        tod = xr.Dataset(
            {
                "power": (("time", "frequency"), power),
            },
            coords={"time": time, "frequency": freq},
        )

        # Simular com parâmetros realistas
        # Tsys = 30 K (bom receptor de radiotelescópio)
        # Bandwidth = 10 MHz
        # Integration time = 1 s
        simulator = NoiseSimulator(rng_seed=12345)
        tod_noisy = simulator.simulate(tod, tsys=30.0, bandwidth=10e6, time_constant=1.0)

        # Validar
        noise = tod_noisy["power"] - tod["power"]
        bandwidth_ref = 1e8  # Normalização padrão
        expected_std = 30.0 / np.sqrt((10e6 / bandwidth_ref) * 1.0)

        # Tolerância maior para flutuação estatística com amostra pequena
        assert abs(noise.mean().item()) < 1.0
        assert abs(noise.std().item() - expected_std) / expected_std < 0.05
