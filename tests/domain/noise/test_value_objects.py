"""
tests/domain/noise/test_value_objects.py

Testes unitários para value objects do domínio Noise.
"""

import pytest

from radioskysims_simulation.domain.noise import (
    SystemTemperature,
    Efficiency,
    Bandwidth,
    IntegrationTime,
    InvalidNoiseParameterError,
)


class TestSystemTemperature:
    """Testes para SystemTemperature value object."""

    def test_valid_creation(self):
        """Cria SystemTemperature válida."""
        t_sys = SystemTemperature(50.0)
        assert t_sys.value == 50.0

    def test_zero_raises(self):
        """Rejeita temperatura zero."""
        with pytest.raises(InvalidNoiseParameterError):
            SystemTemperature(0.0)

    def test_negative_raises(self):
        """Rejeita temperatura negativa."""
        with pytest.raises(InvalidNoiseParameterError):
            SystemTemperature(-10.0)

    def test_type_conversion(self):
        """Converte int para float automaticamente."""
        t_sys = SystemTemperature(50)  # int
        assert isinstance(t_sys.value, float)
        assert t_sys.value == 50.0

    def test_invalid_type_raises(self):
        """Rejeita tipo inválido."""
        with pytest.raises(InvalidNoiseParameterError):
            SystemTemperature("50")

    def test_equality(self):
        """Igualdade baseada em valor."""
        t1 = SystemTemperature(50.0)
        t2 = SystemTemperature(50.0)
        assert t1 == t2

    def test_inequality(self):
        """Desigualdade para valores diferentes."""
        t1 = SystemTemperature(50.0)
        t2 = SystemTemperature(60.0)
        assert t1 != t2

    def test_hash(self):
        """Hash funciona para uso em sets/dicts."""
        t1 = SystemTemperature(50.0)
        t2 = SystemTemperature(50.0)
        assert hash(t1) == hash(t2)

    def test_str_representation(self):
        """String amigável."""
        t_sys = SystemTemperature(50.0)
        assert "50" in str(t_sys)
        assert "K" in str(t_sys)


class TestEfficiency:
    """Testes para Efficiency value object."""

    def test_valid_creation(self):
        """Cria Efficiency válida."""
        eff = Efficiency(0.95)
        assert eff.value == 0.95

    def test_unity_valid(self):
        """Eficiência 1.0 é válida."""
        eff = Efficiency(1.0)
        assert eff.value == 1.0

    def test_zero_raises(self):
        """Rejeita eficiência zero."""
        with pytest.raises(InvalidNoiseParameterError):
            Efficiency(0.0)

    def test_negative_raises(self):
        """Rejeita eficiência negativa."""
        with pytest.raises(InvalidNoiseParameterError):
            Efficiency(-0.1)

    def test_greater_than_one_raises(self):
        """Rejeita eficiência > 1."""
        with pytest.raises(InvalidNoiseParameterError):
            Efficiency(1.5)

    def test_equality(self):
        """Igualdade baseada em valor."""
        e1 = Efficiency(0.95)
        e2 = Efficiency(0.95)
        assert e1 == e2

    def test_str_percentual(self):
        """String em percentual."""
        eff = Efficiency(0.95)
        s = str(eff)
        assert "95" in s
        assert "%" in s


class TestBandwidth:
    """Testes para Bandwidth value object."""

    def test_valid_creation(self):
        """Cria Bandwidth válida."""
        bw = Bandwidth(1e6)  # 1 MHz
        assert bw.value == 1e6

    def test_zero_raises(self):
        """Rejeita bandwidth zero."""
        with pytest.raises(InvalidNoiseParameterError):
            Bandwidth(0.0)

    def test_negative_raises(self):
        """Rejeita bandwidth negativa."""
        with pytest.raises(InvalidNoiseParameterError):
            Bandwidth(-1e6)

    def test_conversions(self):
        """Conversões de unidade funcionam."""
        bw = Bandwidth(1e6)  # 1 MHz
        assert bw.in_mhz() == pytest.approx(1.0)
        assert bw.in_ghz() == pytest.approx(1e-3)

    def test_equality(self):
        """Igualdade baseada em valor."""
        bw1 = Bandwidth(1e6)
        bw2 = Bandwidth(1e6)
        assert bw1 == bw2

    def test_str_formatting(self):
        """String com formatação apropriada."""
        bw_mhz = Bandwidth(1e6)
        bw_ghz = Bandwidth(1e9)
        
        assert "MHz" in str(bw_mhz)
        assert "GHz" in str(bw_ghz)


class TestIntegrationTime:
    """Testes para IntegrationTime value object."""

    def test_valid_creation(self):
        """Cria IntegrationTime válida."""
        t_int = IntegrationTime(1.0)
        assert t_int.value == 1.0

    def test_zero_raises(self):
        """Rejeita tempo zero."""
        with pytest.raises(InvalidNoiseParameterError):
            IntegrationTime(0.0)

    def test_negative_raises(self):
        """Rejeita tempo negativo."""
        with pytest.raises(InvalidNoiseParameterError):
            IntegrationTime(-1.0)

    def test_conversions(self):
        """Conversões de unidade funcionam."""
        t_int = IntegrationTime(1.0)  # 1 segundo
        assert t_int.in_milliseconds() == pytest.approx(1000.0)
        assert t_int.in_microseconds() == pytest.approx(1e6)

    def test_equality(self):
        """Igualdade baseada em valor."""
        t1 = IntegrationTime(1.0)
        t2 = IntegrationTime(1.0)
        assert t1 == t2

    def test_str_formatting(self):
        """String com formatação apropriada."""
        t_sec = IntegrationTime(1.0)
        t_ms = IntegrationTime(0.001)
        
        assert "s" in str(t_sec)
        assert "ms" in str(t_ms)
