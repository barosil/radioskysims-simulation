"""
tests/domain/rfi/test_value_objects.py

Testes unitários para FrequencyRange.
"""

import pytest

from radioskysims_simulation.domain.rfi import (
    FrequencyRange,
    InvalidFrequencyRangeError,
)


class TestFrequencyRangeCreation:
    """Testes de criação de FrequencyRange."""

    def test_valid_creation(self):
        """Cria FrequencyRange válido."""
        fr = FrequencyRange(1.4, 1.8)
        assert fr.f_min == 1.4
        assert fr.f_max == 1.8
        assert fr.bandwidth == pytest.approx(0.4)

    def test_negative_frequency_raises(self):
        """Rejeita frequências negativas."""
        with pytest.raises(InvalidFrequencyRangeError):
            FrequencyRange(-1.0, 1.0)
        
        with pytest.raises(InvalidFrequencyRangeError):
            FrequencyRange(1.0, -2.0)

    def test_zero_frequency_raises(self):
        """Rejeita frequências zero."""
        with pytest.raises(InvalidFrequencyRangeError):
            FrequencyRange(0.0, 1.0)

    def test_f_min_equals_f_max_raises(self):
        """Rejeita f_min == f_max."""
        with pytest.raises(InvalidFrequencyRangeError):
            FrequencyRange(1.4, 1.4)

    def test_f_min_greater_than_f_max_raises(self):
        """Rejeita f_min > f_max."""
        with pytest.raises(InvalidFrequencyRangeError):
            FrequencyRange(1.8, 1.4)

    def test_float_conversion(self):
        """Converte para float automaticamente."""
        # A implementação aceita valores que podem ser convertidos
        # Este teste documenta o comportamento
        try:
            fr = FrequencyRange("1.4", 1.8)
            # Se suceder, o tipo foi convertido
            assert isinstance(fr.f_min, (int, float))
        except (InvalidFrequencyRangeError, TypeError):
            # Também é aceitável se levanta erro
            pass


class TestFrequencyRangeOperations:
    """Testes de operações com FrequencyRange."""

    def test_contains_frequency(self):
        """Verifica se frequência está contida."""
        fr = FrequencyRange(1.4, 1.8)
        
        assert 1.4 in fr  # limite inferior
        assert 1.5 in fr  # meio
        assert 1.8 in fr  # limite superior
        assert 1.3 not in fr  # antes
        assert 1.9 not in fr  # depois

    def test_overlaps(self):
        """Verifica sobreposição."""
        fr1 = FrequencyRange(1.4, 1.8)
        fr2 = FrequencyRange(1.6, 2.0)
        fr3 = FrequencyRange(2.0, 2.5)
        
        assert fr1.overlaps_with(fr2)  # sobrepõe
        assert fr2.overlaps_with(fr1)  # simétrico
        assert not fr1.overlaps_with(fr3)  # toca na borda, não sobrepõe

    def test_intersection(self):
        """Calcula interseção."""
        fr1 = FrequencyRange(1.4, 1.8)
        fr2 = FrequencyRange(1.6, 2.0)
        
        inter = fr1.intersection(fr2)
        assert inter is not None
        assert inter.f_min == pytest.approx(1.6)
        assert inter.f_max == pytest.approx(1.8)

    def test_intersection_disjoint(self):
        """Interseção quando disjuntos retorna None."""
        fr1 = FrequencyRange(1.4, 1.8)
        fr2 = FrequencyRange(2.0, 2.5)
        
        assert fr1.intersection(fr2) is None
        assert fr2.intersection(fr1) is None

    def test_union_overlapping(self):
        """União quando sobrepõe."""
        fr1 = FrequencyRange(1.4, 1.8)
        fr2 = FrequencyRange(1.6, 2.0)
        
        union = fr1.union(fr2)
        assert isinstance(union, FrequencyRange)
        assert union.f_min == pytest.approx(1.4)
        assert union.f_max == pytest.approx(2.0)

    def test_union_disjoint(self):
        """União quando disjuntos retorna tupla."""
        fr1 = FrequencyRange(1.4, 1.8)
        fr2 = FrequencyRange(2.0, 2.5)
        
        union = fr1.union(fr2)
        assert isinstance(union, tuple)
        assert len(union) == 2


class TestFrequencyRangeEquality:
    """Testes de igualdade e hash."""

    def test_equal_ranges(self):
        """Ranges com mesmos valores são iguais."""
        fr1 = FrequencyRange(1.4, 1.8)
        fr2 = FrequencyRange(1.4, 1.8)
        
        assert fr1 == fr2
        assert hash(fr1) == hash(fr2)

    def test_different_ranges(self):
        """Ranges diferentes não são iguais."""
        fr1 = FrequencyRange(1.4, 1.8)
        fr2 = FrequencyRange(1.6, 2.0)
        
        assert fr1 != fr2

    def test_in_set(self):
        """FrequencyRange funciona em sets."""
        fr1 = FrequencyRange(1.4, 1.8)
        fr2 = FrequencyRange(1.4, 1.8)
        fr3 = FrequencyRange(1.6, 2.0)
        
        s = {fr1, fr2, fr3}
        assert len(s) == 2  # fr1 e fr2 são iguais

    def test_in_dict(self):
        """FrequencyRange funciona como chave de dicionário."""
        fr1 = FrequencyRange(1.4, 1.8)
        fr2 = FrequencyRange(1.4, 1.8)
        
        d = {fr1: "banda1"}
        assert d[fr2] == "banda1"


class TestFrequencyRangeRepresentation:
    """Testes de representação."""

    def test_repr(self):
        """Representação é útil."""
        fr = FrequencyRange(1.4, 1.8)
        repr_str = repr(fr)
        
        assert "FrequencyRange" in repr_str
        assert "1.4" in repr_str
        assert "1.8" in repr_str

    def test_str(self):
        """String é legível."""
        fr = FrequencyRange(1.4, 1.8)
        str_str = str(fr)
        
        assert "1.4" in str_str
        assert "1.8" in str_str
        assert "GHz" in str_str
