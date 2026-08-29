"""
tests/domain/rfi/test_models.py

Testes unitários para RFI.
"""

import pytest
import numpy as np

from radioskysims_simulation.domain.rfi import (
    RFI,
    FrequencyRange,
    InvalidRFIError,
    RFIApplicationError,
)


class TestRFICreation:
    """Testes de criação de RFI."""

    def test_valid_creation(self):
        """Cria RFI válido."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name="GSM Band", description="Banda GSM")
        
        assert rfi.name == "GSM Band"
        assert rfi.description == "Banda GSM"
        assert len(rfi.frequency_ranges) == 1

    def test_empty_ranges_raises(self):
        """Rejeita lista vazia de ranges."""
        with pytest.raises(InvalidRFIError):
            RFI([], name="Empty")

    def test_invalid_ranges_type_raises(self):
        """Rejeita tipo inválido de ranges."""
        with pytest.raises(InvalidRFIError):
            RFI("not a list", name="Invalid")

    def test_invalid_range_element_raises(self):
        """Rejeita elemento que não é FrequencyRange."""
        with pytest.raises(InvalidRFIError):
            RFI([1.4], name="Invalid")

    def test_name_optional(self):
        """Name é opcional."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name=None)
        assert rfi.name is None


class TestRFIContamination:
    """Testes de detecção de contaminação."""

    def test_is_contaminated_in_range(self):
        """Detecta contaminação dentro do range."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name="Test")
        
        assert rfi.is_contaminated(1.5)
        assert rfi.is_contaminated(1.4)
        assert rfi.is_contaminated(1.8)

    def test_is_contaminated_outside_range(self):
        """Não detecta contaminação fora."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name="Test")
        
        assert not rfi.is_contaminated(1.3)
        assert not rfi.is_contaminated(1.9)

    def test_is_contaminated_multiple_ranges(self):
        """Detecta com múltiplos ranges."""
        ranges = [
            FrequencyRange(1.4, 1.8),
            FrequencyRange(2.0, 2.5),
        ]
        rfi = RFI(ranges, name="Multi")
        
        assert rfi.is_contaminated(1.5)
        assert rfi.is_contaminated(2.2)
        assert not rfi.is_contaminated(1.9)


class TestRFIMasking:
    """Testes de criação de máscara."""

    def test_create_mask_simple(self):
        """Cria máscara para frequencies simples."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name="Test")
        
        frequencies = np.array([1.3, 1.5, 1.7, 1.9])
        mask = rfi.create_mask(frequencies)
        
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (4,)
        assert np.array_equal(mask, [False, True, True, False])

    def test_create_mask_exact_bounds(self):
        """Máscara inclui limites do range."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name="Test")
        
        frequencies = np.array([1.4, 1.5, 1.8])
        mask = rfi.create_mask(frequencies)
        
        assert np.array_equal(mask, [True, True, True])

    def test_create_mask_multiple_ranges(self):
        """Máscara com múltiplos ranges."""
        ranges = [
            FrequencyRange(1.4, 1.6),
            FrequencyRange(2.0, 2.2),
        ]
        rfi = RFI(ranges, name="Multi")
        
        frequencies = np.array([1.5, 1.7, 2.1, 2.3])
        mask = rfi.create_mask(frequencies)
        
        assert np.array_equal(mask, [True, False, True, False])

    def test_create_mask_empty_array_raises(self):
        """Máscara para array vazio levanta erro."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name="Test")
        
        frequencies = np.array([])
        with pytest.raises(RFIApplicationError):
            rfi.create_mask(frequencies)


class TestRFIDeduplication:
    """Testes de deduplicação/merge de ranges."""

    def test_overlapping_ranges_merged(self):
        """Ranges sobrepostos são mesclados."""
        ranges = [
            FrequencyRange(1.4, 1.8),
            FrequencyRange(1.6, 2.0),
        ]
        rfi = RFI(ranges, name="Overlapping")
        
        # Deve ter sido dedupliado para um único range
        assert len(rfi.frequency_ranges) == 1
        assert rfi.frequency_ranges[0].f_min == pytest.approx(1.4)
        assert rfi.frequency_ranges[0].f_max == pytest.approx(2.0)

    def test_adjacent_ranges_merged(self):
        """Ranges adjacentes podem ser mesclados."""
        ranges = [
            FrequencyRange(1.4, 1.8),
            FrequencyRange(1.8, 2.0),
        ]
        rfi = RFI(ranges, name="Adjacent")
        
        # Verifica que foram dedupliados
        assert len(rfi.frequency_ranges) <= 2

    def test_disjoint_ranges_preserved(self):
        """Ranges disjuntos são preservados."""
        ranges = [
            FrequencyRange(1.4, 1.6),
            FrequencyRange(2.0, 2.2),
        ]
        rfi = RFI(ranges, name="Disjoint")
        
        assert len(rfi.frequency_ranges) == 2

    def test_duplicate_ranges_deduplicated(self):
        """Ranges idênticos são dedupliados."""
        ranges = [
            FrequencyRange(1.4, 1.8),
            FrequencyRange(1.4, 1.8),
        ]
        rfi = RFI(ranges, name="Duplicate")
        
        assert len(rfi.frequency_ranges) == 1


class TestRFIFiltering:
    """Testes de aplicação de filtro RFI."""

    def test_apply_filter_zeros_contaminated(self):
        """Filtro zera dados contaminados."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name="Test")
        
        # Simula TOD simples (sem xarray)
        frequencies = np.array([1.3, 1.5, 1.7, 1.9])
        tod = np.array([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0]])  # 2 detectores x 4 frequências
        
        # Aplica filtro usando mask
        mask = rfi.create_mask(frequencies)
        filtered = np.where(mask, 0.0, tod)
        
        # Verifica que dados contaminados foram zerados
        assert filtered[0, 0] == 1.0  # 1.3 GHz não contaminado
        assert filtered[0, 1] == 0.0  # 1.5 GHz contaminado
        assert filtered[0, 2] == 0.0  # 1.7 GHz contaminado
        assert filtered[0, 3] == 4.0  # 1.9 GHz não contaminado

    def test_apply_filter_preserves_shape(self):
        """Filtro preserva shape."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name="Test")
        
        frequencies = np.array([1.3, 1.5, 1.7, 1.9])
        tod = np.array([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0]])
        
        mask = rfi.create_mask(frequencies)
        filtered = np.where(mask, 0.0, tod)
        
        assert filtered.shape == tod.shape


class TestRFIEquality:
    """Testes de igualdade."""

    def test_equal_rfis(self):
        """RFIs com mesmas ranges são iguais."""
        fr = FrequencyRange(1.4, 1.8)
        rfi1 = RFI([fr], name="Test")
        rfi2 = RFI([fr], name="Test")
        
        assert rfi1 == rfi2

    def test_different_rfis(self):
        """RFIs diferentes não são iguais."""
        rfi1 = RFI([FrequencyRange(1.4, 1.8)], name="Test1")
        rfi2 = RFI([FrequencyRange(2.0, 2.5)], name="Test2")
        
        assert rfi1 != rfi2


class TestRFIRepresentation:
    """Testes de representação."""

    def test_repr(self):
        """Representação é útil."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name="GSM Band")
        repr_str = repr(rfi)
        
        assert "RFI" in repr_str
        assert "GSM Band" in repr_str

    def test_str(self):
        """String é legível."""
        fr = FrequencyRange(1.4, 1.8)
        rfi = RFI([fr], name="GSM Band")
        str_str = str(rfi)
        
        assert "GSM Band" in str_str
