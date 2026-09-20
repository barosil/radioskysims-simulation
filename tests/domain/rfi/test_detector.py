"""
tests/domain/rfi/test_detector.py

Testes unitários para RFIDetector.
"""

import pytest
import numpy as np
import xarray as xr

from radioskysims_simulation.domain.rfi import RFIDetector, RFIEvent


class TestRFIDetectorInitialization:
    """Testes de inicialização do RFIDetector."""

    def test_valid_initialization(self):
        """Cria RFIDetector válido com threshold padrão."""
        detector = RFIDetector()
        assert detector.threshold_sigma == 5.0

    def test_custom_threshold(self):
        """Aceita threshold customizado."""
        detector = RFIDetector(threshold_sigma=3.0)
        assert detector.threshold_sigma == 3.0

    def test_negative_threshold_raises(self):
        """Rejeita threshold negativo."""
        with pytest.raises(ValueError, match="threshold_sigma deve ser > 0"):
            RFIDetector(threshold_sigma=-1.0)

    def test_zero_threshold_raises(self):
        """Rejeita threshold zero."""
        with pytest.raises(ValueError, match="threshold_sigma deve ser > 0"):
            RFIDetector(threshold_sigma=0.0)


class TestRFIDetectorBasic:
    """Testes de funcionalidade básica do detector."""

    def test_detect_smoke(self):
        """Teste smoke: detecta RFI em dados sintéticos."""
        time = np.arange(100)
        freq = np.linspace(1e9, 2e9, 50)
        power = np.random.normal(1.0, 0.1, (100, 50))

        # Injetar RFI (10x amplificado)
        power[50:60, 30:35] *= 10

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector(threshold_sigma=3.0)
        events = detector.detect(tod)

        assert len(events) > 0
        assert events[0].severity > 3.0

    def test_detect_clean_data(self):
        """Retorna poucos eventos para dados com ruído pequeno."""
        np.random.seed(42)  # Para reprodutibilidade
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 10)
        power = np.random.normal(1.0, 0.005, (10, 10))  # Variância muito baixa

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector(threshold_sigma=10.0)  # Threshold bem alto
        events = detector.detect(tod)

        # Com threshold extremamente alto, esperamos poucos ou nenhum evento
        assert len(events) < 3

    def test_detect_returns_list(self):
        """Retorna list[RFIEvent]."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.random.normal(1.0, 0.1, (10, 5))
        power[5, 2] *= 100  # Spike forte

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector(threshold_sigma=3.0)
        events = detector.detect(tod)

        assert isinstance(events, list)
        if events:
            assert isinstance(events[0], RFIEvent)


class TestRFIDetectorProperties:
    """Testes de propriedades dos eventos detectados."""

    def test_event_has_required_fields(self):
        """RFIEvent tem todos os campos obrigatórios."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.random.normal(1.0, 0.1, (10, 5))
        power[5, 2] *= 100

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector(threshold_sigma=3.0)
        events = detector.detect(tod)

        if events:
            event = events[0]
            assert hasattr(event, 'time_idx')
            assert hasattr(event, 'freq_idx')
            assert hasattr(event, 'power')
            assert hasattr(event, 'severity')

    def test_event_indices_valid(self):
        """Índices dos eventos são válidos."""
        n_time, n_freq = 20, 15
        time = np.arange(n_time)
        freq = np.linspace(1e9, 2e9, n_freq)
        power = np.random.normal(1.0, 0.1, (n_time, n_freq))
        power[10:12, 7:9] *= 50

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector(threshold_sigma=3.0)
        events = detector.detect(tod)

        for event in events:
            assert 0 <= event.time_idx < n_time
            assert 0 <= event.freq_idx < n_freq

    def test_severity_threshold(self):
        """Todos os eventos têm severity > threshold."""
        time = np.arange(50)
        freq = np.linspace(1e9, 2e9, 25)
        power = np.random.normal(1.0, 0.1, (50, 25))
        power[25:30, 12:15] *= 20

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        threshold = 4.0
        detector = RFIDetector(threshold_sigma=threshold)
        events = detector.detect(tod)

        for event in events:
            assert event.severity > threshold


class TestRFIDetectorOrdering:
    """Testes de ordenação dos eventos."""

    def test_events_sorted_by_severity(self):
        """Eventos são ordenados por severidade (descendente)."""
        time = np.arange(100)
        freq = np.linspace(1e9, 2e9, 50)
        power = np.random.normal(1.0, 0.1, (100, 50))

        # RFI forte
        power[50:52, 25:27] *= 100
        # RFI fraca
        power[75:76, 35] *= 20

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector(threshold_sigma=3.0)
        events = detector.detect(tod)

        # Verificar se está ordenado
        for i in range(len(events) - 1):
            assert events[i].severity >= events[i + 1].severity


class TestRFIDetectorInput:
    """Testes de validação de entrada."""

    def test_missing_power_raises(self):
        """Levanta erro se 'power' não está no dataset."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        
        tod = xr.Dataset({
            'wrong_name': (('time', 'frequency'), np.ones((10, 5))),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector()
        with pytest.raises(KeyError, match="power"):
            detector.detect(tod)

    def test_wrong_dimensions_raises(self):
        """Levanta erro se power não é 2D."""
        time = np.arange(10)
        
        tod = xr.Dataset({
            'power': (('time',), np.ones(10)),
        }, coords={'time': time})

        detector = RFIDetector()
        with pytest.raises(ValueError, match="2D"):
            detector.detect(tod)

    def test_empty_dataset_handled(self):
        """Lida com dataset vazio."""
        time = np.arange(0)
        freq = np.arange(0)
        power = np.empty((0, 0))

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector()
        # Deve retornar lista vazia sem levanta erro
        events = detector.detect(tod)
        assert events == []


class TestRFIDetectorReferenceSpectrum:
    """Testes com espectro de referência customizado."""

    def test_detect_with_custom_reference(self):
        """Aceita espectro de referência customizado."""
        time = np.arange(20)
        freq = np.linspace(1e9, 2e9, 10)
        power = np.random.normal(1.0, 0.1, (20, 10))
        power[10, 5] *= 50

        # Espectro de referência customizado (aproximadamente a média)
        reference = np.mean(power, axis=0)

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector(threshold_sigma=3.0)
        events = detector.detect(tod, reference_spectrum=reference)

        assert isinstance(events, list)

    def test_reference_spectrum_array(self):
        """Aceita reference como numpy array."""
        time = np.arange(20)
        freq = np.linspace(1e9, 2e9, 10)
        power = np.random.normal(1.0, 0.1, (20, 10))
        power[10, 5] *= 50

        reference = np.ones(10)

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector(threshold_sigma=3.0)
        events = detector.detect(tod, reference_spectrum=reference)

        assert isinstance(events, list)


class TestRFIDetectorEdgeCases:
    """Testes de casos extremos."""

    def test_single_time_sample(self):
        """Lida com um único sample temporal."""
        time = np.array([0])
        freq = np.linspace(1e9, 2e9, 10)
        power = np.ones((1, 10)) * 2.0

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector(threshold_sigma=3.0)
        events = detector.detect(tod)

        # Com um sample, mediana = valor, deviation = 0
        assert len(events) == 0

    def test_single_frequency(self):
        """Lida com uma única frequência."""
        time = np.arange(10)
        freq = np.array([1e9])
        power = np.random.normal(1.0, 0.1, (10, 1))

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector()
        events = detector.detect(tod)

        assert isinstance(events, list)

    def test_uniform_data(self):
        """Lida com dados uniformes (std = 0)."""
        time = np.arange(10)
        freq = np.linspace(1e9, 2e9, 5)
        power = np.ones((10, 5))  # Completamente uniforme

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector = RFIDetector()
        events = detector.detect(tod)

        assert events == []


class TestRFIDetectorIntegration:
    """Testes de integração e casos realistas."""

    def test_realistic_workflow(self):
        """Simula um fluxo de detecção realista."""
        # Setup realista
        n_time = 1000
        n_freq = 100
        
        time = np.arange(n_time)
        freq = np.linspace(1.4e9, 1.8e9, n_freq)  # L-band
        
        # Dados com ruído gaussiano
        power = np.random.normal(1.0, 0.1, (n_time, n_freq))
        
        # RFI #1: RFI broadband contínua em parte do tempo
        power[100:200, 40:60] *= 5
        
        # RFI #2: RFI narrowband pontual
        power[500:510, 70] *= 50
        
        # RFI #3: RFI intermitente
        power[600::100, 80:85] *= 15
        
        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})
        
        # Detecção com threshold sensível
        detector = RFIDetector(threshold_sigma=3.0)
        events = detector.detect(tod)
        
        # Validações
        assert len(events) > 0
        assert all(e.severity > 3.0 for e in events)
        assert all(e.time_idx < n_time and e.freq_idx < n_freq for e in events)

    def test_multiple_detectors_independence(self):
        """Múltiplos detectores são independentes."""
        time = np.arange(20)
        freq = np.linspace(1e9, 2e9, 10)
        power = np.random.normal(1.0, 0.1, (20, 10))
        power[10, 5] *= 50

        tod = xr.Dataset({
            'power': (('time', 'frequency'), power),
        }, coords={'time': time, 'frequency': freq})

        detector1 = RFIDetector(threshold_sigma=3.0)
        detector2 = RFIDetector(threshold_sigma=5.0)

        events1 = detector1.detect(tod)
        events2 = detector2.detect(tod)

        # Detector mais sensível encontra mais eventos
        assert len(events1) >= len(events2)
