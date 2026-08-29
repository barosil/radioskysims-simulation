"""
Pacote de domínio para RFI (Radio Frequency Interference).

Modela fontes de interferência de radiofrequência e fornece operações
para detectar e filtrar RFI em Time-Ordered Data (TOD).

Estrutura:
    - models.py: Entidade RFI
    - value_objects.py: FrequencyRange (imutável)
    - exceptions.py: Exceções de domínio
    - services.py: Serviços de domínio (combinações, aplicações)
    - events.py: Eventos de domínio (sem lógica)

Exemplo:
    >>> from radioskysims_simulation.domain.rfi import RFI, FrequencyRange
    >>> fr1 = FrequencyRange(1.2, 1.4)
    >>> fr2 = FrequencyRange(1.6, 1.8)
    >>> rfi = RFI([fr1, fr2], name="GSM Band")
    >>> is_contaminated = rfi.is_contaminated(1.3)  # True
    >>> mask = rfi.create_mask(frequencies)
    >>> filtered_tod = rfi.apply_rfi_filter(tod, frequencies)
"""

from __future__ import annotations

from .detector import RFIDetector, RFIEvent
from .events import (
    RFIApplied,
    RFICombined,
    RFIMaskCreated,
    TODValidated,
)
from .exceptions import (
    InvalidFrequencyRangeError,
    InvalidRFIError,
    RFIApplicationError,
    RFICombinationError,
    RFIDomainError,
)
from .models import RFI
from .services import RFIService
from .value_objects import FrequencyRange

__all__ = [
    "RFI",
    "FrequencyRange",
    "InvalidFrequencyRangeError",
    "InvalidRFIError",
    "RFIApplicationError",
    "RFIApplied",
    "RFICombinationError",
    "RFICombined",
    "RFIDetector",
    "RFIDomainError",
    "RFIEvent",
    "RFIMaskCreated",
    "RFIService",
    "TODValidated",
]
