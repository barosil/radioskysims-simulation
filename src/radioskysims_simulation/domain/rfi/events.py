"""
domain/rfi/events.py

Eventos de domínio para RFI (Radio Frequency Interference).

Eventos de domínio representam fatos importante que ocorreram no domínio.
Permitem que outras partes da aplicação reajam a eventos do domínio
sem acoplamento direto.

Nota: Eventos não carregam lógica, apenas dados. A assinatura é importante.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DomainEvent:
    """Base para todos os eventos de domínio."""

    occurred_at: datetime = None

    def __post_init__(self) -> None:
        """Define timestamp se não fornecido."""
        if self.occurred_at is None:
            object.__setattr__(self, "occurred_at", datetime.utcnow())


@dataclass(frozen=True)
class RFIMaskCreated(DomainEvent):
    """
    Evento: Uma máscara RFI foi criada.

    Indicates:
    - Um array de frequências foi analisado
    - Uma máscara booleana foi gerada
    - RFI específica foi identificada

    Attributes:
        rfi_name: Nome da fonte RFI
        n_frequencies: Número de frequências analisadas
        n_contaminated: Número de frequências contaminadas
        contamination_ratio: Razão de contaminação (0-1)
    """

    rfi_name: str | None = None
    n_frequencies: int = 0
    n_contaminated: int = 0
    contamination_ratio: float = 0.0


@dataclass(frozen=True)
class RFIApplied(DomainEvent):
    """
    Evento: Um filtro RFI foi aplicado em TOD.

    Indicates:
    - TOD foi processado
    - Dados contaminados foram zerados/removidos
    - Filtro completou com sucesso

    Attributes:
        rfi_name: Nome da fonte RFI
        tod_shape: Shape original do TOD
        contaminated_frequencies: Número de frequências contaminadas
        total_bandwidth_ghz: Largura de banda contaminada (GHz)
    """

    rfi_name: str | None = None
    tod_shape: dict[str, int] | None = None
    contaminated_frequencies: int = 0
    total_bandwidth_ghz: float = 0.0


@dataclass(frozen=True)
class RFICombined(DomainEvent):
    """
    Evento: Múltiplas RFIs foram combinadas.

    Indicates:
    - Várias fontes de RFI foram mescladas
    - Intervalos sobrepostos foram resolvidos
    - Nova RFI composta foi criada

    Attributes:
        n_sources: Número de RFIs combinadas
        resulting_ranges: Número de intervalos no resultado
        total_bandwidth_ghz: Largura de banda total
    """

    n_sources: int = 0
    resulting_ranges: int = 0
    total_bandwidth_ghz: float = 0.0


@dataclass(frozen=True)
class TODValidated(DomainEvent):
    """
    Evento: TOD foi validado após aplicação de RFI.

    Indicates:
    - Validações de integridade passaram
    - TOD está pronto para próximas etapas

    Attributes:
        validation_passed: Todos os testes passaram
        shape_preserved: Shape foi preservado
        no_new_nans: Sem novos NaNs introduzidos
    """

    validation_passed: bool = False
    shape_preserved: bool = False
    no_new_nans: bool = False
