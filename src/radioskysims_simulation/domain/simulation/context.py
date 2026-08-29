"""
Módulo: src/radioskysims/domain/simulation/context.py

Descrição:
    Contexto completo de simulação, agrupando fontes celestes,
    intervalo temporal e radiotelescópio em uma única estrutura.

    ``SimulationContext`` é o contrato principal entre a camada
    de aplicação (que monta as dependências) e o domínio (que
    executa a simulação física).

Copyright BINGO Collaboration
Last modified: 2026-07-28
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass


class SimulationContext(BaseModel):
    """Contexto completo de uma simulação.

    Reúne todas as informações necessárias para executar o pipeline
    de simulação: fontes celestes, intervalo temporal, radiotelescópio
    (com antenas, beams, pointing) e observador.

    Attributes:
        sources: Lista de fontes no céu (satélites, Sol, NVSS, ...).
        time_range: Intervalo temporal da simulação.
        radiotelescope: Especificação do radiotelescópio (antenas,
            beams, backends).
        observer: Observador terrestre (derivado do radiotelescope).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sources: list = Field(..., description="Fontes celestes (Source).")
    time_range: object = Field(..., description="Intervalo temporal (TimeRange).")
    radiotelescope: object = Field(
        ..., description="Especificação do radiotelescópio (RadiotelescopeSpec)."
    )

    @property
    def observer(self):
        """Observador terrestre, extraído do radiotelescópio."""
        return self.radiotelescope.observer

    @property
    def antennas(self):
        """Antenas do radiotelescópio."""
        return self.radiotelescope.antennas

    def sort_sources_by_ra(self) -> None:
        """Ordena as fontes por Ascensão Reta média (in-place).

        Fontes sem RA vão para o final da lista.
        """
        from radioskysims_sources.domain.source.models import Source as SourceModel

        def _ra_key(s: object) -> float:
            if isinstance(s, SourceModel):
                ra = s.ra_mean_deg
                if ra is not None:
                    return ra
            # Fallback: tenta acessar tracks diretamente
            try:
                tracks = getattr(s, "tracks", None)
                if tracks is not None and "ra" in tracks:
                    import numpy as np

                    return float(np.rad2deg(tracks["ra"].mean().values))
            except Exception:
                pass
            return float("inf")

        self.sources.sort(key=_ra_key)
