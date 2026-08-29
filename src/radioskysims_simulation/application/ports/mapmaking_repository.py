from __future__ import annotations

from typing import Protocol

from radioskysims_simulation.domain.mapmaking import (
    MapMakingResult,
)


class MapMakingRepositoryPort(Protocol):
    """
    Port para persistência de resultados de mapmaking.
    """

    def save(
        self,
        *,
        result: MapMakingResult,
        key: str,
        simulation_id: int | None = None,
        overwrite: bool = False,
    ) -> int:
        """
        Persiste um MapMakingResult e retorna seu id.
        """
        ...

    def load(
        self,
        record_id: int,
    ) -> MapMakingResult | None:
        """
        Recupera um resultado persistido.
        """
        ...

    def get_metadata(
        self,
        record_id: int,
    ) -> dict[str, object] | None:
        """
        Recupera apenas os metadados sem materializar os arrays.
        """
        ...

    def delete(
        self,
        record_id: int,
    ) -> bool:
        """
        Remove o registro e seus dados persistidos.
        """
        ...
