"""\
Módulo: src/radioskysims/application/ports/simulations.py

Contrato (port) do repositório de simulações. Permite persistir e
recuperar resumos de simulações executadas pelo ``SimOrchestrator``,
incluindo o caminho do dataset salvo e o estado de registro.

Copyright BINGO Collaboration
Last modified: 2026-08-03
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class SimulationRepositoryPort(Protocol):
    """Port para persistência/acesso de resumos de simulação.

    O ``result`` aceito por ``save`` é um objeto que expõe
    ``summary()`` (dict) e ``to_json()`` (str) — tipicamente um
    ``SimulationOrchestratorResult``.
    """

    def save(
        self,
        result: Any,
        *,
        filepath: str | None = None,
        saved: bool = False,
        registered: bool = True,
    ) -> int:
        """Persiste o resumo de uma simulação e retorna o id do registro.

        Parameters
        ----------
        result:
            Objeto com ``summary()``/``to_json()`` (SimulationOrchestratorResult).
        filepath:
            Caminho do dataset persistido (se aplicável).
        saved:
            Marca se o dataset foi salvo em arquivo.
        registered:
            Marca o registro como completo.
        """
        ...

    def get(self, sim_id: int) -> dict[str, Any] | None:
        """Retorna o resumo completo (dict) de uma simulação por id."""
        ...

    def list(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """Lista resumos de simulações (mais recentes primeiro)."""
        ...

    def search(
        self,
        *,
        created_from: datetime | str | None = None,
        created_to: datetime | str | None = None,
        time_start: datetime | str | None = None,
        time_end: datetime | str | None = None,
        telescope_name: str | None = None,
        frequency_min_mhz: float | None = None,
        frequency_max_mhz: float | None = None,
        saved: bool | None = None,
        registered: bool | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Busca registros por metadados sem carregar dataset persistido."""
        ...

    def latest(self) -> dict[str, Any] | None:
        """Retorna o resumo da simulação mais recente."""
        ...

    def load(self, sim_id: int, *, load_dataset: bool = True) -> Any | None:
        """Carrega o resultado de uma simulação e o define como item corrente.

        Recompoe um ``SimulationOrchestratorResult`` a partir do resumo do banco
        e do dataset persistido em arquivo (HDF5/Zarr). Disponível em
        ``repo.current`` até ``flush()``.
        """
        ...

    def flush(self) -> None:
        """Descarta o item corrente da memória (libera RAM).

        Não remove o registro do banco nem o arquivo persistido.
        """
        ...

    def update_status(
        self,
        sim_id: int,
        *,
        saved: bool | None = None,
        registered: bool | None = None,
        filepath: str | None = None,
    ) -> bool:
        """Atualiza o estado de persistência de um registro."""
        ...

    def delete(
        self,
        sim_id: int | None = None,
        *,
        by_id: int | None = None,
        delete_file: bool = False,
    ) -> bool:
        """Remove um registro de simulação (por ``sim_id`` ou ``by_id``).

        Parameters
        ----------
        sim_id:
            Id do registro (posicional ou keyword).
        by_id:
            Alternativa de nome ao parâmetro ``sim_id``.
        delete_file:
            Se ``True``, também remove o arquivo persistido associado.
        """
        ...
