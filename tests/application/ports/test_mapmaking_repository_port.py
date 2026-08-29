from __future__ import annotations

from radioskysims_simulation.application.ports.mapmaking_repository import (
    MapMakingRepositoryPort,
)


def test_mapmaking_repository_port_declares_contract() -> None:
    required = {
        "save",
        "load",
        "get_metadata",
        "delete",
    }

    assert required.issubset(set(dir(MapMakingRepositoryPort)))
